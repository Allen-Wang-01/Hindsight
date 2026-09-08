/**
 * Writes an agent run's trajectory into `llm_traces`.
 *
 * One model call is one row. A turn whose stream died and was retried leaves an
 * error row for the dead attempt and an ok row for the surviving one, both
 * under the same turn_index — the same convention `report_summary` uses.
 *
 * Rows are written one turn behind: the table is append-only and `stop_reason`
 * is only known when the run ends, so the newest turn is held until either
 * another turn completes or the run finishes. A run cancelled mid-stream
 * therefore still ends with a row saying so.
 *
 * Attaches through AgentHooks, so `loop.ts` keeps no database dependency.
 */
import { randomUUID } from 'crypto'
import { PRICING_VERSION } from '../config/modelPricing.js'
import { hashInput, insertTrace } from '../lib/trace.js'
import type { TracePayload } from '../lib/trace.js'
import type { FaultPlan, FiredFault } from './faults.js'
import type { AgentHooks, AgentRunResult, TurnRecord } from './loop.js'
import { AGENT_PROMPT_HASH, AGENT_SYSTEM_PROMPT, buildEvidenceMessage } from './prompt.js'
import { AGENT_TOOLS_HASH } from './tools/definitions.js'
import type { FixtureInput } from './tools/stubs.js'

/** Shared by every row an agent run writes. */
export const AGENT_CALL_SITE = 'agent_turn'

/** `extra` on an agent trace row. Step D reads this shape. */
export interface AgentTraceExtra {
    /** 1-based attempt within the turn. Above 1 means an earlier attempt died. */
    attempt: number
    /** Which entry point started the run, e.g. 'runner' or 'worker'. */
    origin: string
    promptHash: string
    toolsHash: string
    cachedTokens?: number
    reasoningTokens?: number
    /** Characters streamed and then thrown away. Error rows only. */
    discardedChars?: number
    /** The provider's incomplete_details.reason. Final row of a truncated run only. */
    stopDetail?: string
    /** True on every row of a run that carried a fault plan. */
    faultInjection?: boolean
    /** Faults that fired while this row's call was in flight. */
    injectedFaults?: FiredFault[]
}

export interface RunTracerMeta {
    model: string
    promptVersion: string
    /** The evidence under review. Recorded on the run's first row. */
    input: FixtureInput
    /** Defaults to a fresh uuid. */
    runId?: string
    userId?: string
    recordId?: string
    /** The run's fault plan, read as it fires. */
    faults?: FaultPlan | null
    origin?: string
    /** The caller's own hooks, wrapped by the tracer's. */
    hooks?: AgentHooks
}

export interface RunTracer {
    readonly runId: string
    /** Pass this to runAgent in place of the caller's own hooks. */
    readonly hooks: AgentHooks
    /** Writes the run's final row and waits for every pending insert. */
    finish(result: AgentRunResult): Promise<void>
    /** Writes an error row for the call that threw, then waits. */
    fail(err: unknown): Promise<void>
}

function describe(err: unknown): string {
    return err instanceof Error ? err.message : String(err)
}

export function createRunTracer(meta: RunTracerMeta): RunTracer {
    const runId = meta.runId ?? randomUUID()
    const origin = meta.origin ?? 'unknown'
    const faults = meta.faults ?? null
    const outer = meta.hooks

    const writes: Promise<void>[] = []
    /** The completed turn not yet written. */
    let pending: TurnRecord | null = null
    let currentTurn = 0
    let attempt = 1
    let attemptStartedAt = Date.now()
    /** Text streamed by the attempt in flight. */
    let streamed = ''
    let inputWritten = false
    /** How many of faults.fired have already been claimed by a row. */
    let firedClaimed = 0

    /** Claims the faults that fired since the last row. */
    function claimFaults(): Partial<AgentTraceExtra> {
        if (!faults) return {}
        const fired = faults.fired.slice(firedClaimed)
        firedClaimed = faults.fired.length
        return fired.length > 0
            ? { faultInjection: true, injectedFaults: fired }
            : { faultInjection: true }
    }

    /** The input fields, on the first row of the run only. */
    function inputFields(): Partial<TracePayload> {
        if (inputWritten) return {}
        inputWritten = true
        const prompt = `${AGENT_SYSTEM_PROMPT}\n\n${buildEvidenceMessage(meta.input)}`
        return {
            input: { inputId: meta.input.id, evidence: meta.input.evidence },
            inputChars: prompt.length,
            inputHash: hashInput(prompt),
        }
    }

    function baseRow(): Omit<TracePayload, 'status'> {
        return {
            ...inputFields(),
            runId,
            callSite: AGENT_CALL_SITE,
            callType: 'completion',
            model: meta.model,
            promptVersion: meta.promptVersion,
            pricingVersion: PRICING_VERSION,
            ...(meta.userId ? { userId: meta.userId } : {}),
            ...(meta.recordId ? { recordId: meta.recordId } : {}),
        }
    }

    function baseExtra(): AgentTraceExtra {
        return {
            attempt,
            origin,
            promptHash: AGENT_PROMPT_HASH,
            toolsHash: AGENT_TOOLS_HASH,
        }
    }

    // insertTrace never throws; finish()/fail() waits for the batch.
    function write(payload: TracePayload): void {
        writes.push(insertTrace(payload))
    }

    /**
     * Writes the held turn. `result` is passed only for the row that ends the
     * run, which is the row carrying stop_reason and the conclusion.
     */
    function flushPending(result?: AgentRunResult): void {
        const turn = pending
        if (turn === null) return
        pending = null

        const extra: AgentTraceExtra = {
            ...baseExtra(),
            attempt: turn.attempts,
            cachedTokens: turn.cachedTokens,
            reasoningTokens: turn.reasoningTokens,
            ...claimFaults(),
        }
        if (result?.stopDetail) extra.stopDetail = result.stopDetail

        write({
            ...baseRow(),
            turnIndex: turn.turn,
            reasoning: turn.text,
            toolCalls: turn.toolCalls,
            inputTokens: turn.inputTokens,
            outputTokens: turn.outputTokens,
            totalTokens: turn.inputTokens + turn.outputTokens,
            ...(turn.costUsd === null ? {} : { costUsd: turn.costUsd }),
            latencyMs: turn.latencyMs,
            ...(turn.firstTokenMs === null ? {} : { firstTokenMs: turn.firstTokenMs }),
            status: 'ok',
            extra,
            ...(result
                ? {
                    stopReason: result.stopReason,
                    output: result.finalText,
                    ...(result.retrievalHits.length > 0
                        ? { retrievalHits: result.retrievalHits }
                        : {}),
                }
                : {}),
        })
    }

    /** Writes a row for one model call that died. It never gains a stop_reason. */
    function writeFailedAttempt(error: string, discardedChars: number): void {
        write({
            ...baseRow(),
            turnIndex: currentTurn,
            // The text the viewer saw before it was thrown away.
            reasoning: streamed,
            latencyMs: Date.now() - attemptStartedAt,
            status: 'error',
            error,
            extra: { ...baseExtra(), discardedChars, ...claimFaults() },
        })
    }

    const hooks: AgentHooks = {
        onTurnStart(turn) {
            currentTurn = turn
            attempt = 1
            attemptStartedAt = Date.now()
            streamed = ''
            outer?.onTurnStart?.(turn)
        },

        onTextDelta(delta) {
            streamed += delta
            outer?.onTextDelta?.(delta)
        },

        onStreamRetry(retry) {
            attempt = retry.attempt
            writeFailedAttempt(retry.reason, retry.discardedChars)
            attempt = retry.attempt + 1
            // Fires before the backoff, which is not the next attempt's latency.
            attemptStartedAt = Date.now() + retry.delayMs
            streamed = ''
            outer?.onStreamRetry?.(retry)
        },

        onTurnEnd(turn) {
            // Held until another turn completes: toolCalls is filled in after
            // this hook returns, and a run cancelled during the next model call
            // needs this turn's row to be the one carrying stop_reason.
            flushPending()
            pending = turn
            outer?.onTurnEnd?.(turn)
        },

        onToolStart(name) {
            outer?.onToolStart?.(name)
        },

        onToolEnd(call) {
            outer?.onToolEnd?.(call)
        },
    }

    async function drain(): Promise<void> {
        await Promise.all(writes)
        writes.length = 0
    }

    return {
        runId,
        hooks,

        async finish(result) {
            flushPending(result)
            await drain()
        },

        async fail(err) {
            flushPending()
            writeFailedAttempt(describe(err), streamed.length)
            await drain()
        },
    }
}
