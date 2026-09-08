/**
 * The agent loop.
 *
 * No orchestration framework: four tools and one while loop do not need one,
 * and every turn has to be accounted for at a granularity a framework would
 * hide. What this file owns is the accounting — turns, tool calls, tokens,
 * cost, latency — because step B feeds exactly that into `llm_traces` and
 * phase 5 reports it.
 *
 * The exit condition is that the model stopped asking for tools. Turn and
 * tool-call ceilings exist as budget guards, not as the intended way out;
 * when one trips, the run does not abort — it makes one final call with tools
 * withheld so the model still has to produce a conclusion from what it has.
 */
import OpenAI from 'openai'
import { computeCost } from '../config/modelPricing.js'
import { createLogger } from '../utils/logger.js'
import { injectedErrorMessage } from './faults.js'
import type { FaultPlan, FiredFault } from './faults.js'
import { AGENT_TOOLS } from './tools/definitions.js'
import { TOOL_HANDLERS } from './tools/stubs.js'
import type { FixtureInput } from './tools/stubs.js'
import { isToolError, toolError } from './tools/types.js'
import type {
    RetrievalHit,
    ToolErrorCode,
    ToolHandler,
    ToolName,
    ToolOutcome,
    ToolResultMeta,
} from './tools/types.js'
import {
    AGENT_PROMPT_VERSION,
    AGENT_SYSTEM_PROMPT,
    CONVERGE_MESSAGE,
    buildEvidenceMessage,
} from './prompt.js'

/**
 * The one SDK method this loop calls. Narrow on purpose: a test satisfies it
 * with an object literal replaying a scripted event sequence.
 */
export interface AgentClient {
    responses: {
        create(
            request: OpenAI.Responses.ResponseCreateParamsStreaming,
            options?: { signal?: AbortSignal }
        ): Promise<AsyncIterable<OpenAI.Responses.ResponseStreamEvent>>
    }
}

/**
 * The SDK default is 10 minutes — long enough for stuck requests to stall every
 * worker slot. Bounds connecting and time-to-headers only: the SDK clears the
 * timer once headers arrive, which for a stream is before generation finishes.
 * A stream stalling midway is handled in this file instead.
 */
const AGENT_REQUEST_TIMEOUT_MS = 60_000

/** Time limit on one tool execution. Exceeding it yields a `timeout` ToolError. */
export const DEFAULT_TOOL_TIMEOUT_MS = 10_000

/** Structured logging surface. Injectable so the terminal runner can silence it. */
export interface AgentLogger {
    info: (stage: string, fields?: Record<string, unknown>) => void
    warn: (stage: string, fields?: Record<string, unknown>) => void
    error: (stage: string, fields?: Record<string, unknown>) => void
}

export const SILENT_LOGGER: AgentLogger = {
    info: () => {},
    warn: () => {},
    error: () => {},
}

/** Built on first use: `new OpenAI()` throws without an API key, which would
 *  make merely importing this file fail in tests that never hit the network. */
let defaultClient: AgentClient | undefined

function getDefaultClient(): AgentClient {
    defaultClient ??= new OpenAI({ timeout: AGENT_REQUEST_TIMEOUT_MS })
    return defaultClient
}

export const DEFAULT_MODEL = 'gpt-5.6-luna'

export interface AgentBudget {
    /** Ceiling on model calls that may request tools. */
    maxTurns: number
    /** Ceiling on tool executions across the whole run. */
    maxToolCalls: number
    /**
     * Ceiling on what the run may spend before it is forced to converge. The
     * convergence call itself is billed on top, so a run can finish slightly
     * over. Unenforceable when the model has no pricing entry.
     */
    maxCostUsd: number
}

export const DEFAULT_BUDGET: AgentBudget = {
    maxTurns: 6,
    maxToolCalls: 8,
    maxCostUsd: 0.05,
}

export type StopReason =
    | 'model_done'      // the model stopped asking for tools — the intended exit
    | 'max_turns'
    | 'max_tool_calls'
    | 'max_cost'
    | 'incomplete'      // the provider cut the response short — see stopDetail
    | 'aborted'         // cancelled through the caller's signal

/** Budget stops share one exit: force a conclusion instead of failing the run. */
function isBudgetStop(reason: StopReason): boolean {
    return reason === 'max_turns' || reason === 'max_tool_calls' || reason === 'max_cost'
}

export interface ToolCallRecord {
    turn: number
    name: string
    /** Parsed arguments, or the raw string when the model emitted invalid JSON. */
    arguments: unknown
    /** False when the tool returned a ToolError. An empty result is still ok. */
    ok: boolean
    errorCode?: ToolErrorCode
    durationMs: number
    /** What the tool returned, summarised. Absent on failed and skipped calls. */
    resultMeta?: ToolResultMeta
}

export interface TurnRecord {
    turn: number
    /** The model's narration for this turn. */
    text: string
    toolCalls: ToolCallRecord[]
    inputTokens: number
    outputTokens: number
    cachedTokens: number
    /** Reasoning tokens. Already counted inside outputTokens — billed, but invisible. */
    reasoningTokens: number
    /** null when the model is absent from MODEL_PRICING. */
    costUsd: number | null
    latencyMs: number
    /** Time to the first visible token. Null when nothing was streamed. */
    firstTokenMs: number | null
    /** Model calls this turn cost, including retries. 1 when nothing went wrong.
     *  Only the surviving attempt's tokens are counted — the provider bills the
     *  abandoned ones too, so this is an undercount worth seeing. */
    attempts: number
}

export interface AgentTotals {
    turns: number
    toolCalls: number
    inputTokens: number
    outputTokens: number
    cachedTokens: number
    reasoningTokens: number
    costUsd: number | null
    latencyMs: number
}

/**
 * Progress callbacks for whoever is watching a run. Deliberately minimal and
 * terminal-shaped: step C defines the trajectory event schema, and this must
 * not quietly become that schema before it has been designed.
 */
export interface AgentHooks {
    onTurnStart?: (turn: number) => void
    onTextDelta?: (delta: string) => void
    onTurnEnd?: (turn: TurnRecord) => void
    onToolStart?: (name: string) => void
    onToolEnd?: (call: ToolCallRecord) => void
    onStreamRetry?: (retry: StreamRetry) => void
}

/**
 * A model call that died mid-stream and is about to be made again. Whoever is
 * rendering has already received deltas from the dead attempt: `discardedChars`
 * says how much to throw away before the retry starts producing text.
 */
export interface StreamRetry {
    /** The attempt that just failed, 1-based. */
    attempt: number
    reason: string
    delayMs: number
    /** Characters handed to onTextDelta by the failed attempt. */
    discardedChars: number
}

/** Backoff before each retry. Length + 1 is the attempt ceiling. */
export const DEFAULT_RETRY_DELAYS_MS = [500, 1500]

function sleep(ms: number): Promise<void> {
    return new Promise(resolve => setTimeout(resolve, ms))
}

/** ±20% around the base delay, so concurrent runs do not retry in lockstep. */
function jitter(baseMs: number): number {
    return Math.round(baseMs * (0.8 + Math.random() * 0.4))
}

export interface AgentRunResult {
    inputId: string
    model: string
    promptVersion: string
    turns: TurnRecord[]
    /** The narration from the last turn — the conclusion. */
    finalText: string
    stopReason: StopReason
    /**
     * Why the run stopped, when the reason alone is not enough. Carries the
     * API's `incomplete_details.reason` for `incomplete`; null otherwise.
     */
    stopDetail: string | null
    totals: AgentTotals
    /** Raw similarity floats from every retrieval. Trace-only; never shown to the model. */
    retrievalHits: RetrievalHit[]
    /** Faults that fired this run. Empty unless injection was on and requested. */
    injectedFaults: FiredFault[]
}

/** Items the Responses API accepts back as conversation history. */
type InputItem = OpenAI.Responses.ResponseInputItem

interface ParsedTurn {
    text: string
    functionCalls: Array<{ callId: string; name: string; rawArguments: string }>
    /** Output items to replay as history on the next turn. */
    historyItems: InputItem[]
    inputTokens: number
    outputTokens: number
    cachedTokens: number
    reasoningTokens: number
}

function parseResponse(response: OpenAI.Responses.Response): ParsedTurn {
    const parsed: ParsedTurn = {
        text: '',
        functionCalls: [],
        historyItems: [],
        inputTokens: response.usage?.input_tokens ?? 0,
        outputTokens: response.usage?.output_tokens ?? 0,
        cachedTokens: response.usage?.input_tokens_details?.cached_tokens ?? 0,
        reasoningTokens: response.usage?.output_tokens_details?.reasoning_tokens ?? 0,
    }

    for (const item of response.output) {
        // Reasoning items are replayed along with everything else, carrying the
        // encrypted payload requested via include. Without them the model loses
        // its own train of thought between turns and re-derives everything from
        // the tool output alone, which is the shortest path to concluding early.
        if (item.type === 'reasoning') {
            parsed.historyItems.push(item)
            continue
        }

        if (item.type === 'message') {
            for (const part of item.content) {
                if (part.type === 'output_text') {
                    parsed.text += part.text
                }
            }
            parsed.historyItems.push(item)
            continue
        }

        if (item.type === 'function_call') {
            parsed.functionCalls.push({
                callId: item.call_id,
                name: item.name,
                rawArguments: item.arguments,
            })
            parsed.historyItems.push(item)
        }
    }

    return parsed
}

function isKnownTool(name: string): name is ToolName {
    return name in TOOL_HANDLERS
}

function describe(err: unknown): string {
    return err instanceof Error ? err.message : String(err)
}

/**
 * Runs one tool under a time limit, converting both failure modes into the
 * ToolError the contract promises. The signal is what makes the limit real:
 * the race frees the loop, cancellation frees whatever the tool was holding.
 */
async function callTool(
    handler: ToolHandler,
    args: Record<string, unknown>,
    timeoutMs: number
): Promise<ToolOutcome> {
    const controller = new AbortController()
    let timer: ReturnType<typeof setTimeout> | undefined

    // Never rejects, so a throwing tool cannot take the run down and cannot
    // leave an unhandled rejection behind after the timeout wins the race.
    const settled: Promise<ToolOutcome> = handler(args, { signal: controller.signal }).then(
        outcome => outcome,
        (err: unknown) => ({ result: toolError('internal', `tool threw: ${describe(err)}`) })
    )

    const expired = new Promise<ToolOutcome>(resolve => {
        timer = setTimeout(() => {
            controller.abort()
            resolve({ result: toolError('timeout', `tool did not return within ${timeoutMs}ms`) })
        }, timeoutMs)
    })

    try {
        return await Promise.race([settled, expired])
    } finally {
        clearTimeout(timer)
    }
}

/** Key order is normalised so the same call written differently still matches. */
function canonicalize(value: unknown): unknown {
    if (Array.isArray(value)) return value.map(canonicalize)
    if (value !== null && typeof value === 'object') {
        const source = value as Record<string, unknown>
        return Object.fromEntries(
            Object.keys(source).sort().map(key => [key, canonicalize(source[key])])
        )
    }
    return value
}

function callKey(name: string, args: unknown): string {
    return `${name}:${JSON.stringify(canonicalize(args))}`
}

interface ModelCall {
    parsed: ParsedTurn
    firstTokenMs: number | null
    latencyMs: number
    /**
     * The provider's `incomplete_details.reason` when the response was cut
     * short (typically `max_output_tokens`), null when it completed normally.
     * A truncated response carries no `error`, so this is the only signal that
     * separates "the model finished" from "the model was interrupted".
     */
    incompleteReason: string | null
    attempts: number
}

/**
 * One streamed model call, retried when the stream dies. Streaming is not
 * decoration: without it there is a 30-second silence per turn with nothing to
 * show, and the reasoning text is meant to be watched as it arrives. Deltas go
 * to the hook; the final Response arrives on the completed event and is parsed
 * exactly as a non-streamed one.
 *
 * Retries here cover exactly one gap: the SDK stops retrying once the response
 * headers are in, so a stream that dies halfway is the caller's problem. Errors
 * raised while opening the request are left alone — the SDK has already retried
 * those, and stacking a second layer on top multiplies the attempts.
 *
 * Returns null when the run was cancelled. The half-finished response is thrown
 * away rather than reported: it has no usage and no conclusion.
 */
interface ModelCallParams {
    client: AgentClient
    request: OpenAI.Responses.ResponseCreateParamsStreaming
    hooks: AgentHooks | undefined
    retryDelaysMs: number[]
    signal: AbortSignal | undefined
    /** Turn number, so an injected interruption can be aimed at one turn. */
    turn: number
    faults: FaultPlan | null
    log: AgentLogger
}

/** Enough streamed text that the discarded half-sentence is visible. */
const INJECTED_INTERRUPT_AFTER_CHARS = 40

const INJECTED_INTERRUPT = 'connection reset by peer'

async function callModel({
    client,
    request,
    hooks,
    retryDelaysMs,
    signal,
    turn,
    faults,
    log,
}: ModelCallParams): Promise<ModelCall | null> {
    const startedAt = Date.now()

    for (let attempt = 1; ; attempt++) {
        if (signal?.aborted) return null

        // Consumed per attempt: one-shot, so the retry runs clean and recovers.
        const interrupt = faults?.takeStreamInterrupt(turn) ?? false
        if (interrupt) log.warn('fault_injected', { kind: 'stream-interrupt', turn })

        let firstTokenAt: number | null = null
        let final: OpenAI.Responses.Response | undefined
        // Tracked separately from `final`: an incomplete response and a
        // completed one are both terminal and both parseable, and the response
        // object alone does not say which event delivered it.
        let incomplete = false
        let streamedChars = 0
        let failure: string | null = null

        // Opening the stream is the SDK's layer: it has already retried what it
        // is willing to retry, so anything thrown here goes straight up. The one
        // exception is cancellation, which surfaces as an abort error.
        let stream: AsyncIterable<OpenAI.Responses.ResponseStreamEvent>
        try {
            stream = await client.responses.create(request, { signal })
        } catch (err) {
            if (signal?.aborted) return null
            throw err
        }

        try {
            for await (const event of stream) {
                if (event.type === 'response.output_text.delta') {
                    if (firstTokenAt === null) firstTokenAt = Date.now()
                    streamedChars += event.delta.length
                    hooks?.onTextDelta?.(event.delta)
                    if (interrupt && streamedChars >= INJECTED_INTERRUPT_AFTER_CHARS) {
                        throw new Error(INJECTED_INTERRUPT)
                    }
                } else if (event.type === 'response.completed') {
                    final = event.response
                } else if (event.type === 'response.incomplete') {
                    final = event.response
                    incomplete = true
                } else if (event.type === 'response.failed') {
                    final = event.response
                }
            }
            // A turn that streamed less text than the threshold still has to
            // die if one was injected for it.
            if (interrupt) throw new Error(INJECTED_INTERRUPT)

            if (!final) {
                failure = 'stream ended without a terminal response event'
            } else if (final.error) {
                failure = `model call failed: ${final.error.message}`
            }
        } catch (err) {
            failure = `stream interrupted: ${err instanceof Error ? err.message : String(err)}`
        }

        // Checked before the failure is classified: a cancelled stream ends
        // silently with no terminal event, which is indistinguishable from an
        // interruption. Without this, cancelling would buy three retries first.
        if (signal?.aborted) return null

        if (failure === null && final) {
            return {
                parsed: parseResponse(final),
                firstTokenMs: firstTokenAt === null ? null : firstTokenAt - startedAt,
                latencyMs: Date.now() - startedAt,
                // The reason is what makes the truncation actionable; without
                // it the trace records that something was cut off but not by
                // which limit. Never retried: the same ceiling would truncate
                // the retry too, at full price.
                incompleteReason: incomplete
                    ? final.incomplete_details?.reason ?? 'unknown'
                    : null,
                attempts: attempt,
            }
        }

        if (attempt > retryDelaysMs.length) {
            throw new Error(`${failure} (gave up after ${attempt} attempts)`)
        }

        const delayMs = jitter(retryDelaysMs[attempt - 1] ?? 0)
        log.warn('stream_retry', { turn, attempt, delayMs, reason: failure })
        hooks?.onStreamRetry?.({
            attempt,
            reason: failure ?? 'unknown',
            delayMs,
            discardedChars: streamedChars,
        })
        await sleep(delayMs)
    }
}

export interface RunAgentParams {
    input: FixtureInput
    model?: string
    budget?: AgentBudget
    hooks?: AgentHooks
    /**
     * Per-call output ceiling handed straight to the API. Left unset in normal
     * operation; setting it low is how the truncation scenario is reproduced
     * deterministically. Scoped to one run — never global state, or concurrent
     * runs would interfere.
     */
    maxOutputTokens?: number
    /** Defaults to the shared client. Injected by tests and by callers that
     *  need their own SDK configuration. */
    client?: AgentClient
    /** Overrides DEFAULT_RETRY_DELAYS_MS. Scoped to one run; tests pass zeros. */
    retryDelaysMs?: number[]
    /** Overrides DEFAULT_TOOL_TIMEOUT_MS. Scoped to one run. */
    toolTimeoutMs?: number
    /**
     * Cancellation. Aborting stops the loop at the next checkpoint and returns
     * a normal result with stopReason 'aborted' — no throw, because the caller
     * still needs the bill for what was already spent.
     */
    signal?: AbortSignal
    /** Built by createFaultPlan, which owns the gate. Null when injection is off. */
    faults?: FaultPlan | null
    logger?: AgentLogger
}

export async function runAgent({
    input,
    model = DEFAULT_MODEL,
    budget = DEFAULT_BUDGET,
    hooks,
    maxOutputTokens,
    client = getDefaultClient(),
    retryDelaysMs = DEFAULT_RETRY_DELAYS_MS,
    toolTimeoutMs = DEFAULT_TOOL_TIMEOUT_MS,
    signal,
    faults = null,
    logger,
}: RunAgentParams): Promise<AgentRunResult> {
    const startedAt = Date.now()
    const log = logger ?? createLogger('agent')

    const conversation: InputItem[] = [
        { role: 'developer', content: AGENT_SYSTEM_PROMPT },
        { role: 'user', content: buildEvidenceMessage(input) },
    ]

    const turns: TurnRecord[] = []
    const retrievalHits: RetrievalHit[] = []
    /** Call key to the turn that first made it, for duplicate detection. */
    const seenCalls = new Map<string, number>()
    let toolCallCount = 0
    let stopReason: StopReason = 'model_done'
    let stopDetail: string | null = null
    let turnIndex = 0
    /** Null once any turn comes back unpriced — the ceiling is then unenforceable. */
    let spentUsd: number | null = 0

    log.info('run_started', { inputId: input.id, model, promptVersion: AGENT_PROMPT_VERSION })

    // Spread into every request so the ceiling applies to the convergence call
    // too — a truncated convergence is the same failure as a truncated turn.
    const outputCeiling = maxOutputTokens === undefined
        ? {}
        : { max_output_tokens: maxOutputTokens }

    while (true) {
        // Cancellation checkpoint. First, so a run cancelled between turns does
        // not pay for one more.
        if (signal?.aborted) {
            stopReason = 'aborted'
            break
        }

        // Budget checks come before the call, so an exhausted budget costs one
        // convergence call rather than a wasted round trip with tools attached.
        if (turnIndex >= budget.maxTurns) {
            stopReason = 'max_turns'
            break
        }
        if (toolCallCount >= budget.maxToolCalls) {
            stopReason = 'max_tool_calls'
            break
        }
        if (spentUsd !== null && spentUsd >= budget.maxCostUsd) {
            stopReason = 'max_cost'
            break
        }

        turnIndex += 1
        hooks?.onTurnStart?.(turnIndex)

        const call = await callModel({
            client,
            request: {
                model,
                input: conversation,
                tools: AGENT_TOOLS,
                parallel_tool_calls: false,
                include: ['reasoning.encrypted_content'],
                store: false,
                stream: true,
                ...outputCeiling,
            },
            hooks,
            retryDelaysMs,
            signal,
            turn: turnIndex,
            faults,
            log,
        })

        // Cancelled mid-stream. The turn is dropped: no usage, no conclusion.
        if (call === null) {
            stopReason = 'aborted'
            break
        }

        const { parsed, firstTokenMs, latencyMs, incompleteReason, attempts } = call

        conversation.push(...parsed.historyItems)

        const turn: TurnRecord = {
            turn: turnIndex,
            text: parsed.text,
            toolCalls: [],
            inputTokens: parsed.inputTokens,
            outputTokens: parsed.outputTokens,
            cachedTokens: parsed.cachedTokens,
            reasoningTokens: parsed.reasoningTokens,
            costUsd: computeCost({
                model,
                inputTokens: parsed.inputTokens,
                outputTokens: parsed.outputTokens,
                cachedTokens: parsed.cachedTokens,
            }),
            latencyMs,
            firstTokenMs,
            attempts,
        }
        turns.push(turn)
        if (turn.costUsd === null) {
            if (spentUsd !== null) log.warn('cost_ceiling_unenforceable', { model })
            spentUsd = null
        } else if (spentUsd !== null) {
            spentUsd += turn.costUsd
        }
        hooks?.onTurnEnd?.(turn)

        // Checked before the tool loop: a truncated turn may have been cut off
        // mid-argument. No convergence call follows — the ceiling that was hit
        // would truncate that one too.
        if (incompleteReason !== null) {
            stopReason = 'incomplete'
            stopDetail = incompleteReason
            break
        }

        // The intended exit: nothing more to look up.
        if (parsed.functionCalls.length === 0) {
            stopReason = 'model_done'
            break
        }

        for (const call of parsed.functionCalls) {
            // Cancellation checkpoint. A tool that has not started is not worth
            // starting; one already running is left to finish, since the stubs
            // are local and the real queries will be short.
            if (signal?.aborted) {
                stopReason = 'aborted'
                break
            }

            hooks?.onToolStart?.(call.name)
            const callStartedAt = Date.now()
            let parsedArguments: unknown = call.rawArguments
            let output: unknown
            let ok = false
            let errorCode: ToolErrorCode | undefined
            let resultMeta: ToolResultMeta | undefined

            try {
                parsedArguments = JSON.parse(call.rawArguments) as unknown
            } catch {
                // Left as the raw string so the trace records what was actually
                // emitted — this is one of the failure modes phase 5 counts.
                output = toolError('invalid_argument', 'arguments were not valid JSON')
                errorCode = 'invalid_argument'
            }

            if (output === undefined && !isKnownTool(call.name)) {
                output = toolError('not_found', `no tool named ${call.name}`)
                errorCode = 'not_found'
            }

            if (output === undefined && isKnownTool(call.name)) {
                const key = callKey(call.name, parsedArguments)
                const firstSeenTurn = seenCalls.get(key)

                if (firstSeenTurn !== undefined) {
                    // Told to the model rather than left for the call ceiling to
                    // absorb, so it can react instead of just running out.
                    output = toolError(
                        'duplicate_call',
                        `you already called ${call.name} with these arguments on turn ${firstSeenTurn}; ` +
                        'use that result rather than repeating the call'
                    )
                    errorCode = 'duplicate_call'
                    log.warn('duplicate_tool_call', { tool: call.name, turn: turnIndex, firstSeenTurn })
                } else {
                    const fault = faults?.takeToolFault(call.name, turnIndex) ?? null
                    if (fault) {
                        log.warn('fault_injected', {
                            kind: fault.kind,
                            tool: call.name,
                            turn: turnIndex,
                        })
                    }

                    if (fault?.kind === 'tool-error') {
                        const code = fault.code ?? 'internal'
                        output = toolError(code, injectedErrorMessage(code, call.name))
                        errorCode = code
                    } else {
                        // An injected timeout hangs the handler and lets the real
                        // timeout path end it, rather than faking the error.
                        const handler: ToolHandler = fault?.kind === 'tool-timeout'
                            ? () => new Promise<ToolOutcome>(() => {})
                            : TOOL_HANDLERS[call.name]

                        const outcome = await callTool(
                            handler,
                            parsedArguments as Record<string, unknown>,
                            toolTimeoutMs
                        )
                        output = outcome.result
                        resultMeta = outcome.meta
                        if (outcome.hits) {
                            retrievalHits.push(...outcome.hits)
                        }
                        if (isToolError(outcome.result)) {
                            errorCode = outcome.result.error.code
                        } else {
                            ok = true
                        }
                    }

                    // Only successful calls are remembered. A call that failed
                    // has no result to reuse and is the one worth retrying.
                    if (ok) seenCalls.set(key, turnIndex)
                }
            }

            if (errorCode) {
                log.warn('tool_failed', { tool: call.name, turn: turnIndex, code: errorCode })
            }

            toolCallCount += 1
            const record: ToolCallRecord = {
                turn: turnIndex,
                name: call.name,
                arguments: parsedArguments,
                ok,
                ...(errorCode ? { errorCode } : {}),
                durationMs: Date.now() - callStartedAt,
                ...(resultMeta ? { resultMeta } : {}),
            }
            turn.toolCalls.push(record)
            hooks?.onToolEnd?.(record)

            conversation.push({
                type: 'function_call_output',
                call_id: call.callId,
                output: JSON.stringify(output),
            })
        }

        // Carries the tool-loop break outward.
        if (stopReason === 'aborted') break
    }

    // Cancellation checkpoint. A cancelled run gets no convergence call — the
    // point of cancelling is that the spending stops.
    if (signal?.aborted) {
        stopReason = 'aborted'
    }

    // Forced convergence. Tools are withheld rather than merely discouraged, so
    // the model cannot spend another call instead of answering.
    if (isBudgetStop(stopReason)) {
        log.info('budget_reached', { stopReason, turns: turns.length, toolCalls: toolCallCount })
        conversation.push({ role: 'user', content: CONVERGE_MESSAGE })
        hooks?.onTurnStart?.(turnIndex + 1)

        const call = await callModel({
            client,
            request: {
                model,
                input: conversation,
                include: ['reasoning.encrypted_content'],
                store: false,
                stream: true,
                ...outputCeiling,
            },
            hooks,
            retryDelaysMs,
            signal,
            turn: turnIndex + 1,
            faults,
            log,
        })

        if (call === null) {
            // Cancelled while converging: the budget reason is replaced, since
            // what actually ended the run is the cancellation.
            stopReason = 'aborted'
        } else {
            const { parsed, firstTokenMs, latencyMs, incompleteReason, attempts } = call

            // A truncated conclusion is not a converged one. The budget reason
            // is overwritten because this is the more serious of the two.
            if (incompleteReason !== null) {
                stopReason = 'incomplete'
                stopDetail = incompleteReason
            }

            const turn: TurnRecord = {
                turn: turnIndex + 1,
                text: parsed.text,
                toolCalls: [],
                inputTokens: parsed.inputTokens,
                outputTokens: parsed.outputTokens,
                cachedTokens: parsed.cachedTokens,
                reasoningTokens: parsed.reasoningTokens,
                costUsd: computeCost({
                    model,
                    inputTokens: parsed.inputTokens,
                    outputTokens: parsed.outputTokens,
                    cachedTokens: parsed.cachedTokens,
                }),
                latencyMs,
                firstTokenMs,
                attempts,
            }
            turns.push(turn)
            hooks?.onTurnEnd?.(turn)
        }
    }

    const pricedTurns = turns.filter(t => t.costUsd !== null)
    const totals: AgentTotals = {
        turns: turns.length,
        toolCalls: toolCallCount,
        inputTokens: turns.reduce((sum, t) => sum + t.inputTokens, 0),
        outputTokens: turns.reduce((sum, t) => sum + t.outputTokens, 0),
        cachedTokens: turns.reduce((sum, t) => sum + t.cachedTokens, 0),
        reasoningTokens: turns.reduce((sum, t) => sum + t.reasoningTokens, 0),
        // null rather than a partial sum: an unpriced model must not look cheap.
        costUsd: pricedTurns.length === turns.length
            ? turns.reduce((sum, t) => sum + (t.costUsd ?? 0), 0)
            : null,
        latencyMs: Date.now() - startedAt,
    }

    log.info('run_finished', {
        inputId: input.id,
        stopReason,
        ...(stopDetail ? { stopDetail } : {}),
        turns: totals.turns,
        toolCalls: totals.toolCalls,
        costUsd: totals.costUsd,
        latencyMs: totals.latencyMs,
    })

    return {
        inputId: input.id,
        model,
        promptVersion: AGENT_PROMPT_VERSION,
        turns,
        finalText: turns[turns.length - 1]?.text ?? '',
        stopReason,
        stopDetail,
        totals,
        retrievalHits,
        injectedFaults: faults?.fired ?? [],
    }
}
