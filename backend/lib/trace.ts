import { hashInput } from './hash.js'
import { db } from '../config/postgres.js'
import { createLogger } from '../utils/logger.js'
import { computeCost, PRICING_VERSION } from '../config/modelPricing.js'

const log = createLogger('trace')

export type CallType = 'completion' | 'embedding'
export type TraceStatus = 'ok' | 'error'

/**
 * Fields mirror the `llm_traces` columns (see backend/SQL/llm_traces.sql),
 * in camelCase. Only callSite / callType / model / status are required —
 * everything else is filled in as later phases wire it up.
 */
export interface TracePayload {
    // Subject
    userId?: string
    recordId?: string
    callSite: string
    callType: CallType

    // Model and versions
    model: string
    promptVersion?: string
    schemaVersion?: string
    pricingVersion?: string

    // Agent-run structure. Absent on single-call sites (note_analysis, embeddings).
    /** One complete agent run — one evidence item and all of its turns. */
    runId?: string
    /** 1-based turn number within the run. */
    turnIndex?: number
    /** Set only on a run's final row; a run whose rows all lack it is a run that crashed. */
    stopReason?: string

    // Input / output
    input?: unknown
    output?: string
    inputChars?: number
    inputHash?: string
    reasoning?: string
    toolCalls?: unknown
    retrievalHits?: unknown

    // Accounting
    inputTokens?: number
    outputTokens?: number
    totalTokens?: number
    costUsd?: number
    latencyMs?: number
    firstTokenMs?: number

    // Result
    status: TraceStatus
    error?: string

    extra?: unknown
}

// Re-exported so existing callers keep importing it from here.
export { hashInput }

/**
 * Insert one row into llm_traces. Never throws — a failed trace write
 * must not take down the calling LLM pipeline, so failures are only
 * logged. Callers may fire-and-forget this.
 */
export async function insertTrace(payload: TracePayload): Promise<void> {
    try {
        if (!payload.callSite || !payload.callType || !payload.model || !payload.status) {
            log.warn('trace_missing_required_field', {
                callSite: payload.callSite,
                callType: payload.callType,
                model: payload.model,
                status: payload.status,
            })
        }

        const row: Record<string, unknown> = {
            user_id: payload.userId,
            record_id: payload.recordId,
            call_site: payload.callSite,
            call_type: payload.callType,

            model: payload.model,
            prompt_version: payload.promptVersion,
            schema_version: payload.schemaVersion,
            pricing_version: payload.pricingVersion,

            run_id: payload.runId,
            turn_index: payload.turnIndex,
            stop_reason: payload.stopReason,

            input: payload.input,
            output: payload.output,
            input_chars: payload.inputChars,
            input_hash: payload.inputHash,
            reasoning: payload.reasoning,
            tool_calls: payload.toolCalls,
            retrieval_hits: payload.retrievalHits,

            input_tokens: payload.inputTokens,
            output_tokens: payload.outputTokens,
            total_tokens: payload.totalTokens,
            cost_usd: payload.costUsd,
            latency_ms: payload.latencyMs,
            first_token_ms: payload.firstTokenMs,

            status: payload.status,
            error: payload.error,

            extra: payload.extra,
        }

        // Drop undefined fields so Postgres uses column defaults / NULL
        // instead of writing an explicit `undefined`.
        for (const key of Object.keys(row)) {
            if (row[key] === undefined) {
                delete row[key]
            }
        }

        await db.insert('llm_traces', row)
    } catch (err) {
        log.error('trace_insert_failed', {
            callSite: payload.callSite,
            error: err instanceof Error ? err.message : String(err),
        })
    }
}

/**
 * Delete all traces tied to a given record (e.g. cascading cleanup when
 * a Note is deleted). Never throws — trace cleanup must not block the
 * caller's own delete flow.
 */
export async function deleteTracesByRecord(recordId: string): Promise<void> {
    try {
        await db.delete('llm_traces', { eq: { record_id: recordId } })
    } catch (err) {
        log.error('trace_delete_by_record_failed', {
            recordId,
            error: err instanceof Error ? err.message : String(err),
        })
    }
}

// ---- Recorder ----

interface StartTraceMeta {
    callSite: string
    callType: CallType
    model: string
    userId?: string
    recordId?: string
    promptVersion?: string
    input?: unknown       // omitted on the embedding path
    inputChars?: number
    inputHash?: string

    // Agent-run structure; both known before the call starts.
    runId?: string
    turnIndex?: number
}

interface FinishResult {
    output?: string
    inputTokens?: number
    outputTokens?: number
    cachedTokens?: number
    retrievalHits?: unknown
    toolCalls?: unknown
    reasoning?: string
    /** Only the turn that ends a run passes this; it isn't known any earlier. */
    stopReason?: string
}

export interface TraceRecorder {
    /** Call on the streaming path when the first token arrives. Non-streaming callers skip this. */
    markFirstToken(): void
    /** Records a successful call (status: 'ok'). Never throws. */
    finish(result: FinishResult): Promise<void>
    /** Records a failed call (status: 'error'). Never throws. */
    fail(err: unknown): Promise<void>
}

/**
 * Starts timing an LLM call and returns a recorder to close it out with
 * finish()/fail(). Works for non-streaming, streaming, and embedding
 * calls alike — it does not wrap the provider call itself, since the
 * streaming path (aiWorker) needs to consume chunks directly to drive SSE.
 *
 * Only records the start time; nothing is written to the database until
 * finish() or fail() is called.
 */
export function startTrace(meta: StartTraceMeta): TraceRecorder {
    const startedAt = Date.now()
    let firstTokenAt: number | null = null

    return {
        markFirstToken(): void {
            if (firstTokenAt === null) {
                firstTokenAt = Date.now()
            }
        },

        async finish(result: FinishResult): Promise<void> {
            const latencyMs = Date.now() - startedAt
            const firstTokenMs = firstTokenAt !== null ? firstTokenAt - startedAt : undefined

            const totalTokens = result.inputTokens !== undefined || result.outputTokens !== undefined
                ? (result.inputTokens ?? 0) + (result.outputTokens ?? 0)
                : undefined

            const costUsd = computeCost({
                model: meta.model,
                inputTokens: result.inputTokens,
                outputTokens: result.outputTokens,
                cachedTokens: result.cachedTokens,
            }) ?? undefined

            // cachedTokens has no dedicated column — it lives in `extra` until
            // phase 2a needs it to validate prefix-cache hits.
            const extra = result.cachedTokens !== undefined
                ? { cachedTokens: result.cachedTokens }
                : undefined

            await insertTrace({
                userId: meta.userId,
                recordId: meta.recordId,
                callSite: meta.callSite,
                callType: meta.callType,
                model: meta.model,
                promptVersion: meta.promptVersion,
                pricingVersion: PRICING_VERSION,
                input: meta.input,
                output: result.output,
                inputChars: meta.inputChars,
                inputHash: meta.inputHash,
                reasoning: result.reasoning,
                toolCalls: result.toolCalls,
                retrievalHits: result.retrievalHits,
                runId: meta.runId,
                turnIndex: meta.turnIndex,
                stopReason: result.stopReason,
                inputTokens: result.inputTokens,
                outputTokens: result.outputTokens,
                totalTokens,
                costUsd,
                latencyMs,
                firstTokenMs,
                status: 'ok',
                extra,
            })
        },

        async fail(err: unknown): Promise<void> {
            const latencyMs = Date.now() - startedAt
            const firstTokenMs = firstTokenAt !== null ? firstTokenAt - startedAt : undefined

            await insertTrace({
                userId: meta.userId,
                recordId: meta.recordId,
                callSite: meta.callSite,
                callType: meta.callType,
                model: meta.model,
                promptVersion: meta.promptVersion,
                pricingVersion: PRICING_VERSION,
                input: meta.input,
                inputChars: meta.inputChars,
                inputHash: meta.inputHash,
                runId: meta.runId,
                turnIndex: meta.turnIndex,
                latencyMs,
                firstTokenMs,
                status: 'error',
                error: err instanceof Error ? err.message : String(err),
            })
        },
    }
}
