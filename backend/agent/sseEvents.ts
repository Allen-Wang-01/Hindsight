/**
 * The SSE event stream for one agent run.
 *
 * Every field here also exists as a column or a field in `llm_traces`, so
 * accumulating these events yields the same structure as reading the run's
 * rows back: live delivers narration token by token, a replay delivers it
 * whole. Similarity floats are not among them — those stay in the
 * `retrieval_hits` column.
 */
import type { FiredFault } from './faults.js'
import type {
    AgentHooks,
    AgentRunResult,
    AgentTotals,
    StopReason,
    ToolCallRecord,
    TurnRecord,
} from './loop.js'

export type AgentSseEvent =
    /** Run metadata, always first on the wire. */
    | { type: 'run_started'; runId: string; inputId: string; model: string }
    | { type: 'turn_start'; turn: number }
    /** One slice of narration. Replay delivers the whole turn as a single one. */
    | { type: 'reasoning_delta'; turn: number; delta: string }
    | { type: 'tool_start'; turn: number; name: string }
    | { type: 'tool_end'; turn: number; call: ToolCallRecord }
    /**
     * A model call died mid-stream and is about to be retried. `discardedChars`
     * is how much of this turn's narration was already delivered and is now
     * void.
     */
    | {
        type: 'stream_retry'
        turn: number
        attempt: number
        reason: string
        delayMs: number
        discardedChars: number
    }
    /** The turn's accounting, once it is complete. */
    | { type: 'turn_end'; turn: TurnRecord }
    | {
        type: 'done'
        stopReason: StopReason
        stopDetail: string | null
        totals: AgentTotals
        /** Faults that fired this run. Empty unless injection was requested. */
        injectedFaults: FiredFault[]
    }
    /** The run threw instead of returning a result. */
    | { type: 'failed'; message: string }

export type EmitSseEvent = (event: AgentSseEvent) => void

/**
 * Turns loop callbacks into wire events. Pure: the emit function is the only
 * output, which is what makes this testable without a socket.
 */
export function createSseHooks(emit: EmitSseEvent): AgentHooks {
    // The tool hooks carry no turn number of their own, so it is tracked here.
    let currentTurn = 0

    return {
        onTurnStart(turn) {
            currentTurn = turn
            emit({ type: 'turn_start', turn })
        },
        onTextDelta(delta) {
            emit({ type: 'reasoning_delta', turn: currentTurn, delta })
        },
        onToolStart(name) {
            emit({ type: 'tool_start', turn: currentTurn, name })
        },
        onToolEnd(call) {
            emit({ type: 'tool_end', turn: currentTurn, call })
        },
        onStreamRetry(retry) {
            emit({ type: 'stream_retry', turn: currentTurn, ...retry })
        },
        onTurnEnd(turn) {
            emit({ type: 'turn_end', turn })
        },
    }
}

export function runStartedEvent(
    runId: string,
    inputId: string,
    model: string
): AgentSseEvent {
    return { type: 'run_started', runId, inputId, model }
}

export function doneEvent(result: AgentRunResult): AgentSseEvent {
    return {
        type: 'done',
        stopReason: result.stopReason,
        stopDetail: result.stopDetail,
        totals: result.totals,
        injectedFaults: result.injectedFaults,
    }
}

export function failedEvent(err: unknown): AgentSseEvent {
    return {
        type: 'failed',
        message: err instanceof Error ? err.message : String(err),
    }
}
