import { useCallback, useEffect, useRef, useState } from 'react'
import api from '../api/axios'

/**
 * Starts an agent run and accumulates its SSE events into a trajectory.
 *
 * The events mirror the backend's `agent/sseEvents.ts`. `applyEvent` is the
 * only place they are folded into state, so the same function can rebuild a
 * trajectory from stored rows.
 */

export type RunStatus = 'idle' | 'starting' | 'running' | 'done' | 'error'

export interface TrajectoryToolCall {
    turn: number
    name: string
    arguments: unknown
    ok: boolean
    errorCode?: string
    durationMs: number
    resultMeta?: { count?: number; truncated?: boolean; total_chars?: number }
}

export interface TurnMetrics {
    inputTokens: number
    outputTokens: number
    cachedTokens: number
    reasoningTokens: number
    costUsd: number | null
    latencyMs: number
    firstTokenMs: number | null
    attempts: number
}

export interface TrajectoryTurn {
    turn: number
    /** 'error' marks an attempt that died mid-stream and was retried. */
    status: 'ok' | 'error'
    text: string
    toolCalls: TrajectoryToolCall[]
    /** Set once the turn completed. */
    metrics?: TurnMetrics
    /** Dead attempts only. */
    discardedChars?: number
    error?: string
}

/** The completed turn the backend sends with `turn_end`. */
export interface TurnRecordPayload extends TurnMetrics {
    turn: number
    text: string
    toolCalls: TrajectoryToolCall[]
}

export interface RunTotals {
    turns: number
    toolCalls: number
    inputTokens: number
    outputTokens: number
    cachedTokens: number
    reasoningTokens: number
    costUsd: number | null
    latencyMs: number
}

export interface InjectedFault {
    kind: string
    turn: number
    tool?: string
    code?: string
}

export interface RunOutcome {
    stopReason: string
    stopDetail: string | null
    totals: RunTotals
    injectedFaults: InjectedFault[]
}

export type AgentSseEvent =
    | { type: 'run_started'; runId: string; inputId: string; model: string }
    | { type: 'turn_start'; turn: number }
    | { type: 'reasoning_delta'; turn: number; delta: string }
    | { type: 'tool_start'; turn: number; name: string }
    | { type: 'tool_end'; turn: number; call: TrajectoryToolCall }
    | {
        type: 'stream_retry'
        turn: number
        attempt: number
        reason: string
        delayMs: number
        discardedChars: number
    }
    | { type: 'turn_end'; turn: TurnRecordPayload }
    | ({ type: 'done' } & RunOutcome)
    | { type: 'failed'; message: string }

export interface StartRunOptions {
    inputId: string
    faults?: string[]
    maxTurns?: number
    maxToolCalls?: number
    maxCostUsd?: number
    maxOutputTokens?: number
    maxInputTokens?: number
}

/** Folds one event into the trajectory. */
export function applyEvent(turns: TrajectoryTurn[], event: AgentSseEvent): TrajectoryTurn[] {
    const last = turns[turns.length - 1]

    switch (event.type) {
        case 'turn_start':
            return [...turns, { turn: event.turn, status: 'ok', text: '', toolCalls: [] }]

        case 'reasoning_delta':
            if (!last) return turns
            return replaceLast(turns, { ...last, text: last.text + event.delta })

        case 'tool_end':
            if (!last) return turns
            return replaceLast(turns, { ...last, toolCalls: [...last.toolCalls, event.call] })

        case 'turn_end': {
            if (!last) return turns
            const { toolCalls, text, ...metrics } = event.turn
            return replaceLast(turns, {
                ...last,
                // Trust the turn record for the tool calls: it is the complete
                // list, while the streamed ones arrive one at a time.
                toolCalls: toolCalls ?? last.toolCalls,
                text: text || last.text,
                metrics: {
                    inputTokens: metrics.inputTokens,
                    outputTokens: metrics.outputTokens,
                    cachedTokens: metrics.cachedTokens,
                    reasoningTokens: metrics.reasoningTokens,
                    costUsd: metrics.costUsd,
                    latencyMs: metrics.latencyMs,
                    firstTokenMs: metrics.firstTokenMs,
                    attempts: metrics.attempts,
                },
            })
        }

        // Closes the attempt that died and opens a fresh one under the same
        // turn number.
        case 'stream_retry': {
            if (!last) return turns
            const dead: TrajectoryTurn = {
                ...last,
                status: 'error',
                discardedChars: event.discardedChars,
                error: event.reason,
            }
            return [
                ...turns.slice(0, -1),
                dead,
                { turn: event.turn, status: 'ok', text: '', toolCalls: [] },
            ]
        }

        default:
            return turns
    }
}

function replaceLast(turns: TrajectoryTurn[], turn: TrajectoryTurn): TrajectoryTurn[] {
    return [...turns.slice(0, -1), turn]
}

export const useAgentStream = () => {
    const [status, setStatus] = useState<RunStatus>('idle')
    const [runId, setRunId] = useState<string | null>(null)
    const [turns, setTurns] = useState<TrajectoryTurn[]>([])
    const [outcome, setOutcome] = useState<RunOutcome | null>(null)
    const [activeTool, setActiveTool] = useState<string | null>(null)
    const [error, setError] = useState<string | null>(null)

    const eventSourceRef = useRef<EventSource | null>(null)
    const runIdRef = useRef<string | null>(null)
    /** Mirrors `status` so the stream's error handler can read it without
     *  becoming a dependency of the subscription. */
    const statusRef = useRef<RunStatus>('idle')

    const moveTo = useCallback((next: RunStatus) => {
        statusRef.current = next
        setStatus(next)
    }, [])

    const closeStream = useCallback(() => {
        eventSourceRef.current?.close()
        eventSourceRef.current = null
    }, [])

    useEffect(() => closeStream, [closeStream])

    const subscribe = useCallback((id: string) => {
        const baseUrl = import.meta.env.VITE_API_BASE_URL
        const source = new EventSource(`${baseUrl}/agent/runs/${id}/stream`, {
            withCredentials: true,
        })
        eventSourceRef.current = source

        source.onmessage = (message) => {
            let event: AgentSseEvent
            try {
                event = JSON.parse(message.data)
            } catch {
                return
            }

            if (event.type === 'tool_start') setActiveTool(event.name)
            if (event.type === 'tool_end' || event.type === 'turn_end') setActiveTool(null)

            if (event.type === 'done') {
                setOutcome({
                    stopReason: event.stopReason,
                    stopDetail: event.stopDetail,
                    totals: event.totals,
                    injectedFaults: event.injectedFaults,
                })
                moveTo('done')
                setActiveTool(null)
                closeStream()
                return
            }

            if (event.type === 'failed') {
                setError(event.message)
                moveTo('error')
                setActiveTool(null)
                closeStream()
                return
            }

            setTurns(current => applyEvent(current, event))
        }

        source.onerror = () => {
            // The server ends the stream once the run is over, which arrives
            // here as an error. Only report it when the run had not finished.
            if (statusRef.current !== 'done' && statusRef.current !== 'error') {
                setError('Lost connection to the run')
                moveTo('error')
            }
            closeStream()
        }
    }, [closeStream, moveTo])

    const start = useCallback(async (options: StartRunOptions) => {
        closeStream()
        moveTo('starting')
        setTurns([])
        setOutcome(null)
        setActiveTool(null)
        setError(null)
        setRunId(null)
        // Cleared before the request, so a failed start cannot leave the
        // previous run's id behind for cancel to fire at.
        runIdRef.current = null

        try {
            const { data } = await api.post('/agent/runs', options)
            runIdRef.current = data.runId
            setRunId(data.runId)
            moveTo('running')
            subscribe(data.runId)
        } catch (err: any) {
            setError(err?.response?.data?.error ?? 'Could not start the run')
            moveTo('error')
        }
    }, [closeStream, moveTo, subscribe])

    const cancel = useCallback(async () => {
        const id = runIdRef.current
        if (!id) return
        try {
            await api.delete(`/agent/runs/${id}`)
        } catch {
            // The run may have finished on its own; the stream reports the end.
        }
    }, [])

    return { status, runId, turns, outcome, activeTool, error, start, cancel }
}
