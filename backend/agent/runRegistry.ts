/**
 * Tracks agent runs in flight: identity, cancellation, concurrency ceilings,
 * a wall-clock deadline, and the SSE fan-out for each run.
 *
 * A run is not tied to the request that started it. Disconnecting does not
 * cancel it, and whatever it emitted while nobody was listening is replayed to
 * the next subscriber.
 *
 * The `Map` is private to this file; callers only ever use the exported
 * functions.
 */
import type { ServerResponse } from 'http'
import { createLogger } from '../utils/logger.js'
import { failedEvent } from './sseEvents.js'
import type { AgentSseEvent, EmitSseEvent } from './sseEvents.js'

const log = createLogger('agent-runs')

/** Ceiling on runs in flight across the process. */
export const MAX_ACTIVE_RUNS = 6

/** Ceiling per account, so one viewer cannot take every slot. */
export const MAX_ACTIVE_RUNS_PER_USER = 2

/**
 * Wall-clock limit on one run. Neither the per-tool timeout nor the SDK's
 * request timeout bounds a stream that keeps trickling, so this does.
 */
export const RUN_DEADLINE_MS = 90_000

/** How long a finished run stays subscribable before it is dropped. */
export const FINISHED_RUN_TTL_MS = 60_000

/** Abort reasons. They reach the trace through `stopDetail`. */
export const ABORT_CANCELLED = 'cancelled_by_user'
export const ABORT_DEADLINE = 'run_deadline'
export const ABORT_SHUTDOWN = 'server_shutdown'

export type CancelOutcome = 'cancelled' | 'not_found' | 'already_finished'

export class RunLimitError extends Error {
    readonly scope: 'global' | 'user'

    constructor(scope: 'global' | 'user', message: string) {
        super(message)
        this.name = 'RunLimitError'
        this.scope = scope
    }
}

export interface RunContext {
    signal: AbortSignal
    emit: EmitSseEvent
}

export interface StartRunParams {
    /** Chosen by the caller so the trace tracer can share it. */
    runId: string
    userId: string
    /** Does the actual work. Expected to emit its own terminal event. */
    run: (ctx: RunContext) => Promise<void>
}

interface RunEntry {
    runId: string
    userId: string
    controller: AbortController
    /** Everything emitted so far, replayed to whoever subscribes late. */
    events: AgentSseEvent[]
    subscribers: Set<ServerResponse>
    /** Null while in flight. */
    endedAt: number | null
    settled: Promise<void>
}

const runs = new Map<string, RunEntry>()

function isActive(entry: RunEntry): boolean {
    return entry.endedAt === null
}

/** Drops finished runs past their TTL, so the map cannot grow forever. */
function pruneFinished(): void {
    const cutoff = Date.now() - FINISHED_RUN_TTL_MS
    for (const [runId, entry] of runs) {
        if (entry.endedAt !== null && entry.endedAt < cutoff) runs.delete(runId)
    }
}

export function activeRunCount(): number {
    let count = 0
    for (const entry of runs.values()) if (isActive(entry)) count += 1
    return count
}

function activeRunCountForUser(userId: string): number {
    let count = 0
    for (const entry of runs.values()) {
        if (isActive(entry) && entry.userId === userId) count += 1
    }
    return count
}

function writeEvent(res: ServerResponse, event: AgentSseEvent): void {
    res.write(`data: ${JSON.stringify(event)}\n\n`)
}

/**
 * Appends to the replay buffer, merging consecutive narration slices of the
 * same turn so the buffer holds one entry per turn rather than one per token.
 * A late subscriber therefore receives each finished turn's narration whole.
 */
function record(entry: RunEntry, event: AgentSseEvent): void {
    const last = entry.events[entry.events.length - 1]
    if (
        event.type === 'reasoning_delta'
        && last?.type === 'reasoning_delta'
        && last.turn === event.turn
    ) {
        entry.events[entry.events.length - 1] = { ...last, delta: last.delta + event.delta }
        return
    }
    entry.events.push(event)
}

function emit(entry: RunEntry, event: AgentSseEvent): void {
    record(entry, event)
    for (const res of entry.subscribers) writeEvent(res, event)
}

/**
 * Marks the run finished and ends its viewers' responses. Leaving them open
 * would hold up `server.close()` on shutdown.
 */
function finish(entry: RunEntry): void {
    entry.endedAt = Date.now()
    for (const res of entry.subscribers) res.end()
    entry.subscribers.clear()
}

/** Registers and starts a run. Throws RunLimitError when a ceiling is reached. */
export function startRun({ runId, userId, run }: StartRunParams): void {
    pruneFinished()

    if (activeRunCount() >= MAX_ACTIVE_RUNS) {
        throw new RunLimitError('global', `too many runs in flight (max ${MAX_ACTIVE_RUNS})`)
    }
    if (activeRunCountForUser(userId) >= MAX_ACTIVE_RUNS_PER_USER) {
        throw new RunLimitError(
            'user',
            `you already have ${MAX_ACTIVE_RUNS_PER_USER} run(s) in flight`
        )
    }

    const controller = new AbortController()
    // `settled` is replaced immediately below; the run body needs the entry to
    // emit into, and the entry needs the promise the run body produces.
    const entry: RunEntry = {
        runId,
        userId,
        controller,
        events: [],
        subscribers: new Set(),
        endedAt: null,
        settled: Promise.resolve(),
    }
    runs.set(runId, entry)

    const deadline = setTimeout(() => {
        log.warn('run_deadline_reached', { runId, deadlineMs: RUN_DEADLINE_MS })
        controller.abort(ABORT_DEADLINE)
    }, RUN_DEADLINE_MS)
    // The run's own promise keeps the process alive; this timer must not.
    deadline.unref?.()

    log.info('run_started', { runId, userId, active: activeRunCount() })

    entry.settled = run({ signal: controller.signal, emit: event => emit(entry, event) })
        .catch((err: unknown) => {
            // Last resort only: the run body reports its own failures. Reaching
            // here means it threw before it could.
            log.error('run_failed', { runId, error: err instanceof Error ? err.message : String(err) })
            emit(entry, failedEvent(err))
        })
        .finally(() => {
            clearTimeout(deadline)
            finish(entry)
            log.info('run_ended', { runId, active: activeRunCount() })
        })
}

/**
 * Attaches a viewer. Everything already emitted is written first, then the
 * response joins the live feed — synchronously, so nothing can slip through
 * the gap between the two.
 */
export function subscribe(runId: string, userId: string, res: ServerResponse): boolean {
    const entry = runs.get(runId)
    if (!entry || entry.userId !== userId) return false

    for (const event of entry.events) writeEvent(res, event)

    if (!isActive(entry)) {
        res.end()
        return true
    }

    entry.subscribers.add(res)
    return true
}

export function unsubscribe(runId: string, res: ServerResponse): void {
    runs.get(runId)?.subscribers.delete(res)
}

export function cancelRun(runId: string, userId: string): CancelOutcome {
    const entry = runs.get(runId)
    if (!entry || entry.userId !== userId) return 'not_found'
    if (!isActive(entry)) return 'already_finished'

    log.info('run_cancelled', { runId })
    entry.controller.abort(ABORT_CANCELLED)
    return 'cancelled'
}

/** Aborts every run in flight and waits for them all to unwind. */
export async function abortAllRuns(reason: string = ABORT_SHUTDOWN): Promise<void> {
    const pending = [...runs.values()].filter(isActive)
    if (pending.length === 0) return

    log.info('aborting_all_runs', { count: pending.length, reason })
    for (const entry of pending) entry.controller.abort(reason)
    await Promise.allSettled(pending.map(entry => entry.settled))
}

/** Test seam: drops every entry. Never called by the server. */
export function resetRegistry(): void {
    runs.clear()
}
