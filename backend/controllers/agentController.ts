/**
 * HTTP entry points for agent runs: start, watch, cancel.
 *
 * A run is registered under an id and outlives the request that created it, so
 * the three endpoints all address it by that id. Input comes from the fixture
 * evidence set; the loop, the model and the trace writes are the real ones.
 */
import { randomUUID } from 'crypto'
import type { Request, Response } from 'express'
import { createFaultPlan, isFaultInjectionEnabled, parseFault } from '../agent/faults.js'
import type { Fault, FaultPlan } from '../agent/faults.js'
import { DEFAULT_BUDGET, DEFAULT_MODEL, runAgent } from '../agent/loop.js'
import type { AgentBudget } from '../agent/loop.js'
import { AGENT_PROMPT_VERSION } from '../agent/prompt.js'
import {
    RunLimitError,
    cancelRun as cancelRegisteredRun,
    startRun,
    subscribe,
    unsubscribe,
} from '../agent/runRegistry.js'
import type { RunContext } from '../agent/runRegistry.js'
import { createRunTracer } from '../agent/runTrace.js'
import { createSseHooks, doneEvent, failedEvent, runStartedEvent } from '../agent/sseEvents.js'
import { fixtureInputs } from '../agent/tools/stubs.js'
import type { FixtureInput } from '../agent/tools/stubs.js'

/** Written to every trace row, separating these runs from the terminal runner's. */
const RUN_ORIGIN = 'demo'

const MAX_FAULTS_PER_RUN = 3

/**
 * Ceiling on the per-call output limit a request may set. Lowering it is how a
 * truncated response is produced on demand; it can only reduce what a turn
 * costs, so the cap just keeps the value bounded.
 */
const MAX_OUTPUT_TOKENS_CEILING = 4000

function userIdOf(req: Request): string | null {
    const { user } = req as Request & { user?: { userId?: unknown } }
    return typeof user?.userId === 'string' ? user.userId : null
}

function runIdOf(req: Request): string | null {
    const { runId } = req.params
    return typeof runId === 'string' && runId.length > 0 ? runId : null
}

/**
 * Builds the run's budget from the request. Each ceiling may be lowered but
 * never raised, and the model is not taken from the request at all — an
 * unpriced model makes the cost ceiling unenforceable.
 *
 * Returns an error message instead of a budget when a field is malformed.
 */
function resolveBudget(body: Record<string, unknown>): AgentBudget | string {
    const budget: AgentBudget = { ...DEFAULT_BUDGET }
    const fields: Array<keyof AgentBudget> = [
        'maxTurns',
        'maxToolCalls',
        'maxCostUsd',
        'maxInputTokens',
    ]

    for (const field of fields) {
        const raw = body[field]
        if (raw === undefined || raw === null) continue
        if (typeof raw !== 'number' || !Number.isFinite(raw) || raw <= 0) {
            return `${field} must be a positive number`
        }
        budget[field] = Math.min(raw, DEFAULT_BUDGET[field])
    }

    return budget
}

/**
 * Reads the optional per-call output limit. Returns undefined when the request
 * did not set one, or an error message when it is malformed.
 */
function resolveMaxOutputTokens(body: Record<string, unknown>): number | undefined | string {
    const raw = body['maxOutputTokens']
    if (raw === undefined || raw === null) return undefined
    if (typeof raw !== 'number' || !Number.isInteger(raw) || raw <= 0) {
        return 'maxOutputTokens must be a positive integer'
    }
    return Math.min(raw, MAX_OUTPUT_TOKENS_CEILING)
}

/**
 * Parses the requested fault specs into a plan. Returns null when none were
 * requested, or an error message when the request is rejected.
 */
function resolveFaults(body: Record<string, unknown>): FaultPlan | null | string {
    const raw = body['faults']
    if (raw === undefined || raw === null) return null

    if (!Array.isArray(raw) || raw.some(spec => typeof spec !== 'string')) {
        return 'faults must be an array of strings'
    }
    if (raw.length === 0) return null
    if (raw.length > MAX_FAULTS_PER_RUN) {
        return `at most ${MAX_FAULTS_PER_RUN} faults per run`
    }
    if (!isFaultInjectionEnabled()) {
        return 'fault injection is off on this server'
    }

    let faults: Fault[]
    try {
        faults = raw.map(spec => parseFault(spec as string))
    } catch (err) {
        return err instanceof Error ? err.message : String(err)
    }

    return createFaultPlan(faults)
}

/**
 * GET /api/agent/inputs
 * The available inputs and the limits a run will be held to.
 */
export const listAgentInputs = async (_req: Request, res: Response): Promise<void> => {
    res.json({
        inputs: fixtureInputs.map(input => ({
            id: input.id,
            note: input.note,
            scenarios: input.scenarios === true,
        })),
        faultInjectionEnabled: isFaultInjectionEnabled(),
        model: DEFAULT_MODEL,
        budget: DEFAULT_BUDGET,
    })
}

/**
 * POST /api/agent/runs
 * Body: { inputId, faults?: string[], maxTurns?, maxToolCalls?, maxCostUsd? }
 *
 * Returns once the run is registered. The trajectory is read from the stream
 * endpoint, which may be attached to later — events emitted before then are
 * replayed on connect.
 */
export const createAgentRun = async (req: Request, res: Response): Promise<void> => {
    const userId = userIdOf(req)
    if (!userId) {
        res.status(401).json({ error: 'Not authenticated' })
        return
    }

    const body = (req.body ?? {}) as Record<string, unknown>

    const input = fixtureInputs.find(candidate => candidate.id === body['inputId'])
    if (!input) {
        res.status(400).json({
            error: 'Unknown inputId',
            available: fixtureInputs.map(candidate => candidate.id),
        })
        return
    }

    const budget = resolveBudget(body)
    if (typeof budget === 'string') {
        res.status(400).json({ error: budget })
        return
    }

    const faults = resolveFaults(body)
    if (typeof faults === 'string') {
        res.status(400).json({ error: faults })
        return
    }

    const maxOutputTokens = resolveMaxOutputTokens(body)
    if (typeof maxOutputTokens === 'string') {
        res.status(400).json({ error: maxOutputTokens })
        return
    }

    const runId = randomUUID()

    try {
        startRun({
            runId,
            userId,
            run: ctx =>
                executeRun({ runId, userId, input, budget, faults, maxOutputTokens, ctx }),
        })
    } catch (err) {
        if (err instanceof RunLimitError) {
            res.status(429).json({ error: err.message, scope: err.scope })
            return
        }
        throw err
    }

    res.status(202).json({
        runId,
        streamUrl: `/api/agent/runs/${runId}/stream`,
    })
}

interface ExecuteRunParams {
    runId: string
    userId: string
    input: FixtureInput
    budget: AgentBudget
    faults: FaultPlan | null
    maxOutputTokens: number | undefined
    ctx: RunContext
}

/**
 * Runs the loop with the tracer wrapping the SSE hooks, so each callback
 * reaches both the viewer and the trace table. The terminal event is emitted
 * before the trace writes are awaited.
 */
async function executeRun({
    runId,
    userId,
    input,
    budget,
    faults,
    maxOutputTokens,
    ctx,
}: ExecuteRunParams): Promise<void> {
    const tracer = createRunTracer({
        runId,
        userId,
        model: DEFAULT_MODEL,
        promptVersion: AGENT_PROMPT_VERSION,
        input,
        faults,
        origin: RUN_ORIGIN,
        hooks: createSseHooks(ctx.emit),
    })

    ctx.emit(runStartedEvent(runId, input.id, DEFAULT_MODEL))

    try {
        const result = await runAgent({
            input,
            model: DEFAULT_MODEL,
            budget,
            faults,
            signal: ctx.signal,
            hooks: tracer.hooks,
            ...(maxOutputTokens === undefined ? {} : { maxOutputTokens }),
        })
        ctx.emit(doneEvent(result))
        await tracer.finish(result)
    } catch (err) {
        ctx.emit(failedEvent(err))
        await tracer.fail(err)
    }
}

/**
 * GET /api/agent/runs/:runId/stream
 * Replays what the run has emitted so far, then streams the rest.
 */
export const streamAgentRun = async (req: Request, res: Response): Promise<void> => {
    const userId = userIdOf(req)
    if (!userId) {
        res.status(401).json({ error: 'Not authenticated' })
        return
    }

    const runId = runIdOf(req)
    if (!runId) {
        res.status(400).json({ error: 'runId is required' })
        return
    }

    res.setHeader('Content-Type', 'text/event-stream')
    res.setHeader('Cache-Control', 'no-cache')
    res.setHeader('Connection', 'keep-alive')
    res.flushHeaders()

    // The headers are already out, so an unknown run is reported as an event
    // rather than a status code.
    if (!subscribe(runId, userId, res)) {
        res.write(`data: ${JSON.stringify({ type: 'failed', message: 'run not found' })}\n\n`)
        res.end()
        return
    }

    req.on('close', () => {
        unsubscribe(runId, res)
    })
}

/**
 * DELETE /api/agent/runs/:runId
 * Aborts the run at its next checkpoint.
 */
export const cancelAgentRun = async (req: Request, res: Response): Promise<void> => {
    const userId = userIdOf(req)
    if (!userId) {
        res.status(401).json({ error: 'Not authenticated' })
        return
    }

    const runId = runIdOf(req)
    if (!runId) {
        res.status(400).json({ error: 'runId is required' })
        return
    }

    const outcome = cancelRegisteredRun(runId, userId)
    if (outcome === 'not_found') {
        res.status(404).json({ error: 'Run not found' })
        return
    }

    res.json({ outcome })
}
