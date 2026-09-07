/**
 * Deterministic fault injection.
 *
 * A plan is built per run and handed to runAgent as a parameter — never module
 * state, so concurrent runs cannot see each other's faults. Each fault fires
 * once and is then spent.
 *
 * Injected failures are invisible to the model: they must look exactly like
 * real ones. They surface only in AgentRunResult.injectedFaults, which step B
 * writes to the trace so phase 4/5 can exclude these runs from failure rates.
 */
import { TOOL_NAMES } from './tools/types.js'
import type { ToolErrorCode, ToolName } from './tools/types.js'

export type FaultKind = 'tool-error' | 'tool-timeout' | 'stream-interrupt'

export interface Fault {
    kind: FaultKind
    /** Tool to hit. Undefined means the next tool call, whatever it is. */
    tool?: ToolName
    /** Error code for 'tool-error'. Defaults to 'internal'. */
    code?: ToolErrorCode
    /** Turn to hit for 'stream-interrupt'. Undefined means the next one. */
    turn?: number
}

/** A fault that actually fired. */
export interface FiredFault {
    kind: FaultKind
    turn: number
    tool?: string
    code?: ToolErrorCode
}

export interface FaultPlan {
    /** Consumes a fault for this tool call, or returns null. */
    takeToolFault(tool: ToolName, turn: number): Fault | null
    /** Consumes a stream interruption for this turn. */
    takeStreamInterrupt(turn: number): boolean
    readonly fired: FiredFault[]
}

const ENV_FLAG = 'AGENT_FAULT_INJECTION'

/** Read at call time, not at import, so a test or a restart can flip it. */
export function isFaultInjectionEnabled(): boolean {
    return process.env[ENV_FLAG] === 'on'
}

/**
 * The only way to build a plan, so the gate cannot be bypassed by a caller
 * that forgets it. Returns null when injection is off; the run then takes the
 * ordinary path with no injection points active.
 */
export function createFaultPlan(faults: Fault[]): FaultPlan | null {
    if (faults.length === 0 || !isFaultInjectionEnabled()) return null

    const pending = [...faults]
    const fired: FiredFault[] = []

    function take(match: (fault: Fault) => boolean): Fault | null {
        const index = pending.findIndex(match)
        return index === -1 ? null : pending.splice(index, 1)[0] ?? null
    }

    return {
        fired,

        takeToolFault(tool, turn) {
            const fault = take(
                candidate =>
                    (candidate.kind === 'tool-error' || candidate.kind === 'tool-timeout') &&
                    (candidate.tool === undefined || candidate.tool === tool)
            )
            if (!fault) return null
            fired.push({
                kind: fault.kind,
                turn,
                tool,
                ...(fault.code ? { code: fault.code } : {}),
            })
            return fault
        },

        takeStreamInterrupt(turn) {
            const fault = take(
                candidate =>
                    candidate.kind === 'stream-interrupt' &&
                    (candidate.turn === undefined || candidate.turn === turn)
            )
            if (!fault) return false
            fired.push({ kind: fault.kind, turn })
            return true
        },
    }
}

// ---- Spec parsing ----

const FAULT_KINDS: readonly FaultKind[] = ['tool-error', 'tool-timeout', 'stream-interrupt']

const ERROR_CODES: readonly ToolErrorCode[] = [
    'not_found',
    'invalid_argument',
    'internal',
    'timeout',
    'duplicate_call',
]

function isFaultKind(value: string): value is FaultKind {
    return (FAULT_KINDS as readonly string[]).includes(value)
}

function isToolName(value: string): value is ToolName {
    return (TOOL_NAMES as readonly string[]).includes(value)
}

function isErrorCode(value: string): value is ToolErrorCode {
    return (ERROR_CODES as readonly string[]).includes(value)
}

/**
 * Parses one spec string. Throws on anything malformed — a mistyped fault that
 * silently produces a clean run is worse than an error.
 *
 *   tool-error                              next tool call fails
 *   tool-error:get_record                   next get_record fails
 *   tool-error:get_record:invalid_argument  ...with that code
 *   tool-timeout:search_records             next search_records hangs
 *   stream-interrupt                        next turn's stream dies mid-flight
 *   stream-interrupt:2                      turn 2's stream dies
 */
export function parseFault(spec: string): Fault {
    const [kind, target, code] = spec.split(':')

    if (!kind || !isFaultKind(kind)) {
        throw new Error(`unknown fault kind "${kind ?? ''}". expected one of: ${FAULT_KINDS.join(', ')}`)
    }

    if (kind === 'stream-interrupt') {
        if (code) throw new Error('stream-interrupt takes at most a turn number')
        if (!target) return { kind }
        const turn = Number(target)
        if (!Number.isInteger(turn) || turn < 1) {
            throw new Error(`stream-interrupt turn must be a positive integer, got "${target}"`)
        }
        return { kind, turn }
    }

    const fault: Fault = { kind }

    if (target) {
        if (!isToolName(target)) {
            throw new Error(`unknown tool "${target}". expected one of: ${TOOL_NAMES.join(', ')}`)
        }
        fault.tool = target
    }

    if (code) {
        if (kind !== 'tool-error') {
            throw new Error(`${kind} does not take an error code`)
        }
        if (!isErrorCode(code)) {
            throw new Error(`unknown error code "${code}". expected one of: ${ERROR_CODES.join(', ')}`)
        }
        fault.code = code
    }

    return fault
}

/** What an injected tool failure says to the model. It must not reveal the injection. */
export function injectedErrorMessage(code: ToolErrorCode, tool: string): string {
    switch (code) {
        case 'invalid_argument':
            return `${tool} rejected these arguments; check them against the tool's schema and call it again`
        case 'not_found':
            return `${tool}: that identifier does not resolve to anything`
        case 'timeout':
            return `${tool} did not respond in time`
        case 'duplicate_call':
            return `${tool} has already been called with these arguments`
        case 'internal':
            return `${tool} failed while running`
    }
}
