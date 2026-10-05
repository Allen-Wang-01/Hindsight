import { useState } from 'react'
import { useAgentStream } from '../../hooks/useAgentStream'
import type {
    RunOutcome,
    StartRunOptions,
    TrajectoryToolCall,
    TrajectoryTurn,
} from '../../hooks/useAgentStream'
import styles from '../../styles/AgentDemo.module.scss'

export interface AgentInput {
    id: string
    note: string
    /** Whether this card offers the harness scenarios beyond a plain run. */
    scenarios: boolean
}

interface AgentRunCardProps {
    input: AgentInput
    faultInjectionEnabled: boolean
    /** The server's default ceilings, used as the context bar's scale. */
    defaultMaxInputTokens: number
}

/** Low enough that a turn is cut off mid-sentence. */
const TRUNCATING_OUTPUT_TOKENS = 200

/** Small enough that the conversation outgrows it after a few turns. */
const SQUEEZED_INPUT_TOKENS = 2500

function money(value: number | null | undefined): string {
    if (value === null || value === undefined) return 'unpriced'
    return `$${value.toFixed(4)}`
}

function seconds(ms: number): string {
    return `${(ms / 1000).toFixed(1)}s`
}

function toolArguments(call: TrajectoryToolCall): string {
    try {
        return JSON.stringify(call.arguments)
    } catch {
        return String(call.arguments)
    }
}

const AgentRunCard = ({
    input,
    faultInjectionEnabled,
    defaultMaxInputTokens,
}: AgentRunCardProps) => {
    const { status, turns, outcome, activeTool, error, start, cancel } = useAgentStream()
    const [ceiling, setCeiling] = useState(defaultMaxInputTokens)
    const isRunning = status === 'starting' || status === 'running'

    /** Starts a run and remembers the context ceiling it will be held to. */
    const run = (options: Omit<StartRunOptions, 'inputId'> = {}) => {
        setCeiling(options.maxInputTokens ?? defaultMaxInputTokens)
        start({ inputId: input.id, ...options })
    }

    return (
        <section className={styles.card}>
            <header className={styles.cardHeader}>
                <h2 className={styles.cardTitle}>{input.id}</h2>
                <p className={styles.cardNote}>{input.note}</p>
            </header>

            <div className={styles.actions}>
                <button
                    className={styles.primaryAction}
                    onClick={() => run()}
                    disabled={isRunning}
                >
                    Run
                </button>

                {input.scenarios && (
                    <>
                        <button
                            className={styles.action}
                            onClick={() => run({ maxTurns: 1 })}
                            disabled={isRunning}
                        >
                            Squeeze the budget to one turn
                        </button>

                        <button
                            className={styles.action}
                            onClick={() => run({ maxInputTokens: SQUEEZED_INPUT_TOKENS })}
                            disabled={isRunning}
                        >
                            Shrink the context window
                        </button>

                        <button
                            className={styles.action}
                            onClick={() => run({ maxOutputTokens: TRUNCATING_OUTPUT_TOKENS })}
                            disabled={isRunning}
                        >
                            Cut the output short
                        </button>

                        {faultInjectionEnabled && (
                            <>
                                <button
                                    className={styles.action}
                                    onClick={() => run({ faults: ['tool-error'] })}
                                    disabled={isRunning}
                                >
                                    Inject one tool failure
                                </button>
                                <button
                                    className={styles.action}
                                    onClick={() => run({ faults: ['stream-interrupt:2'] })}
                                    disabled={isRunning}
                                >
                                    Break the stream on turn 2
                                </button>
                            </>
                        )}
                    </>
                )}

                <button className={styles.cancel} onClick={cancel} disabled={!isRunning}>
                    Cancel
                </button>
            </div>

            {error && <p className={styles.error}>{error}</p>}

            {turns.length > 0 && <ContextBar turns={turns} ceiling={ceiling} />}

            {turns.length > 0 && (
                <ol className={styles.trajectory}>
                    {turns.map((turn, index) => (
                        <Turn key={`${turn.turn}-${index}`} turn={turn} />
                    ))}
                </ol>
            )}

            {isRunning && (
                <p className={styles.working}>
                    {activeTool ? `running ${activeTool}…` : 'thinking…'}
                </p>
            )}

            {outcome && <StopNotice outcome={outcome} />}

            {outcome && (
                <footer className={styles.outcome}>
                    <span>
                        stopped: {outcome.stopReason}
                        {outcome.stopDetail ? ` (${outcome.stopDetail})` : ''}
                    </span>
                    <span>
                        {outcome.totals.turns} turns · {outcome.totals.toolCalls} tool calls ·{' '}
                        {money(outcome.totals.costUsd)} · {seconds(outcome.totals.latencyMs)}
                    </span>
                    {outcome.injectedFaults.length > 0 && (
                        <span className={styles.injected}>
                            injected on purpose:{' '}
                            {outcome.injectedFaults
                                .map(fault =>
                                    `${fault.kind}${fault.tool ? `:${fault.tool}` : ''} on turn ${fault.turn}`
                                )
                                .join(', ')}
                        </span>
                    )}
                </footer>
            )}
        </section>
    )
}

/**
 * How full the conversation is, against the ceiling this run was given. The
 * bar falls when the loop compacts, which is the only place compaction shows.
 */
const ContextBar = ({ turns, ceiling }: { turns: TrajectoryTurn[]; ceiling: number }) => {
    const measured = turns.filter(turn => turn.metrics)
    const latest = measured[measured.length - 1]?.metrics
    if (!latest) return null

    const previous = measured[measured.length - 2]?.metrics
    const compacted = previous ? latest.inputTokens < previous.inputTokens : false
    const percent = Math.min(100, Math.round((latest.inputTokens / ceiling) * 100))

    return (
        <div className={styles.context}>
            <div className={styles.contextTrack}>
                <div className={styles.contextFill} style={{ width: `${percent}%` }} />
            </div>
            <span className={styles.contextLabel}>
                context {latest.inputTokens.toLocaleString()} / {ceiling.toLocaleString()} tokens
                {compacted ? ' · compacted to fit' : ''}
            </span>
        </div>
    )
}

const BUDGET_STOPS = ['max_turns', 'max_tool_calls', 'max_cost', 'max_input_tokens']

/**
 * Says what a non-ordinary ending means. A truncated run in particular reads
 * as a terse conclusion unless it is called out.
 */
const StopNotice = ({ outcome }: { outcome: RunOutcome }) => {
    if (outcome.stopReason === 'incomplete') {
        return (
            <p className={styles.alarm}>
                Output truncated ({outcome.stopDetail}) — the text above stops where the limit
                hit. It is not a conclusion.
            </p>
        )
    }

    if (outcome.stopReason === 'aborted') {
        return (
            <p className={styles.alarm}>
                Cancelled — the loop stopped at its next checkpoint. The cost below is everything
                this run spent.
            </p>
        )
    }

    if (BUDGET_STOPS.includes(outcome.stopReason)) {
        return (
            <p className={styles.warn}>
                Budget reached ({outcome.stopReason}) — the conclusion above came from forced
                convergence, with the tools withheld.
            </p>
        )
    }

    return null
}

const Turn = ({ turn }: { turn: TrajectoryTurn }) => {
    const isDead = turn.status === 'error'

    return (
        <li className={isDead ? styles.deadTurn : styles.turn}>
            <div className={styles.turnHeader}>
                <span className={styles.turnNumber}>turn {turn.turn}</span>
                {turn.metrics && (
                    <span className={styles.turnMeta}>
                        {seconds(turn.metrics.latencyMs)} · in {turn.metrics.inputTokens}
                        {turn.metrics.cachedTokens > 0
                            ? ` (${turn.metrics.cachedTokens} cached)`
                            : ''}{' '}
                        / out {turn.metrics.outputTokens} · {money(turn.metrics.costUsd)}
                    </span>
                )}
            </div>

            {turn.text && <p className={styles.turnText}>{turn.text}</p>}

            {isDead && (
                <p className={styles.discarded}>
                    stream died ({turn.error}) — the {turn.discardedChars} characters above are
                    discarded, retrying
                </p>
            )}

            {turn.toolCalls.length > 0 && (
                <ul className={styles.toolCalls}>
                    {turn.toolCalls.map((call, index) => (
                        <li
                            key={`${call.name}-${index}`}
                            className={call.ok ? styles.toolCall : styles.toolCallFailed}
                        >
                            <span className={styles.toolName}>{call.name}</span>
                            <span className={styles.toolArgs}>{toolArguments(call)}</span>
                            <span className={styles.toolMeta}>
                                {call.ok ? '' : `${call.errorCode} · `}
                                {call.durationMs}ms
                                {call.resultMeta?.count !== undefined
                                    ? ` · ${call.resultMeta.count} results`
                                    : ''}
                                {call.resultMeta?.truncated ? ' · truncated' : ''}
                            </span>
                        </li>
                    ))}
                </ul>
            )}
        </li>
    )
}

export default AgentRunCard
