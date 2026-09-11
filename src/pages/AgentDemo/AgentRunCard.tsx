import { useAgentStream } from '../../hooks/useAgentStream'
import type { TrajectoryToolCall, TrajectoryTurn } from '../../hooks/useAgentStream'
import styles from '../../styles/AgentDemo.module.scss'

export interface AgentInput {
    id: string
    note: string
}

interface AgentRunCardProps {
    input: AgentInput
    faultInjectionEnabled: boolean
}

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

const AgentRunCard = ({ input, faultInjectionEnabled }: AgentRunCardProps) => {
    const { status, turns, outcome, activeTool, error, start, cancel } = useAgentStream()
    const isRunning = status === 'starting' || status === 'running'

    return (
        <section className={styles.card}>
            <header className={styles.cardHeader}>
                <h2 className={styles.cardTitle}>{input.id}</h2>
                <p className={styles.cardNote}>{input.note}</p>
            </header>

            <div className={styles.actions}>
                <button
                    className={styles.primaryAction}
                    onClick={() => start({ inputId: input.id })}
                    disabled={isRunning}
                >
                    Run
                </button>

                <button
                    className={styles.action}
                    onClick={() => start({ inputId: input.id, maxTurns: 1 })}
                    disabled={isRunning}
                >
                    Squeeze the budget to one turn
                </button>

                {faultInjectionEnabled && (
                    <>
                        <button
                            className={styles.action}
                            onClick={() =>
                                start({ inputId: input.id, faults: ['tool-error:get_record'] })
                            }
                            disabled={isRunning}
                        >
                            Inject one tool failure
                        </button>
                        <button
                            className={styles.action}
                            onClick={() =>
                                start({ inputId: input.id, faults: ['stream-interrupt:2'] })
                            }
                            disabled={isRunning}
                        >
                            Break the stream on turn 2
                        </button>
                    </>
                )}

                <button className={styles.cancel} onClick={cancel} disabled={!isRunning}>
                    Cancel
                </button>
            </div>

            {error && <p className={styles.error}>{error}</p>}

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

const Turn = ({ turn }: { turn: TrajectoryTurn }) => {
    const isDead = turn.status === 'error'

    return (
        <li className={isDead ? styles.deadTurn : styles.turn}>
            <div className={styles.turnHeader}>
                <span className={styles.turnNumber}>turn {turn.turn}</span>
                {turn.metrics && (
                    <span className={styles.turnMeta}>
                        {seconds(turn.metrics.latencyMs)} · in {turn.metrics.inputTokens} / out{' '}
                        {turn.metrics.outputTokens} · {money(turn.metrics.costUsd)}
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
                                {call.ok ? `${call.durationMs}ms` : call.errorCode}
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
