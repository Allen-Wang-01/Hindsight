import { useEffect, useState } from 'react'
import api from '../../api/axios'
import AgentRunCard from './AgentRunCard'
import type { AgentInput } from './AgentRunCard'
import styles from '../../styles/AgentDemo.module.scss'

interface AgentConfig {
    inputs: AgentInput[]
    faultInjectionEnabled: boolean
    model: string
    budget: { maxTurns: number; maxToolCalls: number; maxCostUsd: number }
}

const AgentDemoPage = () => {
    const [config, setConfig] = useState<AgentConfig | null>(null)
    const [error, setError] = useState<string | null>(null)

    useEffect(() => {
        let cancelled = false

        api.get('/agent/inputs')
            .then(({ data }) => {
                if (!cancelled) setConfig(data)
            })
            .catch(() => {
                if (!cancelled) setError('Could not load the agent inputs')
            })

        return () => {
            cancelled = true
        }
    }, [])

    return (
        <div className={styles.page}>
            <div className={styles.shell}>
                <header className={styles.header}>
                    <h1 className={styles.title}>Agent</h1>
                    <p className={styles.subtitle}>
                        Each card is one piece of evidence. The agent decides for itself how deep
                        to look — pick a scenario and watch it reason.
                    </p>
                    {config && (
                        <p className={styles.limits}>
                            {config.model} · up to {config.budget.maxTurns} turns,{' '}
                            {config.budget.maxToolCalls} tool calls, $
                            {config.budget.maxCostUsd} per run
                        </p>
                    )}
                </header>

                {error && <p className={styles.error}>{error}</p>}

                {config?.inputs.map(input => (
                    <AgentRunCard
                        key={input.id}
                        input={input}
                        faultInjectionEnabled={config.faultInjectionEnabled}
                    />
                ))}
            </div>
        </div>
    )
}

export default AgentDemoPage
