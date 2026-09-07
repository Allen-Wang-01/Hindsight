/**
 * Runs the agent loop against the hand-written fixture evidence and streams the
 * trajectory to the terminal as it happens.
 *
 * The point of this script is the comparison table at the bottom: the same
 * loop, the same tools, different inputs, different depths. If every input
 * takes the same number of tool calls, the loop is not doing what it claims to.
 *
 * Usage:
 *   cd backend
 *   npx tsx scripts/runAgentLoop.ts                     # every fixture input
 *   npx tsx scripts/runAgentLoop.ts --input deep        # just one
 *   npx tsx scripts/runAgentLoop.ts --model gpt-5-nano  # override the model
 *   npx tsx scripts/runAgentLoop.ts --max-turns 3       # squeeze the budget
 *   npx tsx scripts/runAgentLoop.ts --max-output-tokens 200  # force a truncated turn
 *   npx tsx scripts/runAgentLoop.ts --cancel-after 4000      # cancel mid-run
 *   npx tsx scripts/runAgentLoop.ts --max-cost 0.001    # trip the cost ceiling
 *   npx tsx scripts/runAgentLoop.ts --log               # structured logs alongside the trajectory
 *   npx tsx scripts/runAgentLoop.ts --json              # machine-readable, no streaming
 *
 * Fault injection (requires AGENT_FAULT_INJECTION=on in .env), repeatable:
 *   --fault tool-error:get_record:invalid_argument   # model has to correct itself
 *   --fault tool-timeout:search_records --tool-timeout 1500
 *   --fault stream-interrupt:2                       # turn 2's stream dies, then retries
 *
 * Requires OPENAI_API_KEY in .env. Writes nothing to any database — trace
 * persistence arrives in step B.
 */
import 'dotenv/config'
import { createFaultPlan, isFaultInjectionEnabled, parseFault } from '../agent/faults.js'
import type { Fault } from '../agent/faults.js'
import { DEFAULT_BUDGET, DEFAULT_MODEL, SILENT_LOGGER, runAgent } from '../agent/loop.js'
import type { AgentHooks, AgentRunResult, TurnRecord } from '../agent/loop.js'
import { createLogger } from '../utils/logger.js'
import { fixtureInputs } from '../agent/tools/stubs.js'

const DIM = '\x1b[2m'
const BOLD = '\x1b[1m'
const CYAN = '\x1b[36m'
const YELLOW = '\x1b[33m'
const RED = '\x1b[31m'
const RESET = '\x1b[0m'

function arg(name: string): string | undefined {
    const index = process.argv.indexOf(`--${name}`)
    return index === -1 ? undefined : process.argv[index + 1]
}

/** Repeatable flag: every `--name value` pair on the command line. */
function repeatedArg(name: string): string[] {
    const values: string[] = []
    process.argv.forEach((token, index) => {
        if (token !== `--${name}`) return
        const value = process.argv[index + 1]
        if (value) values.push(value)
    })
    return values
}

const wantsJson = process.argv.includes('--json')
const wantsLog = process.argv.includes('--log')
const only = arg('input')
const model = arg('model') ?? DEFAULT_MODEL
const maxTurns = arg('max-turns')
const maxToolCalls = arg('max-tool-calls')
const maxCost = arg('max-cost')
const maxOutputTokens = arg('max-output-tokens')
const cancelAfter = arg('cancel-after')
const toolTimeout = arg('tool-timeout')
const faultSpecs = repeatedArg('fault')

const budget = {
    maxTurns: maxTurns ? Number(maxTurns) : DEFAULT_BUDGET.maxTurns,
    maxToolCalls: maxToolCalls ? Number(maxToolCalls) : DEFAULT_BUDGET.maxToolCalls,
    maxCostUsd: maxCost ? Number(maxCost) : DEFAULT_BUDGET.maxCostUsd,
}

// Off by default, so the trajectory is not interleaved with log lines.
const logger = wantsLog ? createLogger('agent') : SILENT_LOGGER

function money(value: number | null): string {
    return value === null ? 'unpriced' : `$${value.toFixed(6)}`
}

function seconds(ms: number): string {
    return `${(ms / 1000).toFixed(1)}s`
}

/**
 * Streams narration to stdout with a hanging indent. Done here rather than by
 * buffering whole lines so the text appears as the model produces it — the
 * silence between turns was the whole complaint.
 */
function createIndentedWriter(): (delta: string) => void {
    let atLineStart = true
    return (delta: string) => {
        for (const char of delta) {
            if (atLineStart && char !== '\n') {
                process.stdout.write('  ')
                atLineStart = false
            }
            process.stdout.write(char)
            if (char === '\n') atLineStart = true
        }
    }
}

function turnMeta(turn: TurnRecord): string {
    const parts = [
        seconds(turn.latencyMs),
        turn.firstTokenMs === null ? 'no text' : `first token ${seconds(turn.firstTokenMs)}`,
        `in ${turn.inputTokens} / out ${turn.outputTokens}`,
        `reasoning ${turn.reasoningTokens}`,
        money(turn.costUsd),
    ]
    return parts.join(' · ')
}

function createHooks(): AgentHooks {
    const write = createIndentedWriter()
    let sawText = false

    return {
        onTurnStart(turn) {
            sawText = false
            process.stdout.write(`\n${CYAN}── turn ${turn} ──${RESET} ${DIM}thinking…${RESET}\n`)
        },
        onTextDelta(delta) {
            sawText = true
            write(delta)
        },
        onToolStart(name) {
            process.stdout.write(`\n  ${YELLOW}→${RESET} ${BOLD}${name}${RESET}${DIM} running…${RESET}`)
        },
        onToolEnd(call) {
            // Clear the "running…" line and reprint it as the finished one.
            const status = call.ok ? `${YELLOW}→${RESET}` : `${RED}✗ ${call.errorCode}${RESET}`
            const args = JSON.stringify(call.arguments)
            process.stdout.write(
                `\r\x1b[2K  ${status} ${BOLD}${call.name}${RESET}${DIM}(${args}) ${call.durationMs}ms${RESET}\n`
            )
        },
        onStreamRetry(retry) {
            // The half-sentence above stays on screen; saying so is the point of
            // the scenario — this is what the frontend has to survive.
            process.stdout.write(
                `\n${RED}✗ stream died mid-flight (${retry.reason})${RESET}\n` +
                `${DIM}  the ${retry.discardedChars} characters above are discarded; ` +
                `retrying in ${retry.delayMs}ms${RESET}\n`
            )
        },
        onTurnEnd(turn) {
            if (!sawText) {
                process.stdout.write(`${DIM}  (no narration this turn)${RESET}\n`)
            } else {
                process.stdout.write('\n')
            }
            const attempts = turn.attempts > 1 ? ` · ${turn.attempts} attempts` : ''
            process.stdout.write(`${DIM}  ${turnMeta(turn)}${attempts}${RESET}\n`)
        },
    }
}

const BUDGET_STOPS: AgentRunResult['stopReason'][] = ['max_turns', 'max_tool_calls', 'max_cost']

function printRunFooter(result: AgentRunResult): void {
    const t = result.totals

    if (result.stopReason === 'aborted') {
        console.log(
            `\n${RED}✗ cancelled — the loop stopped at its next checkpoint. ` +
            `The cost below is everything this run spent.${RESET}`
        )
    }

    // Loud on purpose: a truncated run ends mid-sentence, and that is easy to
    // mistake for a terse conclusion when it scrolls past in dim grey.
    if (result.stopReason === 'incomplete') {
        console.log(
            `\n${RED}✗ output truncated (${result.stopDetail}) — ` +
            `the text above stops wherever the limit hit, it is not a conclusion${RESET}`
        )
    }

    // Yellow, not red: running out of budget is a degradation, not a failure.
    if (BUDGET_STOPS.includes(result.stopReason)) {
        console.log(
            `\n${YELLOW}⚠ budget reached (${result.stopReason}) — the conclusion above came from ` +
            `forced convergence, with tools withheld${RESET}`
        )
    }

    if (result.injectedFaults.length > 0) {
        const fired = result.injectedFaults
            .map(fault => `${fault.kind}${fault.tool ? `:${fault.tool}` : ''}@turn ${fault.turn}`)
            .join(', ')
        console.log(`${YELLOW}injected (deliberate, marked in the trace): ${fired}${RESET}`)
    }

    const detail = result.stopDetail ? ` (${result.stopDetail})` : ''
    console.log(
        `\n${DIM}stop: ${result.stopReason}${detail} · turns ${t.turns} · tool calls ${t.toolCalls} · ` +
        `tokens ${t.inputTokens}/${t.outputTokens} (reasoning ${t.reasoningTokens}) · ` +
        `${money(t.costUsd)} · ${seconds(t.latencyMs)}${RESET}`
    )

    if (result.retrievalHits.length > 0) {
        // Printed here only to prove the floats exist and stayed out of the
        // model's context. In step B they go to the trace instead.
        const hits = result.retrievalHits
            .map(hit => `${hit.record_id.slice(-6)}=${hit.similarity}`)
            .join(' ')
        console.log(`${DIM}retrieval hits (trace-only, never sent to the model): ${hits}${RESET}`)
    }
}

function printComparison(results: AgentRunResult[]): void {
    console.log(`\n${BOLD}${'═'.repeat(72)}${RESET}`)
    console.log(`${BOLD}depth comparison${RESET}`)
    console.log(
        `${DIM}${'input'.padEnd(14)}${'turns'.padEnd(8)}${'tools'.padEnd(8)}` +
        `${'stop'.padEnd(16)}${'cost'.padEnd(12)}time${RESET}`
    )
    for (const result of results) {
        const t = result.totals
        console.log(
            String(result.inputId).padEnd(14) +
            String(t.turns).padEnd(8) +
            String(t.toolCalls).padEnd(8) +
            result.stopReason.padEnd(16) +
            money(t.costUsd).padEnd(12) +
            seconds(t.latencyMs)
        )
    }

    const distinctDepths = new Set(results.map(r => r.totals.toolCalls)).size
    if (results.length > 1 && distinctDepths === 1) {
        console.log(
            `\n${RED}Every input used the same number of tool calls. ` +
            `The loop is not branching on input — that is the thing step A is supposed to show.${RESET}`
        )
    }
}

async function main(): Promise<void> {
    if (!process.env['OPENAI_API_KEY']) {
        console.error('OPENAI_API_KEY must be set in .env')
        process.exit(1)
    }

    // Hard error rather than a silent clean run: asking for a fault and not
    // getting one is the confusing outcome.
    if (faultSpecs.length > 0 && !isFaultInjectionEnabled()) {
        console.error('fault injection is off — set AGENT_FAULT_INJECTION=on in .env to use --fault')
        process.exit(1)
    }

    let faults: Fault[] = []
    try {
        faults = faultSpecs.map(parseFault)
    } catch (err) {
        console.error(err instanceof Error ? err.message : String(err))
        process.exit(1)
    }

    const selected = only
        ? fixtureInputs.filter(input => input.id === only)
        : fixtureInputs

    if (selected.length === 0) {
        console.error(
            `no fixture input named "${only}". available: ${fixtureInputs.map(i => i.id).join(', ')}`
        )
        process.exit(1)
    }

    const results: AgentRunResult[] = []

    // Sequential on purpose: interleaved trajectories are unreadable, and the
    // latency numbers would stop meaning anything.
    for (const [index, input] of selected.entries()) {
        if (!wantsJson) {
            console.log(`\n${BOLD}${'═'.repeat(72)}${RESET}`)
            console.log(`${BOLD}${input.id}${RESET}  ${DIM}${model} · ${index + 1}/${selected.length}${RESET}`)
            console.log(`${DIM}expected: ${input.note}${RESET}`)
        }

        // A real AbortController on the real code path — the cancellation
        // scenario needs no injection, which is the point of demoing it.
        const controller = new AbortController()
        const cancelTimer = cancelAfter
            ? setTimeout(() => {
                process.stdout.write(`\n\n${RED}⏹ cancelling at ${cancelAfter}ms${RESET}\n`)
                controller.abort()
            }, Number(cancelAfter))
            : undefined

        const result = await runAgent({
            input,
            model,
            budget,
            logger,
            // A fresh plan per input: faults are one-shot, and the sweep would
            // otherwise spend them all on the first run.
            faults: createFaultPlan(faults),
            ...(toolTimeout ? { toolTimeoutMs: Number(toolTimeout) } : {}),
            ...(maxOutputTokens ? { maxOutputTokens: Number(maxOutputTokens) } : {}),
            ...(cancelAfter ? { signal: controller.signal } : {}),
            ...(wantsJson ? {} : { hooks: createHooks() }),
        })
        if (cancelTimer) clearTimeout(cancelTimer)
        results.push(result)

        if (!wantsJson) {
            printRunFooter(result)
        }
    }

    if (wantsJson) {
        console.log(JSON.stringify(results, null, 2))
    } else {
        printComparison(results)
    }
}

main().catch(err => {
    console.error(err)
    process.exit(1)
})
