/**
 * Manual test script for the llm_traces table: seed fake rows, print the
 * most recent ones, or clean up the seed rows again.
 *
 * Usage:
 *   cd backend
 *   npx tsx scripts/checkTraces.ts --seed         # insert 3 fake rows (call_site = '_seed_test')
 *   npx tsx scripts/checkTraces.ts                 # print the 10 most recent rows
 *   npx tsx scripts/checkTraces.ts --clear-seed    # delete only the seed rows
 *   npx tsx scripts/checkTraces.ts --price-check   # print computeCost() on fixed inputs; no network calls
 *   npx tsx scripts/checkTraces.ts --last-output   # print call_site + first 400 chars of the latest row's output
 *   npx tsx scripts/checkTraces.ts --summary       # per call_site: calls, total tokens, total cost, avg latency
 *   npx tsx scripts/checkTraces.ts --run           # replay the most recent agent run
 *   npx tsx scripts/checkTraces.ts --run <id>      # replay one run; a record id lists its runs
 *
 * Requires in .env: SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY
 */
import 'dotenv/config'
import { createClient } from '@supabase/supabase-js'
import { db } from '../config/postgres.js'
import { insertTrace, hashInput } from '../lib/trace.js'
import { computeCost } from '../config/modelPricing.js'

const SUPABASE_URL = process.env['SUPABASE_URL']
const SUPABASE_SERVICE_ROLE_KEY = process.env['SUPABASE_SERVICE_ROLE_KEY']

if (!SUPABASE_URL || !SUPABASE_SERVICE_ROLE_KEY) {
    console.error('SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY must be set in .env')
    process.exit(1)
}

const supabase = createClient(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)

const SEED_CALL_SITE = '_seed_test'

async function seed(): Promise<void> {
    const completionInput = { prompt: 'seed completion prompt for checkTraces --seed' }
    const completionOutput = 'seed completion output'

    await insertTrace({
        callSite: SEED_CALL_SITE,
        callType: 'completion',
        model: 'gpt-5-nano',
        promptVersion: 'v1-seed',
        input: completionInput,
        output: completionOutput,
        inputChars: JSON.stringify(completionInput).length,
        inputHash: hashInput(JSON.stringify(completionInput)),
        inputTokens: 120,
        outputTokens: 45,
        totalTokens: 165,
        costUsd: 0.000231,
        latencyMs: 842,
        firstTokenMs: 210,
        status: 'ok',
    })

    const embeddingText = 'seed embedding input for checkTraces --seed'
    await insertTrace({
        callSite: SEED_CALL_SITE,
        callType: 'embedding',
        model: 'text-embedding-3-small',
        inputChars: embeddingText.length,
        inputHash: hashInput(embeddingText),
        inputTokens: 12,
        totalTokens: 12,
        status: 'ok',
    })

    await insertTrace({
        callSite: SEED_CALL_SITE,
        callType: 'completion',
        model: 'gpt-5-nano',
        status: 'error',
        error: 'seed error: simulated failure for checkTraces --seed',
    })

    console.log('Seeded 3 rows with call_site = "_seed_test"')
}

async function listRecent(): Promise<void> {
    const { data, error } = await supabase
        .from('llm_traces')
        .select(
            'call_site, call_type, model, prompt_version, input_tokens, output_tokens, cost_usd, latency_ms, status, created_at'
        )
        .order('created_at', { ascending: false })
        .limit(10)

    if (error) {
        console.error(error)
        process.exit(1)
    }

    if (!data || data.length === 0) {
        console.log('(no rows in llm_traces)')
        return
    }

    console.table(data)
}

interface PriceCheckRow {
    model: string
    inputTokens: number
    cachedTokens: number
    outputTokens: number
}

const PRICE_CHECK_ROWS: PriceCheckRow[] = [
    { model: 'gpt-5-nano', inputTokens: 3412, cachedTokens: 0, outputTokens: 891 },
    { model: 'gpt-5-nano', inputTokens: 3412, cachedTokens: 2048, outputTokens: 891 },
    { model: 'text-embedding-3-small', inputTokens: 284, cachedTokens: 0, outputTokens: 0 },
    { model: 'unknown-model', inputTokens: 1000, cachedTokens: 0, outputTokens: 1000 },
]

function priceCheck(): void {
    const modelWidth = Math.max(
        'model'.length,
        ...PRICE_CHECK_ROWS.map((row) => row.model.length)
    )

    const header =
        'model'.padEnd(modelWidth) +
        '  ' + 'in_tok'.padStart(6) +
        '  ' + 'cached'.padStart(6) +
        '  ' + 'out_tok'.padStart(7) +
        '  ' + 'cost_usd'

    const lines = PRICE_CHECK_ROWS.map((row) => {
        const cost = computeCost({
            model: row.model,
            inputTokens: row.inputTokens,
            outputTokens: row.outputTokens,
            cachedTokens: row.cachedTokens,
        })
        const costStr = cost === null ? 'null' : cost.toFixed(10)

        return (
            row.model.padEnd(modelWidth) +
            '  ' + String(row.inputTokens).padStart(6) +
            '  ' + String(row.cachedTokens).padStart(6) +
            '  ' + String(row.outputTokens).padStart(7) +
            '  ' + costStr
        )
    })

    console.log([header, ...lines].join('\n'))
}

async function lastOutput(): Promise<void> {
    const { data, error } = await supabase
        .from('llm_traces')
        .select('call_site, output')
        .order('created_at', { ascending: false })
        .limit(1)

    if (error) {
        console.error(error)
        process.exit(1)
    }

    const row = data?.[0]
    if (!row) {
        console.log('(no rows in llm_traces)')
        return
    }

    console.log(`call_site: ${row.call_site}`)
    console.log('output (first 400 chars):')
    console.log(String(row.output ?? '').slice(0, 400))
}

interface SummaryTraceRow {
    call_site: string
    total_tokens: number | null
    cost_usd: number | null
    latency_ms: number | null
}

async function summary(): Promise<void> {
    const { data, error } = await supabase
        .from('llm_traces')
        .select('call_site, total_tokens, cost_usd, latency_ms')
        .limit(10000)

    if (error) {
        console.error(error)
        process.exit(1)
    }

    if (!data || data.length === 0) {
        console.log('(no rows in llm_traces)')
        return
    }

    const groups = new Map<string, { calls: number; totalTokens: number; totalCost: number; latencySum: number; latencyCount: number }>()

    for (const row of data as SummaryTraceRow[]) {
        const group = groups.get(row.call_site) ?? { calls: 0, totalTokens: 0, totalCost: 0, latencySum: 0, latencyCount: 0 }
        group.calls += 1
        group.totalTokens += row.total_tokens ?? 0
        group.totalCost += row.cost_usd ?? 0
        if (row.latency_ms !== null) {
            group.latencySum += row.latency_ms
            group.latencyCount += 1
        }
        groups.set(row.call_site, group)
    }

    const rows = Array.from(groups.entries())
        .map(([callSite, g]) => ({
            callSite,
            calls: g.calls,
            totalTokens: g.totalTokens,
            totalCost: g.totalCost,
            avgLatencyMs: g.latencyCount > 0 ? g.latencySum / g.latencyCount : null,
        }))
        .sort((a, b) => b.totalCost - a.totalCost)

    const callSiteWidth = Math.max('call_site'.length, ...rows.map((r) => r.callSite.length))

    const header =
        'call_site'.padEnd(callSiteWidth) +
        '  ' + 'calls'.padStart(5) +
        '  ' + 'total_tokens'.padStart(12) +
        '  ' + 'total_cost_usd'.padStart(14) +
        '  ' + 'avg_latency_ms'

    const lines = rows.map((r) =>
        r.callSite.padEnd(callSiteWidth) +
        '  ' + String(r.calls).padStart(5) +
        '  ' + String(r.totalTokens).padStart(12) +
        '  ' + r.totalCost.toFixed(6).padStart(14) +
        '  ' + (r.avgLatencyMs !== null ? r.avgLatencyMs.toFixed(1) : 'n/a')
    )

    console.log([header, ...lines].join('\n'))
}

// ---- Run replay ----

const RUN_COLUMNS =
    'run_id, turn_index, model, prompt_version, status, error, reasoning, output, ' +
    'tool_calls, retrieval_hits, input_tokens, output_tokens, cost_usd, latency_ms, ' +
    'first_token_ms, stop_reason, extra, created_at'

interface ToolCallRow {
    name: string
    arguments: unknown
    ok: boolean
    errorCode?: string
    durationMs: number
    resultMeta?: { count?: number; truncated?: boolean; totalChars?: number }
}

interface RunRow {
    run_id: string
    turn_index: number | null
    model: string
    prompt_version: string | null
    status: string
    error: string | null
    reasoning: string | null
    output: string | null
    tool_calls: ToolCallRow[] | null
    retrieval_hits: Array<{ record_id: string; similarity: number }> | null
    input_tokens: number | null
    output_tokens: number | null
    cost_usd: number | null
    latency_ms: number | null
    first_token_ms: number | null
    stop_reason: string | null
    extra: Record<string, unknown> | null
    created_at: string
}

function clip(text: string, limit: number): string {
    const flat = text.replace(/\s+/g, ' ').trim()
    return flat.length > limit ? `${flat.slice(0, limit)}…` : flat
}

function ms(value: number | null): string {
    return value === null ? 'n/a' : `${(value / 1000).toFixed(1)}s`
}

async function fetchRunRows(runId: string): Promise<RunRow[]> {
    const { data, error } = await supabase
        .from('llm_traces')
        .select(RUN_COLUMNS)
        .eq('run_id', runId)
        // turn_index alone is not enough: a retried turn has two rows.
        .order('turn_index', { ascending: true })
        .order('created_at', { ascending: true })

    if (error) {
        console.error(error)
        process.exit(1)
    }
    return (data ?? []) as unknown as RunRow[]
}

/** One record can produce several runs — phase 5 runs the agent per evidence item. */
async function listRunsForRecord(recordId: string): Promise<boolean> {
    const { data, error } = await supabase
        .from('llm_traces')
        .select('run_id, created_at, stop_reason, cost_usd')
        .eq('record_id', recordId)
        .order('created_at', { ascending: false })
        .limit(500)

    if (error) {
        console.error(error)
        process.exit(1)
    }

    const runs = new Map<string, { createdAt: string; stopReason: string | null; cost: number }>()
    for (const row of (data ?? []) as Array<{
        run_id: string | null
        created_at: string
        stop_reason: string | null
        cost_usd: number | null
    }>) {
        if (!row.run_id) continue
        const run = runs.get(row.run_id) ?? { createdAt: row.created_at, stopReason: null, cost: 0 }
        run.cost += Number(row.cost_usd ?? 0)
        if (row.stop_reason) run.stopReason = row.stop_reason
        runs.set(row.run_id, run)
    }

    if (runs.size === 0) return false

    console.log(`${runs.size} run(s) for record ${recordId}:`)
    for (const [runId, run] of runs) {
        console.log(
            `  ${runId}  ${run.createdAt}  ${run.stopReason ?? 'crashed'}  $${run.cost.toFixed(6)}`
        )
    }
    console.log('\nPass one of these run ids to replay it.')
    return true
}

async function latestRunId(): Promise<string | null> {
    const { data, error } = await supabase
        .from('llm_traces')
        .select('run_id')
        .not('run_id', 'is', null)
        .order('created_at', { ascending: false })
        .limit(1)

    if (error) {
        console.error(error)
        process.exit(1)
    }
    return (data?.[0] as { run_id: string } | undefined)?.run_id ?? null
}

function printRunRow(row: RunRow): void {
    const turn = `turn ${row.turn_index ?? '?'}`
    const attempt = row.extra?.['attempt']
    const attemptLabel = typeof attempt === 'number' && attempt > 1 ? ` (attempt ${attempt})` : ''

    if (row.status === 'error') {
        console.log(`${turn}${attemptLabel}  ERROR  ${row.error ?? ''}  ${ms(row.latency_ms)}`)
        const discarded = row.extra?.['discardedChars']
        if (typeof discarded === 'number') {
            console.log(`    discarded ${discarded} chars: "${clip(row.reasoning ?? '', 160)}"`)
        }
        return
    }

    const reasoningTokens = row.extra?.['reasoningTokens']
    const cost = row.cost_usd === null ? 'unpriced' : `$${Number(row.cost_usd).toFixed(6)}`
    console.log(
        `${turn}${attemptLabel}  ok  in ${row.input_tokens ?? 0} / out ${row.output_tokens ?? 0}` +
        `${typeof reasoningTokens === 'number' ? ` (reasoning ${reasoningTokens})` : ''}` +
        `  ${cost}  ${ms(row.latency_ms)}  first token ${ms(row.first_token_ms)}`
    )

    if (row.reasoning) {
        console.log(`    ${clip(row.reasoning, 300)}`)
    }

    for (const call of row.tool_calls ?? []) {
        const status = call.ok ? '→' : `✗ ${call.errorCode ?? 'error'}`
        const meta = call.resultMeta
        const metaParts: string[] = []
        if (typeof meta?.count === 'number') metaParts.push(`count ${meta.count}`)
        if (meta?.truncated) metaParts.push(`truncated of ${meta.totalChars} chars`)
        console.log(
            `    ${status} ${call.name}(${clip(JSON.stringify(call.arguments ?? {}), 80)}) ` +
            `${call.durationMs}ms${metaParts.length > 0 ? ` · ${metaParts.join(' · ')}` : ''}`
        )
    }
}

async function replayRun(id?: string): Promise<void> {
    let runId = id
    if (!runId) {
        const latest = await latestRunId()
        if (!latest) {
            console.log('(no rows with a run_id in llm_traces)')
            return
        }
        runId = latest
    }

    let rows = await fetchRunRows(runId)
    if (rows.length === 0) {
        if (await listRunsForRecord(runId)) return
        console.log(`(no rows for run_id or record_id "${runId}")`)
        return
    }

    const first = rows[0] as RunRow
    const last = rows[rows.length - 1] as RunRow
    const okRows = rows.filter((row) => row.status === 'ok')
    const failedRows = rows.length - okRows.length
    const turns = new Set(okRows.map((row) => row.turn_index)).size
    const toolCalls = okRows.reduce((sum, row) => sum + (row.tool_calls?.length ?? 0), 0)
    const cost = rows.reduce((sum, row) => sum + Number(row.cost_usd ?? 0), 0)

    console.log(
        `run ${runId} · ${first.model} · ${first.prompt_version ?? 'no prompt_version'} · ` +
        `origin ${String(first.extra?.['origin'] ?? 'unknown')} · ${first.created_at}`
    )
    console.log('─'.repeat(72))

    for (const row of rows) {
        printRunRow(row)
    }

    console.log('─'.repeat(72))
    // Turns counts distinct ok rows: a failed attempt is another row, not another turn.
    console.log(
        `stop: ${last.stop_reason ?? 'none (run crashed or was cut off)'} · ` +
        `turns ${turns} · rows ${rows.length} (${failedRows} failed attempt(s)) · ` +
        `tool calls ${toolCalls} · $${cost.toFixed(6)}`
    )

    const hits = rows.flatMap((row) => row.retrieval_hits ?? [])
    if (hits.length > 0) {
        console.log(
            `retrieval hits (trace-only): ` +
            hits.map((hit) => `${hit.record_id.slice(-6)}=${hit.similarity}`).join(' ')
        )
    }

    const injected = rows.flatMap((row) => {
        const fired = row.extra?.['injectedFaults']
        return Array.isArray(fired) ? (fired as Array<{ kind: string; turn: number }>) : []
    })
    if (injected.length > 0) {
        console.log(
            `injected (deliberate, excluded from failure rates): ` +
            injected.map((fault) => `${fault.kind}@turn ${fault.turn}`).join(', ')
        )
    }

    if (rows.some((row) => row.output)) {
        const conclusion = rows.filter((row) => row.output).at(-1)
        console.log(`\nconclusion:\n${conclusion?.output ?? ''}`)
    }
}

async function clearSeed(): Promise<void> {
    // Hardcoded to the seed marker only — must never delete real traces.
    await db.delete('llm_traces', { eq: { call_site: '_seed_test' } })
    console.log('Deleted rows with call_site = "_seed_test"')
}

async function run(): Promise<void> {
    const mode = process.argv[2]

    if (mode === '--seed') {
        await seed()
    } else if (mode === '--clear-seed') {
        await clearSeed()
    } else if (mode === '--price-check') {
        priceCheck()
    } else if (mode === '--last-output') {
        await lastOutput()
    } else if (mode === '--summary') {
        await summary()
    } else if (mode === '--run') {
        await replayRun(process.argv[3])
    } else if (!mode) {
        await listRecent()
    } else {
        console.error(`Unknown argument: ${mode}`)
        console.error('Usage: npx tsx scripts/checkTraces.ts [--seed | --clear-seed | --price-check | --last-output | --summary | --run [id]]')
        process.exit(1)
    }

    process.exit(0)
}

run().catch((err: unknown) => {
    console.error(err)
    process.exit(1)
})
