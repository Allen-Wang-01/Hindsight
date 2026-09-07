/**
 * Stub implementations of the four agent tools, backed by the hand-written
 * fixture dataset. Every one of them satisfies ToolHandler, so replacing a
 * stub with a real query later touches this file only — the loop, the tool
 * definitions and the trace plumbing never learn the difference.
 *
 * The fakery is deliberately shallow but behaviourally honest:
 *   - search_records scores against summaries, so different queries really do
 *     return different sets (and some return none)
 *   - raw similarity is bucketed into relatedness before the model sees it,
 *     and the float is handed back out-of-band for the trace
 *   - an unknown unit is count 0, not an error
 */
import { readFileSync } from 'fs'
import { toolError } from './types.js'
import type {
    EvidenceEntry,
    GetRecordArgs,
    GetTopicDistributionArgs,
    GetTopicDistributionResult,
    GetUnitHistoryArgs,
    GetUnitHistoryResult,
    RecordDetail,
    Relatedness,
    RetrievalHit,
    SearchRecordsArgs,
    SearchRecordsResult,
    SourceType,
    ToolError,
    ToolHandler,
    ToolName,
    ToolOutcome,
    UnitType,
} from './types.js'

// ---- Fixture loading ----

interface FixtureRecord {
    record_id: string
    occurred_at: string
    channel: string
    source_type: SourceType
    description: string | null
    similarity: number
    summary: string
    content: string
}

interface FixtureUnit {
    unit_type: UnitType
    unit_id: string
    name: string
}

interface FixtureSubject {
    dimension: string
    name: string
    count: number
    last_seen: string
}

export interface FixtureInput {
    id: string
    note: string
    evidence: {
        unit_type: UnitType
        unit: string
        outcome: string
        difficulty: string | null
        occurred_at: string
        excerpt: string
    }
}

interface FixtureDataset {
    records: FixtureRecord[]
    units: FixtureUnit[]
    evidence: Record<string, EvidenceEntry[]>
    distribution: FixtureSubject[]
    inputs: FixtureInput[]
}

// Read at module load rather than importing the JSON, so this stays free of
// import-attribute and resolveJsonModule differences between tsx and tsc.
const dataset = JSON.parse(
    readFileSync(new URL('../fixtures/dataset.json', import.meta.url), 'utf8')
) as FixtureDataset

export const fixtureInputs: FixtureInput[] = dataset.inputs

// ---- Tuning constants ----

/** Retrieval floor, mirroring vectorSearch.js. Below this, a record is not a hit. */
const MATCH_THRESHOLD = 0.7

/** Max hits handed back. Deliberately not exposed to the model — phase 5 compares
 *  the agent against a fixed top-k baseline, which only means something if k is ours. */
const MATCH_COUNT = 5

/** Boundary between the two relatedness buckets. Phase 4's threshold sweep sets
 *  the real value; until then this is a placeholder, not a measured number. */
const STRONG_RELATEDNESS_AT = 0.85

const DEFAULT_WINDOW = '90d'

/** Character limit on get_record's full text. Longer content is cut and marked. */
export const MAX_RECORD_CHARS = 4000

function bucket(similarity: number): Relatedness {
    return similarity >= STRONG_RELATEDNESS_AT ? 'strong' : 'weak'
}

// ---- Argument helpers ----

function invalid(message: string): ToolError {
    return toolError('invalid_argument', message)
}

function readString(args: Record<string, unknown>, key: string): string | null {
    const value = args[key]
    return typeof value === 'string' && value.trim() !== '' ? value.trim() : null
}

// ---- search_records ----

const STOPWORDS = new Set([
    'about', 'after', 'again', 'asked', 'been', 'being', 'could', 'from', 'have',
    'into', 'just', 'like', 'more', 'only', 'other', 'over', 'said', 'same',
    'some', 'such', 'than', 'that', 'their', 'them', 'then', 'there', 'these',
    'they', 'this', 'time', 'were', 'what', 'when', 'which', 'with', 'would',
    'your',
])

function tokenize(text: string): string[] {
    return text
        .toLowerCase()
        .split(/[^a-z0-9]+/)
        .filter(token => token.length >= 4 && !STOPWORDS.has(token))
}

/**
 * Stands in for the pgvector query. Scoring is keyword overlap against the
 * summary — enough that distinct queries return distinct sets, which is the
 * only property the loop actually depends on. The score returned is the
 * record's fixture similarity, not the overlap count.
 */
export const searchRecords: ToolHandler = async (args): Promise<ToolOutcome> => {
    const query = readString(args, 'query')
    if (!query) {
        return { result: invalid('query is required and must be a non-empty string') }
    }

    const dateRange = args['date_range'] as SearchRecordsArgs['date_range'] | undefined
    const from = dateRange?.from ? Date.parse(dateRange.from) : null
    const to = dateRange?.to ? Date.parse(dateRange.to) : null

    const queryTokens = new Set(tokenize(query))

    const scored = dataset.records
        .filter(record => {
            const at = Date.parse(record.occurred_at)
            if (from !== null && at < from) return false
            if (to !== null && at > to) return false
            return tokenize(record.summary).some(token => queryTokens.has(token))
        })
        .filter(record => record.similarity >= MATCH_THRESHOLD)
        .sort((a, b) => b.similarity - a.similarity)
        .slice(0, MATCH_COUNT)

    const result: SearchRecordsResult = {
        count: scored.length,
        results: scored.map(record => ({
            record_id: record.record_id,
            occurred_at: record.occurred_at,
            channel: record.channel,
            summary: record.summary,
            relatedness: bucket(record.similarity),
        })),
    }

    // The floats stay here and go to the trace. They never enter the model's context.
    const hits: RetrievalHit[] = scored.map(record => ({
        record_id: record.record_id,
        similarity: record.similarity,
        relatedness: bucket(record.similarity),
    }))

    return { result, hits }
}

// ---- get_unit_history ----

export const getUnitHistory: ToolHandler = async (args): Promise<ToolOutcome> => {
    const unitType = readString(args, 'unit_type')
    const unit = readString(args, 'unit')

    if (unitType !== 'dimension' && unitType !== 'subject') {
        return { result: invalid("unit_type must be 'dimension' or 'subject'") }
    }
    if (!unit) {
        return { result: invalid('unit is required and must be a non-empty string') }
    }

    const needle = unit.toLowerCase()
    const match = dataset.units.find(
        candidate =>
            candidate.unit_type === unitType &&
            (candidate.unit_id.toLowerCase() === needle || candidate.name.toLowerCase() === needle)
    )

    // Unknown unit is not an error: "nothing on record yet" is a real answer,
    // and the agent is supposed to respond to it differently than to a failure.
    if (!match) {
        const result: GetUnitHistoryResult = {
            unit: { unit_type: unitType, unit_id: null, name: unit },
            count: 0,
            evidence: [],
        }
        return { result }
    }

    const evidence = [...(dataset.evidence[match.unit_id] ?? [])].sort(
        (a, b) => Date.parse(a.occurred_at) - Date.parse(b.occurred_at)
    )

    const result: GetUnitHistoryResult = {
        unit: { unit_type: match.unit_type, unit_id: match.unit_id, name: match.name },
        count: evidence.length,
        evidence,
    }
    return { result }
}

// ---- get_topic_distribution ----

function windowCutoff(window: string): number | null {
    const parsed = /^(\d+)d$/.exec(window)
    if (!parsed?.[1]) return null
    return Date.now() - Number(parsed[1]) * 24 * 60 * 60 * 1000
}

export const getTopicDistribution: ToolHandler = async (args): Promise<ToolOutcome> => {
    const dimension = readString(args, 'dimension')
    const window = readString(args, 'window') ?? DEFAULT_WINDOW

    const cutoff = windowCutoff(window)
    if (cutoff === null) {
        return { result: invalid("window must look like '30d' or '90d'") }
    }

    const subjects = dataset.distribution
        .filter(subject => !dimension || subject.dimension === dimension)
        .filter(subject => Date.parse(subject.last_seen) >= cutoff)
        .sort((a, b) => b.count - a.count)
        .map(({ name, count, last_seen }) => ({ name, count, last_seen }))

    const result: GetTopicDistributionResult = {
        dimension: dimension ?? null,
        window,
        subjects,
    }
    return { result }
}

// ---- get_record ----

export const getRecord: ToolHandler = async (args): Promise<ToolOutcome> => {
    const recordId = readString(args, 'record_id')
    if (!recordId) {
        return { result: invalid('record_id is required and must be a non-empty string') }
    }

    const record = dataset.records.find(candidate => candidate.record_id === recordId)
    if (!record) {
        return { result: toolError('not_found', `no record with id ${recordId}`) }
    }

    const totalChars = record.content.length
    const truncated = totalChars > MAX_RECORD_CHARS

    // The marker sits at the cut so the model meets it while reading, rather
    // than having to notice the flag at the end of the object.
    const content = truncated
        ? record.content.slice(0, MAX_RECORD_CHARS) +
          `\n\n[TRUNCATED: showing the first ${MAX_RECORD_CHARS} of ${totalChars} characters. ` +
          'The rest of this record was not retrieved — do not assume anything about it.]'
        : record.content

    const result: RecordDetail = {
        record_id: record.record_id,
        occurred_at: record.occurred_at,
        channel: record.channel,
        source_type: record.source_type,
        description: record.description,
        content,
        truncated,
        total_chars: totalChars,
    }
    return { result }
}

// ---- Registry ----

export const TOOL_HANDLERS: Record<ToolName, ToolHandler> = {
    search_records: searchRecords,
    get_unit_history: getUnitHistory,
    get_topic_distribution: getTopicDistribution,
    get_record: getRecord,
}

// Referenced so the arg interfaces stay tied to the handlers that read them;
// the handlers themselves validate at runtime rather than trusting the model.
export type { GetRecordArgs, GetTopicDistributionArgs, GetUnitHistoryArgs, SearchRecordsArgs }
