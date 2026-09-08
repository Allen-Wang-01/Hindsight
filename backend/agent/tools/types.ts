/**
 * The agent tool contract.
 *
 * These shapes are an INTERFACE CONTRACT, not throwaway fixture types.
 * Phase 2a fills the `unit_evidence` schema against them; when the stub
 * implementations are replaced by real queries, only the function bodies
 * change — these types and every call site stay untouched.
 *
 * Three rules the shapes encode:
 *
 *   1. Retrieval tools return summaries only. Full text requires a separate
 *      get_record call. This is not a token optimisation — the constraint is
 *      what forces the model into multiple turns.
 *
 *   2. Raw similarity scores never reach the model. It sees a `relatedness`
 *      bucket; the underlying float is carried separately in RetrievalHit,
 *      which goes to the trace only.
 *
 *   3. An empty result is a success with count 0, never an error. The agent
 *      has to distinguish "nothing found" from "the tool broke" in order to
 *      produce the three empty-result behaviours.
 *
 * Field names stay domain-neutral. Values that belong to a domain (outcome
 * levels, dimension names) are plain strings sourced from configuration and
 * are never hardcoded or unioned here.
 */

// ---- Primitives ----

/** MongoDB ObjectId rendered as a 24-character hex string. */
export type RecordId = string

/** ISO 8601 timestamp. Always a string — tool results are JSON-serialised. */
export type IsoTimestamp = string

/**
 * How strongly a retrieved record relates to the query. This is what the
 * model sees instead of a cosine similarity: a raw 0.87 is not actionable
 * for it and only ever ends up copied into the reasoning text, where the
 * readable numbers are supposed to be counts and dates.
 */
export type Relatedness = 'strong' | 'weak'

/** Cognitive unit kind. Both values are structural, not domain vocabulary. */
export type UnitType = 'dimension' | 'subject'

/**
 * Trust level of a record's content: `authored` means the user produced it,
 * `saved` means they preserved someone else's material. The agent must never
 * read a `saved` record as evidence of the user's own performance.
 *
 * If saved content later migrates into `saved_memories`, records become
 * uniformly `authored` and this field degrades into a compatibility value —
 * removing it then is a one-line change that breaks no caller.
 */
export type SourceType = 'authored' | 'saved'

// ---- Errors ----

export type ToolErrorCode =
    | 'not_found'        // the identifier resolved to nothing that exists
    | 'invalid_argument' // arguments failed validation
    | 'internal'         // the tool itself failed
    | 'timeout'          // the tool did not return inside its time limit
    | 'duplicate_call'   // this exact call was already made — raised by the loop

export interface ToolError {
    error: {
        code: ToolErrorCode
        message: string
    }
}

/** Every tool returns either its success shape or a ToolError. */
export type ToolResult<T> = T | ToolError

export function isToolError(result: unknown): result is ToolError {
    return typeof result === 'object' && result !== null && 'error' in result
}

export function toolError(code: ToolErrorCode, message: string): ToolError {
    return { error: { code, message } }
}

// ---- search_records ----

export interface SearchRecordsArgs {
    query: string
    date_range?: {
        from?: IsoTimestamp
        to?: IsoTimestamp
    }
}

export interface RecordSummary {
    record_id: RecordId
    occurred_at: IsoTimestamp
    channel: string
    /** Summary only. The full text requires a separate get_record call. */
    summary: string
    relatedness: Relatedness
}

export interface SearchRecordsResult {
    /** Always equal to results.length — not a total hit count. */
    count: number
    results: RecordSummary[]
}

/**
 * The similarity numbers stripped out of SearchRecordsResult before it is
 * handed to the model. Written to the trace's `retrieval_hits` column, where
 * phase 4's threshold sweep and PR curve read them.
 */
export interface RetrievalHit {
    record_id: RecordId
    similarity: number
    relatedness: Relatedness
}

// ---- get_unit_history ----

export interface GetUnitHistoryArgs {
    unit_type: UnitType
    /**
     * Unit id or unit name. Names are accepted because on its first turn the
     * agent only has the name it read off the evidence entry — requiring an
     * id would make this tool uncallable exactly when it is needed.
     */
    unit: string
}

export interface UnitRef {
    unit_type: UnitType
    /** null when the system has never established this unit. */
    unit_id: string | null
    name: string
}

export interface EvidenceEntry {
    record_id: RecordId
    occurred_at: IsoTimestamp
    /** Outcome level. Values come from domain config; never hardcoded. */
    outcome: string
    /** Difficulty level, or null when unknown. Values come from domain config. */
    difficulty: string | null
    channel: string
    /** Excerpt only. The full record requires a separate get_record call. */
    excerpt: string
}

/**
 * Never returns not_found. A unit the system has never seen comes back as
 * count 0 with a null unit_id — "no history" is an answer, not a failure,
 * and the agent's response to it differs from its response to a broken tool.
 */
export interface GetUnitHistoryResult {
    unit: UnitRef
    count: number
    /** Ascending by occurred_at, so the list reads chronologically. */
    evidence: EvidenceEntry[]
}

// ---- get_topic_distribution ----

export interface GetTopicDistributionArgs {
    /** Dimension name from domain config. Omit to cover all dimensions. */
    dimension?: string
    /** Lookback window, e.g. '30d'. Defaults to the tool's own window. */
    window?: string
}

export interface SubjectCount {
    name: string
    count: number
    last_seen: IsoTimestamp
}

export interface GetTopicDistributionResult {
    /** null when the caller asked for all dimensions. */
    dimension: string | null
    window: string
    subjects: SubjectCount[]
}

// ---- get_record ----

export interface GetRecordArgs {
    record_id: RecordId
}

/** The only tool result carrying full text. */
export interface RecordDetail {
    record_id: RecordId
    occurred_at: IsoTimestamp
    channel: string
    source_type: SourceType
    /** The user's stated reason for keeping this record; null when absent. */
    description: string | null
    /** Cut to the tool's character limit when `truncated` is true, with a marker appended. */
    content: string
    truncated: boolean
    /** Length of the untruncated content, for the trace. */
    total_chars: number
}

// ---- Registry ----

export const TOOL_NAMES = [
    'search_records',
    'get_unit_history',
    'get_topic_distribution',
    'get_record',
] as const

export type ToolName = (typeof TOOL_NAMES)[number]

/**
 * A tool implementation. Stubs satisfy this today; the real queries will
 * satisfy the same signature without any change on the loop side.
 *
 * `hits` is the out-of-band channel for data the trace wants but the model
 * must not see.
 */
/** Trace-only summary of a tool's result. Each field applies only where relevant. */
export interface ToolResultMeta {
    /** Rows returned. 0 is an empty result, not a failure. */
    count?: number
    /** The output was cut to fit a limit. */
    truncated?: boolean
    /** Length of the output before truncation. */
    totalChars?: number
}

export interface ToolOutcome {
    /** Serialised back to the model as the function call output. */
    result: unknown
    /** Trace-only payload. Never reaches the model. */
    hits?: RetrievalHit[]
    /** Trace-only result summary. Never reaches the model. */
    meta?: ToolResultMeta
}

export interface ToolContext {
    /**
     * Aborted when the tool exceeds its time limit. Real implementations must
     * pass it to their driver, or a timed-out query keeps its connection after
     * the loop has moved on.
     */
    signal: AbortSignal
}

export type ToolHandler = (
    args: Record<string, unknown>,
    ctx: ToolContext
) => Promise<ToolOutcome>
