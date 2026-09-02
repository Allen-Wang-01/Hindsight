-- =====================================================
-- Table: llm_traces
-- Purpose: Append-only log of every LLM call made by the
-- backend (note analysis, report summary, embeddings, ...).
-- This is the ground truth for token counts, cost, latency,
-- and prompt/schema versions — every eval and cost number in
-- the project is derived from this table.
--
-- Core fields are typed; anything call-site-specific or still
-- evolving goes into `extra` (JSONB) rather than a new column.
-- This is an exploration-phase table: additive only, never
-- backfilled or ALTERed — leave columns NULL until a phase
-- actually populates them.
-- =====================================================
CREATE TABLE IF NOT EXISTS llm_traces (
    id           UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    created_at   TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    -- Subject
    user_id      TEXT,           -- MongoDB User _id as string; NULL for calls with no user context
    record_id    TEXT,           -- MongoDB Note / Report _id as string; NULL when the call isn't tied to one record
    call_site    TEXT NOT NULL,  -- 'note_analysis' | 'report_summary' | 'note_embedding' | ... — identifies the calling code path
    call_type    TEXT NOT NULL,  -- 'completion' | 'embedding'

    -- Model and versions
    model            TEXT NOT NULL,  -- model id used for this call, e.g. 'gpt-5-nano'
    prompt_version   TEXT,           -- set once prompts are versioned (phase 2a+); NULL until then
    schema_version   TEXT,           -- set once structured-output schemas are versioned (phase 2a+); NULL until then
    pricing_version  TEXT,           -- set once cost calculation lands (phase 0.2+); NULL until then

    -- Input / output (embedding rows leave input and output NULL — they log input_hash/input_chars instead)
    input          JSONB,   -- structured input sent to the model (e.g. prompt + context), when applicable
    output         TEXT,    -- raw text output for completion calls; NULL for embedding calls
    input_chars    INTEGER, -- character length of the input text, for both completion and embedding calls
    input_hash     TEXT,    -- sha256 hex of the input text; doubles as the embedding cache key
    reasoning      TEXT,    -- model reasoning trace; filled from phase 5 onward, NULL for now
    tool_calls     JSONB,   -- sequence of tool calls made during the request; filled from phase 5 onward, NULL for now
    retrieval_hits JSONB,   -- ids/scores of retrieval results used as context; filled from phase 2a/5 onward, NULL for now

    -- Accounting
    input_tokens    INTEGER,       -- prompt/input token count, when reported by the provider
    output_tokens   INTEGER,       -- completion/output token count, when reported by the provider
    total_tokens    INTEGER,       -- total token count, when reported by the provider
    cost_usd        NUMERIC(12,6), -- computed cost in USD; NULL until phase 0.2 adds pricing calculation
    latency_ms      INTEGER,       -- wall-clock duration of the call
    first_token_ms  INTEGER,       -- time to first streamed token; only set for streaming call sites, NULL otherwise

    -- Result
    status  TEXT NOT NULL, -- 'ok' | 'error'
    error   TEXT,          -- error message when status = 'error'; NULL otherwise

    extra   JSONB -- catch-all for call-site-specific data that doesn't warrant its own column yet
);

-- Recent-first scan, the default access pattern for inspection/debugging
CREATE INDEX IF NOT EXISTS idx_llm_traces_created_at
    ON llm_traces (created_at DESC);

-- Look up every trace tied to a specific Note/Report
CREATE INDEX IF NOT EXISTS idx_llm_traces_record_id
    ON llm_traces (record_id);

-- Analysis almost always filters by call_type first (e.g. cost of completions vs embeddings)
CREATE INDEX IF NOT EXISTS idx_llm_traces_call_type_created_at
    ON llm_traces (call_type, created_at DESC);

-- =====================================================
-- Migration (2026-08-21): widen cost_usd precision.
-- Reason: a single embedding call can cost well under $0.000001;
-- NUMERIC(12,6) rounds costs that small down to 0.
-- =====================================================
ALTER TABLE llm_traces
    ALTER COLUMN cost_usd TYPE NUMERIC(16,10);

-- =====================================================
-- Migration (2026-09-01): agent-run structure.
--
-- One agent run writes one row per turn, as each turn completes — a run that
-- crashes still leaves its finished turns behind, which is exactly the data
-- failure analysis needs. These three columns are what makes those rows
-- addressable as a run rather than a pile of unrelated calls.
-- =====================================================
ALTER TABLE llm_traces
    -- One complete agent run: one evidence item and all of its turns.
    -- Aggregate by this for per-run metrics — tool calls, cost, total latency.
    -- Cannot be reconstructed after the fact: which turns grouped into which
    -- run is known only at write time.
    ADD COLUMN IF NOT EXISTS run_id      TEXT,

    -- 1-based turn number within the run. created_at cannot order turns
    -- reliably — two turns can land in the same millisecond.
    ADD COLUMN IF NOT EXISTS turn_index  INTEGER,

    -- 'model_done' | 'max_turns' | 'max_tool_calls'. Set ONLY on a run's final
    -- row, since it is not known until the run ends and this table is
    -- append-only. Consequence worth keeping: a run_id whose rows all have
    -- NULL stop_reason is a run that crashed.
    ADD COLUMN IF NOT EXISTS stop_reason TEXT;

-- Replaying one run in order; also the lookup the trajectory UI does
CREATE INDEX IF NOT EXISTS idx_llm_traces_run_id_turn
    ON llm_traces (run_id, turn_index);
