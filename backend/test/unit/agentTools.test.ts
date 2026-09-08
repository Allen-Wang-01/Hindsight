/**
 * Stub-level tests. These call the tools directly, so they assert on the
 * contract shape rather than on anything the model did.
 */
import { describe, it, expect } from 'vitest';
import {
    MAX_RECORD_CHARS,
    getRecord,
    getTopicDistribution,
    getUnitHistory,
    searchRecords,
} from '../../agent/tools/stubs.js';
import { isToolError } from '../../agent/tools/types.js';
import type {
    GetUnitHistoryResult,
    RecordDetail,
    ToolContext,
    ToolError,
    ToolHandler,
} from '../../agent/tools/types.js';

const CTX: ToolContext = { signal: new AbortController().signal };

/** The fixture's long transcript, the only record over the limit. */
const LONG_RECORD = '66a1f0c2e4b09a1d3c5f7a07';
const SHORT_RECORD = '66a1f0c2e4b09a1d3c5f7a06';

async function fetchRecord(recordId: string): Promise<RecordDetail> {
    const outcome = await getRecord({ record_id: recordId }, CTX);
    if (isToolError(outcome.result)) throw new Error('expected a record, got a ToolError');
    return outcome.result as RecordDetail;
}

describe('get_record truncation', () => {
    it('cuts long content and says so both in the text and in the fields', async () => {
        const record = await fetchRecord(LONG_RECORD);

        expect(record.truncated).toBe(true);
        expect(record.total_chars).toBeGreaterThan(MAX_RECORD_CHARS);
        // The marker sits at the cut, and names how much was dropped.
        expect(record.content.indexOf('[TRUNCATED:')).toBe(MAX_RECORD_CHARS + 2);
        expect(record.content).toContain(String(MAX_RECORD_CHARS));
        expect(record.content).toContain(String(record.total_chars));
    });

    it('leaves short content exactly as it is', async () => {
        const record = await fetchRecord(SHORT_RECORD);

        expect(record.truncated).toBe(false);
        expect(record.content).not.toContain('[TRUNCATED:');
        expect(record.content.length).toBe(record.total_chars);
    });

    it('still reports not_found for an unknown id', async () => {
        const outcome = await getRecord({ record_id: 'nope' }, CTX);
        expect(isToolError(outcome.result)).toBe(true);
    });
});

/** Runs a tool expecting rejection, and returns the error it produced. */
async function reject(
    handler: ToolHandler,
    args: Record<string, unknown>
): Promise<ToolError['error']> {
    const outcome = await handler(args, CTX);
    if (!isToolError(outcome.result)) {
        throw new Error('expected a ToolError, got a result');
    }
    return (outcome.result as ToolError).error;
}

/**
 * The arguments a model can get wrong. Tools reject them as data rather than
 * by throwing, and each message names what would have been accepted — that
 * message is the model's only basis for correcting itself.
 */
describe('tools reject bad arguments', () => {
    it('rejects a missing query rather than searching for nothing', async () => {
        const error = await reject(searchRecords, {});
        expect(error.code).toBe('invalid_argument');
        expect(error.message).toContain('query');
    });

    it('rejects a blank query, which is not the same as no results', async () => {
        const error = await reject(searchRecords, { query: '   ' });
        expect(error.code).toBe('invalid_argument');
    });

    it('rejects a unit_type outside the enum and names both legal values', async () => {
        const error = await reject(getUnitHistory, {
            unit_type: 'skill',
            unit: 'system design',
        });
        expect(error.code).toBe('invalid_argument');
        expect(error.message).toContain('dimension');
        expect(error.message).toContain('subject');
    });

    it('rejects a missing unit', async () => {
        const error = await reject(getUnitHistory, { unit_type: 'dimension' });
        expect(error.code).toBe('invalid_argument');
        expect(error.message).toContain('unit');
    });

    it('rejects a window that is not a day count and shows the shape it wants', async () => {
        const error = await reject(getTopicDistribution, { window: 'last 3 months' });
        expect(error.code).toBe('invalid_argument');
        expect(error.message).toMatch(/30d|90d/);
    });

    it('rejects a missing record_id', async () => {
        const error = await reject(getRecord, {});
        expect(error.code).toBe('invalid_argument');
        expect(error.message).toContain('record_id');
    });

    it('rejects a wrongly typed argument instead of coercing it', async () => {
        const error = await reject(searchRecords, { query: 42 });
        expect(error.code).toBe('invalid_argument');
    });

    it('never throws, whatever it is handed', async () => {
        const handlers: ToolHandler[] = [
            searchRecords,
            getUnitHistory,
            getTopicDistribution,
            getRecord,
        ];
        const junk: Record<string, unknown>[] = [
            {},
            { query: null, unit: null, record_id: null, unit_type: null },
            { query: [], unit_type: {}, window: 0, record_id: false },
        ];

        for (const handler of handlers) {
            for (const args of junk) {
                await expect(handler(args, CTX)).resolves.toBeDefined();
            }
        }
    });
});

describe('tools distinguish empty from broken', () => {
    it('answers an unknown unit with count 0, not an error', async () => {
        const outcome = await getUnitHistory(
            { unit_type: 'dimension', unit: 'a unit nobody has ever recorded' },
            CTX
        );

        expect(isToolError(outcome.result)).toBe(false);
        const result = outcome.result as GetUnitHistoryResult;
        expect(result.count).toBe(0);
        // Null id is how the agent tells "never seen" from "seen, no evidence".
        expect(result.unit.unit_id).toBeNull();
    });
});
