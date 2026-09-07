/**
 * Stub-level tests. These call the tools directly, so they assert on the
 * contract shape rather than on anything the model did.
 */
import { describe, it, expect } from 'vitest';
import { MAX_RECORD_CHARS, getRecord } from '../../agent/tools/stubs.js';
import { isToolError } from '../../agent/tools/types.js';
import type { RecordDetail, ToolContext } from '../../agent/tools/types.js';

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
