/**
 * Harness tests for context estimation and compaction.
 *
 * Compaction has no single correct output, so most of these assert the
 * properties it must never break rather than a specific result.
 */
import { describe, it, expect } from 'vitest';
import type OpenAI from 'openai';
import {
    DEFAULT_CHARS_PER_TOKEN,
    compact,
    createEstimator,
    flattenConversation,
} from '../../agent/context.js';
import type { ConversationTurn } from '../../agent/context.js';

type InputItem = OpenAI.Responses.ResponseInputItem;

const PREAMBLE: InputItem[] = [
    { role: 'developer', content: 'you review one piece of evidence' },
    { role: 'user', content: 'evidence: stalled on leaky bucket' },
];

function reasoning(id: string, size = 200): InputItem {
    return {
        type: 'reasoning',
        id,
        summary: [],
        encrypted_content: 'r'.repeat(size),
    } as unknown as InputItem;
}

function message(text: string): InputItem {
    return {
        type: 'message',
        role: 'assistant',
        status: 'completed',
        id: `msg_${text.length}`,
        content: [{ type: 'output_text', text, annotations: [] }],
    } as unknown as InputItem;
}

function functionCall(callId: string, name: string): InputItem {
    return {
        type: 'function_call',
        call_id: callId,
        name,
        arguments: '{}',
    } as unknown as InputItem;
}

/** Carries the record ids the way a real tool result does. */
function functionCallOutput(callId: string, recordIds: string[], size = 500): InputItem {
    return {
        type: 'function_call_output',
        call_id: callId,
        output: JSON.stringify({
            results: recordIds.map(id => ({ record_id: id })),
            content: 'c'.repeat(size),
        }),
    } as unknown as InputItem;
}

/** A turn with narration, a tool call and its result. */
function turn(index: number, recordIds: string[], outputSize = 500): ConversationTurn {
    const callId = `call_${index}`;
    return {
        turn: index,
        items: [
            reasoning(`rs_${index}`),
            message(`turn ${index} narration`),
            functionCall(callId, 'get_record'),
            functionCallOutput(callId, recordIds, outputSize),
        ],
        recordIds,
    };
}

/** One token per character, so budgets in these tests are easy to reason about. */
const byChars = (items: InputItem[]) =>
    items.reduce((total, item) => total + JSON.stringify(item).length, 0);

function itemsOfType(items: InputItem[], type: string): InputItem[] {
    return items.filter(item => 'type' in item && item.type === type);
}

/** The four properties compaction must preserve, whatever it drops. */
function expectInvariants(before: InputItem[], after: InputItem[], recordIds: string[]): void {
    // 1. every function_call still has its output, and vice versa
    const callIds = (items: InputItem[], type: string) =>
        itemsOfType(items, type)
            .map(item => (item as { call_id: string }).call_id)
            .sort();
    expect(callIds(after, 'function_call')).toEqual(callIds(after, 'function_call_output'));

    // 2. the preamble survives untouched
    expect(after.slice(0, PREAMBLE.length)).toEqual(PREAMBLE);

    // 3. every record id is still findable
    const serialised = JSON.stringify(after);
    for (const id of recordIds) expect(serialised).toContain(id);

    // 4. it actually got smaller
    expect(byChars(after)).toBeLessThan(byChars(before));
}

describe('context estimator', () => {
    it('scales with the size of the conversation', () => {
        const estimator = createEstimator();
        const small = estimator.estimate([message('short')]);
        const large = estimator.estimate([message('l'.repeat(4000))]);

        expect(small).toBeGreaterThan(0);
        expect(large).toBeGreaterThan(small * 10);
    });

    it('moves the ratio towards what a call actually reported', () => {
        const estimator = createEstimator();
        const items = [message('x'.repeat(1000))];
        const before = estimator.estimate(items);

        // Reporting far fewer tokens than assumed means each token covers more
        // characters, so later estimates come down.
        estimator.calibrate(items, Math.round(byChars(items) / 10));

        expect(estimator.charsPerToken).toBeGreaterThan(DEFAULT_CHARS_PER_TOKEN);
        expect(estimator.estimate(items)).toBeLessThan(before);
    });

    it('ignores a usage report that cannot be true', () => {
        const estimator = createEstimator();
        estimator.calibrate([message('x')], 0);
        expect(estimator.charsPerToken).toBe(DEFAULT_CHARS_PER_TOKEN);
    });
});

describe('context compaction', () => {
    it('changes nothing when the conversation already fits', () => {
        const turns = [turn(1, ['rec-1']), turn(2, ['rec-2'])];
        const result = compact({
            preamble: PREAMBLE,
            turns,
            maxTokens: 1_000_000,
            estimate: byChars,
        });

        expect(result.elided).toEqual([]);
        expect(result.droppedReasoning).toEqual([]);
        expect(result.turns).toBe(turns);
        expect(result.withinBudget).toBe(true);
    });

    it('elides the oldest tool result first and keeps the newest turn whole', () => {
        const turns = [turn(1, ['rec-1']), turn(2, ['rec-2']), turn(3, ['rec-3'])];
        const before = flattenConversation(PREAMBLE, turns);

        const result = compact({
            preamble: PREAMBLE,
            turns,
            maxTokens: Math.round(byChars(before) * 0.75),
            estimate: byChars,
        });

        expect(result.elided[0]).toBe(1);
        expect(result.elided).not.toContain(3);

        const after = flattenConversation(PREAMBLE, result.turns);
        expectInvariants(before, after, ['rec-1', 'rec-2', 'rec-3']);
    });

    it('keeps the narration of a turn whose result it dropped', () => {
        const turns = [turn(1, ['rec-1']), turn(2, ['rec-2'])];
        const result = compact({
            preamble: PREAMBLE,
            turns,
            maxTokens: Math.round(byChars(flattenConversation(PREAMBLE, turns)) * 0.7),
            estimate: byChars,
        });

        const narration = JSON.stringify(result.turns[0]?.items);
        expect(narration).toContain('turn 1 narration');
        expect(narration).not.toContain('ccccc');
    });

    it('drops reasoning only once every older result is already elided', () => {
        const turns = [turn(1, ['rec-1']), turn(2, ['rec-2']), turn(3, ['rec-3'])];
        const before = flattenConversation(PREAMBLE, turns);

        // Tight enough that eliding the results cannot get there on its own.
        const result = compact({
            preamble: PREAMBLE,
            turns,
            maxTokens: Math.round(byChars(before) * 0.35),
            estimate: byChars,
        });

        expect(result.droppedReasoning.length).toBeGreaterThan(0);
        // Everything it could elide, it elided before touching reasoning.
        expect(result.elided).toEqual([1, 2]);
        expect(result.droppedReasoning).not.toContain(3);

        expectInvariants(before, flattenConversation(PREAMBLE, result.turns), [
            'rec-1',
            'rec-2',
            'rec-3',
        ]);
    });

    it('reports that it could not fit rather than dropping what it must keep', () => {
        const turns = [turn(1, ['rec-1']), turn(2, ['rec-2'])];
        const result = compact({
            preamble: PREAMBLE,
            turns,
            maxTokens: 10,
            estimate: byChars,
        });

        expect(result.withinBudget).toBe(false);

        const after = flattenConversation(PREAMBLE, result.turns);
        expect(after.slice(0, PREAMBLE.length)).toEqual(PREAMBLE);
        expect(JSON.stringify(after)).toContain('rec-1');
    });

    it('tells the model the result was removed rather than empty', () => {
        const turns = [turn(1, ['rec-1', 'rec-9']), turn(2, ['rec-2'])];
        const result = compact({
            preamble: PREAMBLE,
            turns,
            maxTokens: Math.round(byChars(flattenConversation(PREAMBLE, turns)) * 0.7),
            estimate: byChars,
        });

        const elided = result.turns[0]?.items.find(
            item => 'type' in item && item.type === 'function_call_output'
        ) as { output: string } | undefined;
        const payload = JSON.parse(elided?.output ?? '{}');

        expect(payload.elided).toBe(true);
        expect(payload.record_ids).toEqual(['rec-1', 'rec-9']);
    });

    it('does not elide the same turn twice across successive calls', () => {
        const turns = [turn(1, ['rec-1']), turn(2, ['rec-2']), turn(3, ['rec-3'])];
        const budget = Math.round(byChars(flattenConversation(PREAMBLE, turns)) * 0.75);

        const first = compact({ preamble: PREAMBLE, turns, maxTokens: budget, estimate: byChars });
        const second = compact({
            preamble: PREAMBLE,
            turns: first.turns,
            maxTokens: budget,
            estimate: byChars,
        });

        expect(second.elided).toEqual([]);
    });
});
