/**
 * Harness tests for the agent loop: deterministic, zero API calls, milliseconds.
 *
 * What they cover is the loop's own retry layer, not the model. The SDK stops
 * retrying once the response headers are in, so every failure scripted here is
 * one that reaches the loop as a half-delivered stream.
 */
import { describe, it, expect, vi } from 'vitest';
import type OpenAI from 'openai';
import { runAgent } from '../../agent/loop.js';
import type { AgentClient, StreamRetry } from '../../agent/loop.js';
import type { FixtureInput } from '../../agent/tools/stubs.js';

type StreamEvent = OpenAI.Responses.ResponseStreamEvent;

const INPUT: FixtureInput = {
    id: 'test',
    note: 'harness fixture',
    evidence: {
        unit_type: 'dimension',
        unit: 'system design',
        outcome: 'partial',
        difficulty: 'medium',
        occurred_at: '2026-09-01T10:00:00Z',
        excerpt: 'Named token bucket but could not distinguish it from leaky bucket.',
    },
};

// Retries with no backoff: the delays are real behaviour, but waiting for them
// would make the suite slow without testing anything extra.
const NO_DELAY = [0, 0];

function textDelta(delta: string): StreamEvent {
    return { type: 'response.output_text.delta', delta } as unknown as StreamEvent;
}

/** A terminal event carrying a minimal Response the loop can parse. */
function terminal(
    type: 'response.completed' | 'response.incomplete' | 'response.failed',
    response: Record<string, unknown>
): StreamEvent {
    return { type, response: { output: [], ...response } } as unknown as StreamEvent;
}

function message(text: string): Record<string, unknown> {
    return { type: 'message', content: [{ type: 'output_text', text }] };
}

function functionCall(name: string, args: Record<string, unknown>): Record<string, unknown> {
    return { type: 'function_call', call_id: 'call_1', name, arguments: JSON.stringify(args) };
}

interface Attempt {
    events: StreamEvent[];
    /** Thrown after the events are yielded — a connection dying mid-stream. */
    thenThrow?: Error;
    /**
     * Called after the events are yielded, before the stream ends. Aborting
     * here reproduces what the SDK really does on cancellation: it swallows the
     * abort and the stream just stops, with no terminal event.
     */
    then?: () => void;
}

/** Replays one scripted attempt per call, so a retry gets the next script. */
function createFakeClient(attempts: Attempt[]): { client: AgentClient; calls: () => number } {
    let index = 0;
    const client: AgentClient = {
        responses: {
            create: async () => {
                const attempt = attempts[index++];
                if (!attempt) throw new Error('fake client ran out of scripted attempts');
                return (async function* () {
                    for (const event of attempt.events) yield event;
                    attempt.then?.();
                    if (attempt.thenThrow) throw attempt.thenThrow;
                })();
            },
        },
    };
    return { client, calls: () => index };
}

describe('runAgent stream retries', () => {
    it('retries a stream that dies mid-flight and keeps the recovered turn', async () => {
        const { client, calls } = createFakeClient([
            { events: [textDelta('Half a sen')], thenThrow: new Error('ECONNRESET') },
            {
                events: [
                    textDelta('The whole conclusion.'),
                    terminal('response.completed', { output: [message('The whole conclusion.')] }),
                ],
            },
        ]);
        const onStreamRetry = vi.fn<(retry: StreamRetry) => void>();

        const result = await runAgent({
            input: INPUT,
            client,
            retryDelaysMs: NO_DELAY,
            hooks: { onStreamRetry },
        });

        expect(calls()).toBe(2);
        expect(result.stopReason).toBe('model_done');
        expect(result.finalText).toBe('The whole conclusion.');
        // One turn, two model calls — the dead attempt is not a turn of its own.
        expect(result.totals.turns).toBe(1);
        expect(result.turns[0]?.attempts).toBe(2);

        // The renderer has already printed the half sentence and has to be told
        // how much of it to drop.
        expect(onStreamRetry).toHaveBeenCalledTimes(1);
        const retry = onStreamRetry.mock.calls[0]?.[0];
        expect(retry?.attempt).toBe(1);
        expect(retry?.discardedChars).toBe('Half a sen'.length);
        expect(retry?.reason).toContain('stream interrupted');
    });

    it('retries a stream that ends without a terminal event', async () => {
        const { client, calls } = createFakeClient([
            { events: [textDelta('cut off')] },
            { events: [terminal('response.completed', { output: [message('Done.')] })] },
        ]);

        const result = await runAgent({ input: INPUT, client, retryDelaysMs: NO_DELAY });

        expect(calls()).toBe(2);
        expect(result.turns[0]?.attempts).toBe(2);
        expect(result.finalText).toBe('Done.');
    });

    it('retries a response that comes back failed', async () => {
        const { client, calls } = createFakeClient([
            {
                events: [
                    terminal('response.failed', {
                        error: { code: 'server_error', message: 'upstream blew up' },
                    }),
                ],
            },
            { events: [terminal('response.completed', { output: [message('Done.')] })] },
        ]);

        const result = await runAgent({ input: INPUT, client, retryDelaysMs: NO_DELAY });

        expect(calls()).toBe(2);
        expect(result.turns[0]?.attempts).toBe(2);
    });

    it('throws once the retries are exhausted', async () => {
        const { client, calls } = createFakeClient([
            { events: [], thenThrow: new Error('ECONNRESET') },
            { events: [], thenThrow: new Error('ECONNRESET') },
            { events: [], thenThrow: new Error('ECONNRESET') },
        ]);

        await expect(
            runAgent({ input: INPUT, client, retryDelaysMs: NO_DELAY })
        ).rejects.toThrow(/gave up after 3 attempts/);

        // Three attempts, not four: the delay list sets the ceiling.
        expect(calls()).toBe(3);
    });

    it('does not retry a truncated response', async () => {
        const { client, calls } = createFakeClient([
            {
                events: [
                    textDelta('Half a sen'),
                    terminal('response.incomplete', {
                        output: [message('Half a sen')],
                        incomplete_details: { reason: 'max_output_tokens' },
                    }),
                ],
            },
        ]);

        const result = await runAgent({ input: INPUT, client, retryDelaysMs: NO_DELAY });

        // Retrying would hit the same ceiling and bill for it twice.
        expect(calls()).toBe(1);
        expect(result.stopReason).toBe('incomplete');
        expect(result.stopDetail).toBe('max_output_tokens');
        expect(result.turns[0]?.attempts).toBe(1);
    });
});

describe('runAgent cancellation', () => {
    it('stops mid-stream and returns a result instead of throwing', async () => {
        const controller = new AbortController();
        const { client, calls } = createFakeClient([
            // No terminal event after the abort — exactly what the SDK does.
            { events: [textDelta('Half a sen')], then: () => controller.abort() },
        ]);

        const result = await runAgent({
            input: INPUT,
            client,
            retryDelaysMs: NO_DELAY,
            signal: controller.signal,
        });

        expect(result.stopReason).toBe('aborted');
        // A cancelled stream is indistinguishable from an interrupted one, so
        // this is the assertion that keeps cancelling from buying three retries.
        expect(calls()).toBe(1);
        // The turn was in flight: no usage, no conclusion, so it is not a turn.
        expect(result.totals.turns).toBe(0);
        expect(result.totals.costUsd).toBe(0);
    });

    it('keeps the turn already paid for and skips the tool it had not started', async () => {
        const controller = new AbortController();
        const { client, calls } = createFakeClient([
            {
                events: [
                    terminal('response.completed', {
                        output: [
                            message('Checking the history.'),
                            functionCall('get_unit_history', {
                                unit_type: 'dimension',
                                unit: 'system design',
                            }),
                        ],
                    }),
                ],
            },
        ]);

        const result = await runAgent({
            input: INPUT,
            client,
            retryDelaysMs: NO_DELAY,
            signal: controller.signal,
            // Cancelled after the turn completed, before its tool call runs.
            hooks: { onTurnEnd: () => controller.abort() },
        });

        expect(result.stopReason).toBe('aborted');
        expect(calls()).toBe(1);
        // The turn happened and was billed, so it stays on the bill.
        expect(result.totals.turns).toBe(1);
        expect(result.totals.toolCalls).toBe(0);
        expect(result.turns[0]?.toolCalls).toEqual([]);
    });

    it('makes no model call at all when the signal is already aborted', async () => {
        const controller = new AbortController();
        controller.abort();
        const { client, calls } = createFakeClient([
            { events: [terminal('response.completed', { output: [message('Done.')] })] },
        ]);

        const result = await runAgent({
            input: INPUT,
            client,
            retryDelaysMs: NO_DELAY,
            signal: controller.signal,
        });

        expect(calls()).toBe(0);
        expect(result.stopReason).toBe('aborted');
        expect(result.totals.turns).toBe(0);
    });
});
