/**
 * Harness tests for the agent loop: deterministic, zero API calls, milliseconds.
 *
 * What they cover is the loop's own retry layer, not the model. The SDK stops
 * retrying once the response headers are in, so every failure scripted here is
 * one that reaches the loop as a half-delivered stream.
 */
import { describe, it, expect, vi, afterEach } from 'vitest';
import type OpenAI from 'openai';
import { createFaultPlan } from '../../agent/faults.js';
import { DEFAULT_BUDGET, runAgent } from '../../agent/loop.js';
import type { AgentClient, StreamRetry } from '../../agent/loop.js';
import { TOOL_HANDLERS } from '../../agent/tools/stubs.js';
import type { FixtureInput } from '../../agent/tools/stubs.js';
import type { ToolContext } from '../../agent/tools/types.js';

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

function functionCall(
    name: string,
    args: Record<string, unknown>,
    callId = 'call_1'
): Record<string, unknown> {
    return { type: 'function_call', call_id: callId, name, arguments: JSON.stringify(args) };
}

/** A completed turn that narrates and then asks for one tool. */
function toolTurn(name: string, args: Record<string, unknown>, callId?: string): StreamEvent {
    return terminal('response.completed', {
        output: [message('Looking that up.'), functionCall(name, args, callId)],
    });
}

/** A completed turn that only narrates — the loop's intended exit. */
function finalTurn(text = 'Done.'): StreamEvent {
    return terminal('response.completed', { output: [message(text)] });
}

afterEach(() => {
    vi.restoreAllMocks();
    vi.unstubAllEnvs();
});

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
        const { client, calls } = createFakeClient([{ events: [finalTurn()] }]);

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

describe('runAgent tool failures', () => {
    it('converts a throwing tool into an internal ToolError', async () => {
        vi.spyOn(TOOL_HANDLERS, 'get_topic_distribution').mockRejectedValue(
            new Error('connection pool exhausted')
        );
        const { client } = createFakeClient([
            { events: [toolTurn('get_topic_distribution', { window: '90d' })] },
            { events: [finalTurn()] },
        ]);

        const result = await runAgent({ input: INPUT, client, retryDelaysMs: NO_DELAY });

        // The run survives: the contract says tools return errors, so a throw
        // is converted rather than propagated.
        expect(result.stopReason).toBe('model_done');
        const call = result.turns[0]?.toolCalls[0];
        expect(call?.ok).toBe(false);
        expect(call?.errorCode).toBe('internal');
    });

    it('times a hanging tool out and aborts its signal', async () => {
        let captured: ToolContext | undefined;
        vi.spyOn(TOOL_HANDLERS, 'search_records').mockImplementation(async (_args, ctx) => {
            captured = ctx;
            return new Promise(() => {});
        });
        const { client } = createFakeClient([
            { events: [toolTurn('search_records', { query: 'rate limiting' })] },
            { events: [finalTurn()] },
        ]);

        const result = await runAgent({
            input: INPUT,
            client,
            retryDelaysMs: NO_DELAY,
            toolTimeoutMs: 20,
        });

        expect(result.turns[0]?.toolCalls[0]?.errorCode).toBe('timeout');
        // Freeing the loop is not enough — the query has to be told to stop.
        expect(captured?.signal.aborted).toBe(true);
    });

    it('answers a repeated call with duplicate_call and does not run the tool twice', async () => {
        const handler = vi.spyOn(TOOL_HANDLERS, 'get_record');
        const args = { record_id: '66a1f0c2e4b09a1d3c5f7a01' };
        const { client } = createFakeClient([
            { events: [toolTurn('get_record', args, 'call_1')] },
            { events: [toolTurn('get_record', args, 'call_2')] },
            { events: [finalTurn()] },
        ]);

        const result = await runAgent({ input: INPUT, client, retryDelaysMs: NO_DELAY });

        expect(handler).toHaveBeenCalledTimes(1);
        expect(result.turns[0]?.toolCalls[0]?.ok).toBe(true);

        const repeat = result.turns[1]?.toolCalls[0];
        expect(repeat?.ok).toBe(false);
        expect(repeat?.errorCode).toBe('duplicate_call');
        // Still billed against the ceiling, so a looping model still terminates.
        expect(result.totals.toolCalls).toBe(2);
    });

    it('matches duplicates regardless of argument key order', async () => {
        const handler = vi.spyOn(TOOL_HANDLERS, 'get_unit_history');
        const { client } = createFakeClient([
            { events: [toolTurn('get_unit_history', { unit_type: 'dimension', unit: 'system design' })] },
            { events: [toolTurn('get_unit_history', { unit: 'system design', unit_type: 'dimension' })] },
            { events: [finalTurn()] },
        ]);

        const result = await runAgent({ input: INPUT, client, retryDelaysMs: NO_DELAY });

        expect(handler).toHaveBeenCalledTimes(1);
        expect(result.turns[1]?.toolCalls[0]?.errorCode).toBe('duplicate_call');
    });
});

describe('runAgent cost ceiling', () => {
    const USAGE = { usage: { input_tokens: 20_000, output_tokens: 5_000 } };

    it('forces convergence once the run has spent its budget', async () => {
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
                        ...USAGE,
                    }),
                ],
            },
            { events: [finalTurn('Concluding from what I have.')] },
        ]);

        const result = await runAgent({
            input: INPUT,
            client,
            retryDelaysMs: NO_DELAY,
            budget: { ...DEFAULT_BUDGET, maxCostUsd: 0.001 },
        });

        expect(result.stopReason).toBe('max_cost');
        // Two calls: the turn that spent the budget, then the convergence call
        // that still has to produce a conclusion.
        expect(calls()).toBe(2);
        expect(result.finalText).toBe('Concluding from what I have.');
        expect(result.totals.costUsd).toBeGreaterThan(0.001);
    });

    it('leaves the run alone when the ceiling is not reached', async () => {
        const { client } = createFakeClient([
            { events: [terminal('response.completed', { output: [message('Done.')], ...USAGE })] },
        ]);

        const result = await runAgent({ input: INPUT, client, retryDelaysMs: NO_DELAY });

        expect(result.stopReason).toBe('model_done');
    });
});

describe('fault injection', () => {
    it('builds no plan while the env flag is off', () => {
        vi.stubEnv('AGENT_FAULT_INJECTION', '');
        expect(createFaultPlan([{ kind: 'tool-error' }])).toBeNull();
    });

    it('fails one tool call, records it, and leaves the next one alone', async () => {
        vi.stubEnv('AGENT_FAULT_INJECTION', 'on');
        const args = { unit_type: 'dimension', unit: 'system design' };
        const { client } = createFakeClient([
            { events: [toolTurn('get_unit_history', args, 'call_1')] },
            { events: [toolTurn('get_topic_distribution', {}, 'call_2')] },
            { events: [finalTurn()] },
        ]);

        const result = await runAgent({
            input: INPUT,
            client,
            retryDelaysMs: NO_DELAY,
            faults: createFaultPlan([
                { kind: 'tool-error', tool: 'get_unit_history', code: 'invalid_argument' },
            ]),
        });

        expect(result.turns[0]?.toolCalls[0]?.errorCode).toBe('invalid_argument');
        // Spent: the second tool call runs for real.
        expect(result.turns[1]?.toolCalls[0]?.ok).toBe(true);
        expect(result.injectedFaults).toEqual([
            { kind: 'tool-error', turn: 1, tool: 'get_unit_history', code: 'invalid_argument' },
        ]);
    });

    it('hangs a tool so the real timeout path ends it', async () => {
        vi.stubEnv('AGENT_FAULT_INJECTION', 'on');
        const { client } = createFakeClient([
            { events: [toolTurn('search_records', { query: 'rate limiting' })] },
            { events: [finalTurn()] },
        ]);

        const result = await runAgent({
            input: INPUT,
            client,
            retryDelaysMs: NO_DELAY,
            toolTimeoutMs: 20,
            faults: createFaultPlan([{ kind: 'tool-timeout', tool: 'search_records' }]),
        });

        expect(result.turns[0]?.toolCalls[0]?.errorCode).toBe('timeout');
        expect(result.injectedFaults[0]?.kind).toBe('tool-timeout');
    });

    it('kills one stream mid-flight and recovers on the retry', async () => {
        vi.stubEnv('AGENT_FAULT_INJECTION', 'on');
        const onStreamRetry = vi.fn<(retry: StreamRetry) => void>();
        const { client, calls } = createFakeClient([
            { events: [textDelta('A sentence long enough to be worth discarding when it dies.')] },
            { events: [finalTurn('The whole conclusion.')] },
        ]);

        const result = await runAgent({
            input: INPUT,
            client,
            retryDelaysMs: NO_DELAY,
            hooks: { onStreamRetry },
            faults: createFaultPlan([{ kind: 'stream-interrupt', turn: 1 }]),
        });

        expect(calls()).toBe(2);
        expect(result.stopReason).toBe('model_done');
        expect(result.finalText).toBe('The whole conclusion.');
        expect(result.turns[0]?.attempts).toBe(2);
        expect(onStreamRetry).toHaveBeenCalledTimes(1);
        expect(result.injectedFaults).toEqual([{ kind: 'stream-interrupt', turn: 1 }]);
    });
});
