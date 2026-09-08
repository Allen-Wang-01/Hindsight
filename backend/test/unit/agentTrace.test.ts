/**
 * Harness tests for the trace rows an agent run writes.
 *
 * lib/trace.js is mocked, so no database client is ever constructed and the
 * assertions are on the row payloads themselves — the shape step D reads.
 */
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import type OpenAI from 'openai';
import { createFaultPlan } from '../../agent/faults.js';
import { runAgent } from '../../agent/loop.js';
import type { AgentClient } from '../../agent/loop.js';
import { createRunTracer } from '../../agent/runTrace.js';
import type { AgentTraceExtra } from '../../agent/runTrace.js';
import type { TracePayload } from '../../lib/trace.js';
import type { FixtureInput } from '../../agent/tools/stubs.js';

const { insertTraceMock } = vi.hoisted(() => ({ insertTraceMock: vi.fn() }));

vi.mock('../../lib/trace.js', () => ({
    insertTrace: insertTraceMock,
    hashInput: (text: string) => `sha256-${text.length}`,
}));

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

const NO_DELAY = [0, 0];

function textDelta(delta: string): StreamEvent {
    return { type: 'response.output_text.delta', delta } as unknown as StreamEvent;
}

function terminal(response: Record<string, unknown>): StreamEvent {
    return {
        type: 'response.completed',
        response: { output: [], ...response },
    } as unknown as StreamEvent;
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

interface Attempt {
    events: StreamEvent[];
    thenThrow?: Error;
    then?: () => void;
}

function createFakeClient(attempts: Attempt[]): AgentClient {
    let index = 0;
    return {
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
}

/** Every payload handed to insertTrace, in write order. */
function rows(): TracePayload[] {
    return insertTraceMock.mock.calls.map((call) => call[0] as TracePayload);
}

function extraOf(row: TracePayload): AgentTraceExtra {
    return row.extra as AgentTraceExtra;
}

beforeEach(() => {
    insertTraceMock.mockReset();
    insertTraceMock.mockResolvedValue(undefined);
});

afterEach(() => {
    vi.restoreAllMocks();
    vi.unstubAllEnvs();
});

async function run(
    client: AgentClient,
    options: { faults?: ReturnType<typeof createFaultPlan>; signal?: AbortSignal } = {}
): Promise<void> {
    const tracer = createRunTracer({
        model: 'gpt-5.6-luna',
        promptVersion: 'agent-test',
        input: INPUT,
        origin: 'test',
        ...(options.faults ? { faults: options.faults } : {}),
    });

    const result = await runAgent({
        input: INPUT,
        client,
        retryDelaysMs: NO_DELAY,
        hooks: tracer.hooks,
        ...(options.faults ? { faults: options.faults } : {}),
        ...(options.signal ? { signal: options.signal } : {}),
    });

    await tracer.finish(result);
}

describe('agent trace rows', () => {
    it('writes one row per turn, with stop_reason only on the last', async () => {
        const client = createFakeClient([
            {
                events: [
                    terminal({
                        output: [
                            message('Checking whether this has come up before.'),
                            functionCall('get_unit_history', {
                                unit_type: 'dimension',
                                unit: 'system design',
                            }),
                        ],
                    }),
                ],
            },
            { events: [terminal({ output: [message('The conclusion.')] })] },
        ]);

        await run(client);

        expect(rows()).toHaveLength(2);
        const [first, last] = rows() as [TracePayload, TracePayload];

        expect(first.turnIndex).toBe(1);
        expect(first.status).toBe('ok');
        expect(first.reasoning).toBe('Checking whether this has come up before.');
        expect(first.stopReason).toBeUndefined();
        expect(first.output).toBeUndefined();

        // Filled in after onTurnEnd returns, which is why the row is held back.
        expect(first.toolCalls).toHaveLength(1);

        expect(last.turnIndex).toBe(2);
        expect(last.stopReason).toBe('model_done');
        expect(last.output).toBe('The conclusion.');
        expect(last.reasoning).toBe('The conclusion.');
    });

    it('records the tool result summary alongside the call', async () => {
        const client = createFakeClient([
            {
                events: [
                    terminal({
                        output: [
                            message('Looking for anything on record.'),
                            functionCall('get_unit_history', {
                                unit_type: 'dimension',
                                unit: 'a unit nobody has ever seen',
                            }),
                        ],
                    }),
                ],
            },
            { events: [terminal({ output: [message('Nothing on record.')] })] },
        ]);

        await run(client);

        const toolCalls = rows()[0]?.toolCalls as Array<{
            ok: boolean;
            resultMeta?: { count?: number };
        }>;
        // An unknown unit is a successful empty result, not a failure.
        expect(toolCalls[0]?.ok).toBe(true);
        expect(toolCalls[0]?.resultMeta?.count).toBe(0);
    });

    it('gives a dead attempt its own error row under the same turn_index', async () => {
        const client = createFakeClient([
            { events: [textDelta('Half a sen')], thenThrow: new Error('ECONNRESET') },
            {
                events: [
                    textDelta('The whole conclusion.'),
                    terminal({ output: [message('The whole conclusion.')] }),
                ],
            },
        ]);

        await run(client);

        expect(rows()).toHaveLength(2);
        const [failed, recovered] = rows() as [TracePayload, TracePayload];

        expect(failed.status).toBe('error');
        expect(failed.turnIndex).toBe(1);
        expect(failed.error).toContain('ECONNRESET');
        // The half sentence the viewer saw, so a replay can show it being dropped.
        expect(failed.reasoning).toBe('Half a sen');
        expect(extraOf(failed).discardedChars).toBe(10);
        expect(extraOf(failed).attempt).toBe(1);
        expect(failed.costUsd).toBeUndefined();

        expect(recovered.status).toBe('ok');
        expect(recovered.turnIndex).toBe(1);
        expect(extraOf(recovered).attempt).toBe(2);
        expect(recovered.stopReason).toBe('model_done');
    });

    it('marks injected faults so they can be excluded from failure rates', async () => {
        vi.stubEnv('AGENT_FAULT_INJECTION', 'on');
        const plan = createFaultPlan([{ kind: 'tool-error', code: 'invalid_argument' }]);

        const client = createFakeClient([
            {
                events: [
                    terminal({
                        output: [
                            message('Opening the record.'),
                            functionCall('get_record', { record_id: 'whatever' }),
                        ],
                    }),
                ],
            },
            { events: [terminal({ output: [message('That failed, so here is what I have.')] })] },
        ]);

        await run(client, { faults: plan });

        const all = rows();
        expect(all.every((row) => extraOf(row).faultInjection === true)).toBe(true);

        const fired = all.flatMap((row) => extraOf(row).injectedFaults ?? []);
        expect(fired).toHaveLength(1);
        expect(fired[0]?.kind).toBe('tool-error');
        expect(fired[0]?.turn).toBe(1);
    });

    it('still writes a final row when the run is cancelled mid-stream', async () => {
        const controller = new AbortController();
        const client = createFakeClient([
            {
                events: [
                    terminal({
                        output: [
                            message('Starting to look.'),
                            functionCall('get_topic_distribution', {}),
                        ],
                    }),
                ],
            },
            // Cancelled while the second turn streams: that turn is discarded,
            // so the first turn's row has to be the one carrying 'aborted'.
            { events: [textDelta('Still thi')], then: () => controller.abort() },
        ]);

        await run(client, { signal: controller.signal });

        expect(rows()).toHaveLength(1);
        const only = rows()[0] as TracePayload;
        expect(only.turnIndex).toBe(1);
        expect(only.stopReason).toBe('aborted');
        expect(only.reasoning).toBe('Starting to look.');
    });

    it('records the input once and the retrieval floats only on the trace', async () => {
        const client = createFakeClient([
            {
                events: [
                    terminal({
                        output: [
                            message('Searching.'),
                            functionCall('search_records', { query: 'rate limiting bucket' }),
                        ],
                    }),
                ],
            },
            { events: [terminal({ output: [message('Done.')] })] },
        ]);

        await run(client);

        const all = rows();
        expect(all.filter((row) => row.input !== undefined)).toHaveLength(1);
        expect(all[0]?.inputHash).toMatch(/^sha256-/);

        // Similarity floats reach the trace and nothing else.
        const hits = all.at(-1)?.retrievalHits as Array<{ similarity: number }> | undefined;
        expect(hits?.length).toBeGreaterThan(0);
        expect(typeof hits?.[0]?.similarity).toBe('number');
        expect(JSON.stringify(all[0]?.toolCalls)).not.toContain('similarity');
    });
});
