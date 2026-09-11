/**
 * Harness tests for the mapping from loop callbacks to SSE events.
 */
import { describe, it, expect } from 'vitest';
import { createSseHooks, doneEvent, failedEvent, runStartedEvent } from '../../agent/sseEvents.js';
import type { AgentSseEvent } from '../../agent/sseEvents.js';
import type { AgentRunResult, ToolCallRecord, TurnRecord } from '../../agent/loop.js';

const TOOL_CALL: ToolCallRecord = {
    turn: 1,
    name: 'search_records',
    arguments: { query: 'rate limiting' },
    ok: true,
    durationMs: 12,
    resultMeta: { count: 2 },
};

const TURN: TurnRecord = {
    turn: 1,
    text: 'Checking the record.',
    toolCalls: [TOOL_CALL],
    inputTokens: 100,
    outputTokens: 50,
    cachedTokens: 0,
    reasoningTokens: 20,
    costUsd: 0.0009,
    latencyMs: 4200,
    firstTokenMs: 900,
    attempts: 1,
};

const RESULT: AgentRunResult = {
    inputId: 'deep',
    model: 'gpt-5.6-luna',
    promptVersion: 'agent-v2',
    turns: [TURN],
    finalText: 'Checking the record.',
    stopReason: 'model_done',
    stopDetail: null,
    totals: {
        turns: 1,
        toolCalls: 1,
        inputTokens: 100,
        outputTokens: 50,
        cachedTokens: 0,
        reasoningTokens: 20,
        costUsd: 0.0009,
        latencyMs: 4200,
    },
    retrievalHits: [{ record_id: 'rec-1', similarity: 0.8123, relatedness: 'strong' }],
    injectedFaults: [{ kind: 'tool-error', turn: 1, tool: 'get_record' }],
};

function collect(): { events: AgentSseEvent[]; emit: (event: AgentSseEvent) => void } {
    const events: AgentSseEvent[] = [];
    return { events, emit: event => events.push(event) };
}

describe('sse events', () => {
    it('maps each callback to an event, tagging tool events with the current turn', () => {
        const { events, emit } = collect();
        const hooks = createSseHooks(emit);

        hooks.onTurnStart?.(1);
        hooks.onTextDelta?.('Check');
        hooks.onTextDelta?.('ing.');
        hooks.onToolStart?.('search_records');
        hooks.onToolEnd?.(TOOL_CALL);
        hooks.onTurnEnd?.(TURN);
        hooks.onTurnStart?.(2);
        hooks.onToolStart?.('get_record');

        expect(events).toEqual([
            { type: 'turn_start', turn: 1 },
            { type: 'reasoning_delta', turn: 1, delta: 'Check' },
            { type: 'reasoning_delta', turn: 1, delta: 'ing.' },
            { type: 'tool_start', turn: 1, name: 'search_records' },
            { type: 'tool_end', turn: 1, call: TOOL_CALL },
            { type: 'turn_end', turn: TURN },
            { type: 'turn_start', turn: 2 },
            { type: 'tool_start', turn: 2, name: 'get_record' },
        ]);
    });

    it('reports a dead attempt with the length of the narration to discard', () => {
        const { events, emit } = collect();
        const hooks = createSseHooks(emit);

        hooks.onTurnStart?.(2);
        hooks.onStreamRetry?.({
            attempt: 1,
            reason: 'stream interrupted: connection reset by peer',
            delayMs: 500,
            discardedChars: 42,
        });

        expect(events[1]).toEqual({
            type: 'stream_retry',
            turn: 2,
            attempt: 1,
            reason: 'stream interrupted: connection reset by peer',
            delayMs: 500,
            discardedChars: 42,
        });
    });

    it('accumulating a turn reproduces the turn record', () => {
        const { events, emit } = collect();
        const hooks = createSseHooks(emit);

        hooks.onTurnStart?.(1);
        hooks.onTextDelta?.('Checking ');
        hooks.onTextDelta?.('the record.');
        hooks.onToolEnd?.(TOOL_CALL);
        hooks.onTurnEnd?.(TURN);

        const text = events
            .filter(event => event.type === 'reasoning_delta')
            .map(event => event.delta)
            .join('');
        const toolCalls = events
            .filter(event => event.type === 'tool_end')
            .map(event => event.call);
        const ended = events.find(event => event.type === 'turn_end');

        expect(text).toBe(TURN.text);
        expect(toolCalls).toEqual(TURN.toolCalls);
        expect(ended?.turn).toEqual(TURN);
    });

    it('keeps similarity floats off the wire', () => {
        const { events, emit } = collect();
        const hooks = createSseHooks(emit);

        hooks.onTurnStart?.(1);
        hooks.onToolEnd?.(TOOL_CALL);
        hooks.onTurnEnd?.(TURN);
        emit(runStartedEvent('run-1', 'deep', 'gpt-5.6-luna'));
        emit(doneEvent(RESULT));

        const wire = JSON.stringify(events);
        expect(wire).not.toContain('similarity');
        expect(wire).not.toContain('retrievalHits');
        expect(wire).not.toContain('0.8123');
    });

    it('carries the stop reason, the totals and the injected faults on done', () => {
        expect(doneEvent(RESULT)).toEqual({
            type: 'done',
            stopReason: 'model_done',
            stopDetail: null,
            totals: RESULT.totals,
            injectedFaults: RESULT.injectedFaults,
        });
    });

    it('describes a thrown value as a failure message', () => {
        expect(failedEvent(new Error('boom'))).toEqual({ type: 'failed', message: 'boom' });
        expect(failedEvent('boom')).toEqual({ type: 'failed', message: 'boom' });
    });
});
