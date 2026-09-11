/**
 * Harness tests for the run registry: fan-out, replay, ceilings, cancellation
 * and the wall-clock deadline. The run body is a stub promise, so nothing here
 * touches the model or the database.
 */
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import type { ServerResponse } from 'http';
import {
    ABORT_CANCELLED,
    ABORT_DEADLINE,
    ABORT_SHUTDOWN,
    MAX_ACTIVE_RUNS,
    MAX_ACTIVE_RUNS_PER_USER,
    RUN_DEADLINE_MS,
    RunLimitError,
    abortAllRuns,
    activeRunCount,
    cancelRun,
    resetRegistry,
    startRun,
    subscribe,
    unsubscribe,
} from '../../agent/runRegistry.js';
import type { RunContext } from '../../agent/runRegistry.js';
import type { AgentSseEvent } from '../../agent/sseEvents.js';

const USER = 'user-1';
const OTHER_USER = 'user-2';

function fakeResponse() {
    const chunks: string[] = [];
    let ended = false;
    const res = {
        write(chunk: string) {
            chunks.push(chunk);
            return true;
        },
        end() {
            ended = true;
        },
    };
    return {
        res: res as unknown as ServerResponse,
        isEnded: () => ended,
        events(): AgentSseEvent[] {
            return chunks.map(chunk => JSON.parse(chunk.replace(/^data: /, '').trim()));
        },
    };
}

/** A run that stays in flight until released, exposing its context. */
function controllableRun() {
    let release!: () => void;
    const gate = new Promise<void>(resolve => {
        release = resolve;
    });
    let context: RunContext | null = null;

    return {
        release,
        ctx: () => {
            if (!context) throw new Error('run has not started');
            return context;
        },
        run: async (ctx: RunContext) => {
            context = ctx;
            await gate;
        },
    };
}

/** A run that finishes only when it is aborted. */
function abortableRun() {
    let context: RunContext | null = null;
    return {
        ctx: () => context,
        run: (ctx: RunContext) =>
            new Promise<void>(resolve => {
                context = ctx;
                ctx.signal.addEventListener('abort', () => resolve());
            }),
    };
}

beforeEach(() => {
    resetRegistry();
    vi.spyOn(console, 'log').mockImplementation(() => { });
});

afterEach(() => {
    vi.restoreAllMocks();
});

describe('run registry', () => {
    it('delivers events to a subscriber attached before they are emitted', async () => {
        const subject = controllableRun();
        startRun({ runId: 'run-1', userId: USER, run: subject.run });

        const viewer = fakeResponse();
        expect(subscribe('run-1', USER, viewer.res)).toBe(true);

        subject.ctx().emit({ type: 'turn_start', turn: 1 });
        subject.ctx().emit({ type: 'reasoning_delta', turn: 1, delta: 'look' });

        expect(viewer.events()).toEqual([
            { type: 'turn_start', turn: 1 },
            { type: 'reasoning_delta', turn: 1, delta: 'look' },
        ]);

        subject.release();
    });

    it('replays what a late subscriber missed, merging narration per turn', async () => {
        const subject = controllableRun();
        startRun({ runId: 'run-1', userId: USER, run: subject.run });

        const ctx = subject.ctx();
        ctx.emit({ type: 'turn_start', turn: 1 });
        ctx.emit({ type: 'reasoning_delta', turn: 1, delta: 'I am ' });
        ctx.emit({ type: 'reasoning_delta', turn: 1, delta: 'checking' });
        ctx.emit({ type: 'turn_start', turn: 2 });
        ctx.emit({ type: 'reasoning_delta', turn: 2, delta: 'now' });

        const viewer = fakeResponse();
        subscribe('run-1', USER, viewer.res);

        // Consecutive deltas of one turn arrive as a single block; a new turn
        // starts a new one.
        expect(viewer.events()).toEqual([
            { type: 'turn_start', turn: 1 },
            { type: 'reasoning_delta', turn: 1, delta: 'I am checking' },
            { type: 'turn_start', turn: 2 },
            { type: 'reasoning_delta', turn: 2, delta: 'now' },
        ]);

        subject.release();
    });

    it('ends the response of a subscriber attached after the run finished', async () => {
        const subject = controllableRun();
        startRun({ runId: 'run-1', userId: USER, run: subject.run });
        subject.ctx().emit({ type: 'turn_start', turn: 1 });
        subject.release();
        await vi.waitFor(() => expect(activeRunCount()).toBe(0));

        const viewer = fakeResponse();
        expect(subscribe('run-1', USER, viewer.res)).toBe(true);
        expect(viewer.events()).toEqual([{ type: 'turn_start', turn: 1 }]);
        expect(viewer.isEnded()).toBe(true);
    });

    it('ends live subscribers when the run finishes', async () => {
        const subject = controllableRun();
        startRun({ runId: 'run-1', userId: USER, run: subject.run });

        const viewer = fakeResponse();
        subscribe('run-1', USER, viewer.res);
        expect(viewer.isEnded()).toBe(false);

        subject.release();
        await vi.waitFor(() => expect(viewer.isEnded()).toBe(true));
    });

    it('stops delivering to a viewer that unsubscribed', async () => {
        const subject = controllableRun();
        startRun({ runId: 'run-1', userId: USER, run: subject.run });

        const viewer = fakeResponse();
        subscribe('run-1', USER, viewer.res);
        unsubscribe('run-1', viewer.res);
        subject.ctx().emit({ type: 'turn_start', turn: 1 });

        expect(viewer.events()).toEqual([]);
        subject.release();
    });

    it('refuses a subscriber that does not own the run', () => {
        const subject = controllableRun();
        startRun({ runId: 'run-1', userId: USER, run: subject.run });

        const viewer = fakeResponse();
        expect(subscribe('run-1', OTHER_USER, viewer.res)).toBe(false);
        expect(subscribe('missing', USER, viewer.res)).toBe(false);

        subject.release();
    });

    it('rejects a run past the per-user ceiling', () => {
        const runs = [];
        for (let i = 0; i < MAX_ACTIVE_RUNS_PER_USER; i++) {
            const subject = controllableRun();
            runs.push(subject);
            startRun({ runId: `run-${i}`, userId: USER, run: subject.run });
        }

        try {
            startRun({ runId: 'one-too-many', userId: USER, run: controllableRun().run });
            expect.unreachable('the ceiling should have thrown');
        } catch (err) {
            expect(err).toBeInstanceOf(RunLimitError);
            expect((err as RunLimitError).scope).toBe('user');
        }

        for (const subject of runs) subject.release();
    });

    it('rejects a run past the global ceiling', () => {
        const runs = [];
        let started = 0;
        // Spread across enough accounts that the per-user ceiling is never the
        // one that trips.
        for (let user = 0; started < MAX_ACTIVE_RUNS; user++) {
            for (let i = 0; i < MAX_ACTIVE_RUNS_PER_USER && started < MAX_ACTIVE_RUNS; i++) {
                const subject = controllableRun();
                runs.push(subject);
                startRun({ runId: `run-${started}`, userId: `user-${user}`, run: subject.run });
                started += 1;
            }
        }

        expect(activeRunCount()).toBe(MAX_ACTIVE_RUNS);

        try {
            startRun({ runId: 'one-too-many', userId: 'fresh-user', run: controllableRun().run });
            expect.unreachable('the ceiling should have thrown');
        } catch (err) {
            expect(err).toBeInstanceOf(RunLimitError);
            expect((err as RunLimitError).scope).toBe('global');
        }

        for (const subject of runs) subject.release();
    });

    it('cancels a run with a reason and reports the run finished afterwards', async () => {
        const subject = abortableRun();
        startRun({ runId: 'run-1', userId: USER, run: subject.run });

        expect(cancelRun('run-1', USER)).toBe('cancelled');
        expect(subject.ctx()?.signal.aborted).toBe(true);
        expect(subject.ctx()?.signal.reason).toBe(ABORT_CANCELLED);

        await vi.waitFor(() => expect(activeRunCount()).toBe(0));
        expect(cancelRun('run-1', USER)).toBe('already_finished');
    });

    it('will not cancel a run belonging to someone else', async () => {
        const subject = abortableRun();
        startRun({ runId: 'run-1', userId: USER, run: subject.run });

        expect(cancelRun('run-1', OTHER_USER)).toBe('not_found');
        expect(cancelRun('missing', USER)).toBe('not_found');
        expect(subject.ctx()?.signal.aborted).toBe(false);

        cancelRun('run-1', USER);
        await vi.waitFor(() => expect(activeRunCount()).toBe(0));
    });

    it('aborts a run that outlives the deadline', async () => {
        vi.useFakeTimers();
        try {
            const subject = abortableRun();
            startRun({ runId: 'run-1', userId: USER, run: subject.run });

            await vi.advanceTimersByTimeAsync(RUN_DEADLINE_MS - 1);
            expect(subject.ctx()?.signal.aborted).toBe(false);

            await vi.advanceTimersByTimeAsync(1);
            expect(subject.ctx()?.signal.aborted).toBe(true);
            expect(subject.ctx()?.signal.reason).toBe(ABORT_DEADLINE);
            expect(activeRunCount()).toBe(0);
        } finally {
            vi.useRealTimers();
        }
    });

    it('aborts everything on shutdown and waits for it to unwind', async () => {
        const first = abortableRun();
        const second = abortableRun();
        startRun({ runId: 'run-1', userId: USER, run: first.run });
        startRun({ runId: 'run-2', userId: OTHER_USER, run: second.run });

        await abortAllRuns();

        expect(first.ctx()?.signal.reason).toBe(ABORT_SHUTDOWN);
        expect(second.ctx()?.signal.reason).toBe(ABORT_SHUTDOWN);
        expect(activeRunCount()).toBe(0);
    });

    it('reports a run that threw as a failed event', async () => {
        const run = async () => {
            throw new Error('boom');
        };
        startRun({ runId: 'run-1', userId: USER, run });

        await vi.waitFor(() => expect(activeRunCount()).toBe(0));

        const viewer = fakeResponse();
        subscribe('run-1', USER, viewer.res);
        expect(viewer.events()).toEqual([{ type: 'failed', message: 'boom' }]);
    });
});
