/**
 * Context budget management: estimating how large the conversation is, and
 * shrinking it when it no longer fits.
 *
 * Pure functions. The estimate has to be produced before a call is made — the
 * API only reports real token counts afterwards — so it is derived from
 * character counts and corrected against each turn's reported `input_tokens`.
 *
 * Compaction replaces the oldest tool results with a placeholder before it
 * touches anything else, because the narration that follows a tool call
 * already records what was read out of it. The placeholder keeps the record
 * ids, which conclusions have to cite.
 */
import type OpenAI from 'openai'

type InputItem = OpenAI.Responses.ResponseInputItem
type FunctionCallOutput = OpenAI.Responses.ResponseInputItem.FunctionCallOutput

/** One model call and the tool results that answered it. */
export interface ConversationTurn {
    turn: number
    items: InputItem[]
    /** Record ids this turn's tool calls surfaced. */
    recordIds: string[]
}

/** Starting point before any turn has reported real usage. */
export const DEFAULT_CHARS_PER_TOKEN = 4

/** Estimates are inflated by this much, so compaction triggers early. */
export const SAFETY_MARGIN = 1.15

/** Weight of the newest observation when correcting the ratio. */
const CALIBRATION_WEIGHT = 0.5

export interface ContextEstimator {
    /** Estimated tokens for these items, safety margin included. */
    estimate(items: InputItem[]): number
    /** Corrects the ratio from what a call actually reported. */
    calibrate(items: InputItem[], actualTokens: number): void
    readonly charsPerToken: number
}

function charCount(items: InputItem[]): number {
    let total = 0
    for (const item of items) total += JSON.stringify(item).length
    return total
}

export function createEstimator(): ContextEstimator {
    let charsPerToken = DEFAULT_CHARS_PER_TOKEN

    return {
        get charsPerToken() {
            return charsPerToken
        },

        estimate(items) {
            return Math.ceil((charCount(items) / charsPerToken) * SAFETY_MARGIN)
        },

        calibrate(items, actualTokens) {
            if (actualTokens <= 0) return
            const observed = charCount(items) / actualTokens
            if (!Number.isFinite(observed) || observed <= 0) return
            charsPerToken =
                charsPerToken * (1 - CALIBRATION_WEIGHT) + observed * CALIBRATION_WEIGHT
        },
    }
}

export function flattenConversation(
    preamble: InputItem[],
    turns: ConversationTurn[],
    trailing: InputItem[] = []
): InputItem[] {
    return [...preamble, ...turns.flatMap(turn => turn.items), ...trailing]
}

function isToolOutput(item: InputItem): item is FunctionCallOutput {
    return 'type' in item && item.type === 'function_call_output'
}

function isReasoning(item: InputItem): boolean {
    return 'type' in item && item.type === 'reasoning'
}

const ELIDED_MARKER = '"elided":true'

function isElided(item: InputItem): boolean {
    return (
        isToolOutput(item)
        && 'output' in item
        && typeof item.output === 'string'
        && item.output.includes(ELIDED_MARKER)
    )
}

/**
 * What replaces a dropped tool result. It says the result is gone rather than
 * that it was empty, and carries the record ids so a conclusion can still
 * cite them.
 */
function elidedOutput(recordIds: string[]): string {
    return JSON.stringify({
        elided: true,
        reason: 'context budget',
        note: 'the full result was removed to fit the context; the reading you already recorded for it still stands',
        record_ids: recordIds,
    })
}

function elideToolOutputs(turn: ConversationTurn): ConversationTurn {
    return {
        ...turn,
        items: turn.items.map(item =>
            isToolOutput(item) && !isElided(item)
                ? { ...item, output: elidedOutput(turn.recordIds) }
                : item
        ),
    }
}

function dropReasoning(turn: ConversationTurn): ConversationTurn {
    return { ...turn, items: turn.items.filter(item => !isReasoning(item)) }
}

export interface CompactionResult {
    turns: ConversationTurn[]
    /** Turns whose tool results were replaced. */
    elided: number[]
    /** Turns whose reasoning items were dropped. */
    droppedReasoning: number[]
    estimatedTokens: number
    /** False when the conversation still does not fit. */
    withinBudget: boolean
}

export interface CompactParams {
    preamble: InputItem[]
    turns: ConversationTurn[]
    trailing?: InputItem[]
    maxTokens: number
    estimate: (items: InputItem[]) => number
}

/**
 * Shrinks the conversation until it fits, oldest turns first.
 *
 * Tool results go before reasoning items: a dropped result is already
 * summarised by the narration that follows it, whereas dropping reasoning
 * makes the model re-derive its own train of thought. The preamble, the
 * narration, the function calls themselves and the most recent turn are never
 * touched — dropping a function call would leave its output unpaired, and the
 * newest turn is what the model is answering.
 */
export function compact({
    preamble,
    turns,
    trailing = [],
    maxTokens,
    estimate,
}: CompactParams): CompactionResult {
    const working = [...turns]
    const elided: number[] = []
    const droppedReasoning: number[] = []

    const total = () => estimate(flattenConversation(preamble, working, trailing))

    let estimated = total()
    if (estimated <= maxTokens) {
        return { turns, elided, droppedReasoning, estimatedTokens: estimated, withinBudget: true }
    }

    // The last turn is excluded from both passes.
    const oldest = Math.max(0, working.length - 1)

    for (let index = 0; index < oldest && estimated > maxTokens; index++) {
        const turn = working[index]
        if (!turn || !turn.items.some(item => isToolOutput(item) && !isElided(item))) continue
        working[index] = elideToolOutputs(turn)
        elided.push(turn.turn)
        estimated = total()
    }

    for (let index = 0; index < oldest && estimated > maxTokens; index++) {
        const turn = working[index]
        if (!turn || !turn.items.some(isReasoning)) continue
        working[index] = dropReasoning(turn)
        droppedReasoning.push(turn.turn)
        estimated = total()
    }

    return {
        turns: working,
        elided,
        droppedReasoning,
        estimatedTokens: estimated,
        withinBudget: estimated <= maxTokens,
    }
}
