/**
 * Per-model token pricing, in USD per 1,000,000 tokens.
 * Source: provider pricing pages, captured 2026-08-19.
 */

export interface ModelPrice {
    input: number          // USD per 1M input tokens
    cachedInput?: number   // USD per 1M cached input tokens; omitted when the model has no cache pricing
    output?: number        // USD per 1M output tokens; omitted for embedding models
}

export const PRICING_VERSION = '2026-08'

export const MODEL_PRICING: Record<string, ModelPrice> = {
    'gpt-5.6-luna': {
        input: 0.20,
        cachedInput: 0.02,
        output: 1.20,
    },
    'gpt-5-nano': {
        input: 0.05,
        cachedInput: 0.005,
        output: 0.40,
    },
    'text-embedding-3-small': {
        input: 0.02,
    },
}

interface ComputeCostParams {
    model: string
    inputTokens?: number
    outputTokens?: number
    cachedTokens?: number
}

/**
 * Computes the USD cost of one LLM call from its token counts.
 * Returns null when `model` isn't in MODEL_PRICING — callers must not
 * fall back to a default price, since a silently wrong cost is more
 * dangerous than a missing one. The result is unrounded; rounding
 * happens at the database column (NUMERIC), not here.
 */
export function computeCost({
    model,
    inputTokens = 0,
    outputTokens = 0,
    cachedTokens = 0,
}: ComputeCostParams): number | null {
    const price = MODEL_PRICING[model]
    if (!price) {
        return null
    }

    const uncachedInputTokens = inputTokens - cachedTokens
    const inputCost = (uncachedInputTokens / 1_000_000) * price.input
    const cachedCost = price.cachedInput !== undefined
        ? (cachedTokens / 1_000_000) * price.cachedInput
        : 0
    const outputCost = price.output !== undefined
        ? (outputTokens / 1_000_000) * price.output
        : 0

    return inputCost + cachedCost + outputCost
}
