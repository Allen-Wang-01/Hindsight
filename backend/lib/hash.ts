import crypto from 'crypto'

/**
 * sha256 hex digest of a text input.
 *
 * Lives on its own rather than inside trace.ts because that module constructs
 * a Postgres client at import time. Modules that only need a fingerprint —
 * the agent prompt and tool definitions, which hash themselves at load — must
 * not drag a database connection in behind them.
 *
 * Used for input_hash (which doubles as the embedding cache key) and for the
 * prompt / tool-definition fingerprints recorded on every agent trace row.
 */
export function hashInput(text: string): string {
    return crypto.createHash('sha256').update(text, 'utf8').digest('hex')
}
