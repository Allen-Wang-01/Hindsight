import rateLimit from "express-rate-limit";

/**
 * Rate limiter for normal API requests (CRUD, regenerate, etc.)
 */
export const apiRateLimiter = rateLimit({
    windowMs: 60 * 1000, // 1 minute
    max: 60,             // max 60 requests per minute per IP
    standardHeaders: true,
    legacyHeaders: false,
});

/**
 * Rate limiter for SSE endpoints (long-lived connections)
 */
export const sseRateLimiter = rateLimit({
    windowMs: 60 * 1000, // 1 minute
    max: 10,             // max 10 SSE connections per minute per IP
    standardHeaders: true,
    legacyHeaders: false,
});

/**
 * Rate limiter for starting agent runs. Each run is several model calls, so
 * this is far tighter than the ordinary API limit.
 */
export const agentRunRateLimiter = rateLimit({
    windowMs: 5 * 60 * 1000, // 5 minutes
    max: 20,                 // 20 runs per IP per 5 minutes
    standardHeaders: true,
    legacyHeaders: false,
});

/**
 * Rate limiter for watching an agent run. Looser than starting one: attaching
 * costs nothing, and a page refresh must not lock the viewer out.
 */
export const agentStreamRateLimiter = rateLimit({
    windowMs: 60 * 1000, // 1 minute
    max: 60,
    standardHeaders: true,
    legacyHeaders: false,
});

export const authRateLimiter = rateLimit({
    windowMs: 15 * 60 * 1000, // 15 minutes
    max: 20,                 // 20 attempts per IP
});

/**
 * Rate limiter for the public, unauthenticated OAuth Dynamic Client Registration
 * endpoint (/api/mcp/register). This is the highest-priority limit: /register is
 * unauthenticated and writes a document to MongoDB on every call, so without a
 * cap it can be spammed to bloat the database. A normal Claude connection
 * registers once per connector setup, so a generous per-IP cap that tolerates
 * retries (and Claude's shared egress IPs) still blocks abusive volume.
 */
export const mcpRegisterRateLimiter = rateLimit({
    windowMs: 60 * 60 * 1000, // 1 hour
    max: 20,                  // 20 registrations per IP per hour
    standardHeaders: true,
    legacyHeaders: false,
});

/**
 * Rate limiter for the public OAuth token + revocation endpoints
 * (/api/mcp/token, /api/mcp/revoke). These hit Redis and perform JWT signing on
 * every call. The cap is higher than registration because a user with several
 * connections legitimately refreshes hourly (each ~1h access-token expiry), plus
 * the initial code exchange and occasional revokes.
 */
export const mcpTokenRateLimiter = rateLimit({
    windowMs: 15 * 60 * 1000, // 15 minutes
    max: 60,                  // 60 token/revoke requests per IP per 15 min
    standardHeaders: true,
    legacyHeaders: false,
});