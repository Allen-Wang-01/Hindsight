/**
 * Graceful shutdown on SIGTERM / SIGINT.
 *
 * Stops accepting connections, aborts the agent runs in flight so each one
 * closes out its trace, lets the BullMQ workers finish what they are holding,
 * then exits. A second signal exits immediately.
 */
import type { Server } from 'http'
import mongoose from 'mongoose'
import { abortAllRuns } from '../agent/runRegistry.js'
import { createLogger } from '../utils/logger.js'
import { aiWorkerController } from '../workers/aiWorkerController.js'
import { reportWorkerController } from '../workers/reportWorkerController.js'

const log = createLogger('shutdown')

/** Upper bound on the whole sequence. Must stay below fly.toml's kill_timeout. */
const SHUTDOWN_TIMEOUT_MS = 25_000

const SIGNALS = ['SIGTERM', 'SIGINT'] as const

let shuttingDown = false

export function installShutdownHandlers(server: Server): void {
    for (const signal of SIGNALS) {
        process.on(signal, () => {
            void handleSignal(signal, server)
        })
    }
}

async function handleSignal(signal: string, server: Server): Promise<void> {
    if (shuttingDown) {
        log.warn('shutdown_forced', { signal })
        process.exit(1)
    }
    shuttingDown = true
    log.info('shutdown_started', { signal })

    const forceExit = setTimeout(() => {
        log.error('shutdown_timed_out', { timeoutMs: SHUTDOWN_TIMEOUT_MS })
        process.exit(1)
    }, SHUTDOWN_TIMEOUT_MS)
    forceExit.unref?.()

    try {
        // Stops accepting new connections right away; the promise settles once
        // the open ones have ended, which is why it is awaited last.
        const serverClosed = closeServer(server)

        await abortAllRuns()
        server.closeIdleConnections?.()

        await Promise.all([
            aiWorkerController.shutdown({ ifIdle: false }),
            reportWorkerController.shutdown({ ifIdle: false }),
        ])

        await serverClosed
        await mongoose.connection.close()
    } catch (err) {
        log.error('shutdown_error', {
            error: err instanceof Error ? err.message : String(err),
        })
    }

    clearTimeout(forceExit)
    log.info('shutdown_complete')
    process.exit(0)
}

function closeServer(server: Server): Promise<void> {
    return new Promise(resolve => {
        server.close(() => resolve())
    })
}
