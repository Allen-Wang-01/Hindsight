const DEFAULT_IDLE_TIMEOUT = 20_000;     // 20s automatic shutdown when idle
const DEFAULT_CHECK_INTERVAL = 5_000;   // check every 5s

export function createWorkerController({
    name,
    startWorker,
    idleTimeout = DEFAULT_IDLE_TIMEOUT,
    checkInterval = DEFAULT_CHECK_INTERVAL,
}) {
    let worker = null
    let idleTimer = null
    let lastActiveAt = 0
    let starting = false
    let activeJobCount = 0

    function log(...args) {
        console.log(`[WorkerController: ${name}]`, ...args);
    }

    function touch() {
        lastActiveAt = Date.now()
    }

    async function ensureRunning() {
        if (worker || starting) {
            touch()
            return
        }

        starting = true
        log("Starting worker")

        try {
            worker = await startWorker()
            touch()
            attachWorkerLifecycle(worker)
            startIdleWatcher()
            log("Worker started")
        } catch (err) {
            log("Failed to start worker: ", err)
            worker = null
            throw err
        } finally {
            starting = false
        }
    }

    function attachWorkerLifecycle(workerInstance) {
        // any job activity refreshes idle timer
        workerInstance.on("active", () => {
            activeJobCount++
            touch()
        })
        workerInstance.on("completed", () => {
            activeJobCount = Math.max(0, activeJobCount - 1)
            touch()
        })
        workerInstance.on("failed", () => {
            activeJobCount = Math.max(0, activeJobCount - 1)
            touch()
        })
        workerInstance.on("error", (err) => {
            log("Worker error: ", err)
        })
    }

    function startIdleWatcher() {
        if (idleTimer) return

        idleTimer = setInterval(async () => {
            try {
                if (!worker) return
                const idleFor = Date.now() - lastActiveAt
                if (idleFor < idleTimeout) return

                if (activeJobCount > 0) {
                    touch()
                    return
                }

                log(`Idle for ${idleFor}ms, shutting down worker...`)
                await shutdown()
            } catch (err) {
                log("Idle watcher error:", err)
            }
        }, checkInterval)

        // Prevent timer from keeping Node process alive
        if (idleTimer.unref) {
            idleTimer.unref()
        }
    }

    /**
     * Closes the worker.
     *
     * `ifIdle` (the default) does nothing while jobs are active — the idle
     * watcher calls it that way and looks again on its next tick. Pass false to
     * close regardless: BullMQ stops taking new jobs and waits for the active
     * ones, which is what a process shutting down needs.
     */
    async function shutdown({ ifIdle = true } = {}) {
        if (!worker) return
        if (ifIdle && activeJobCount > 0) {
            log(`Shutdown skipped: ${activeJobCount} job(s) still active`)
            return
        }
        log("Closing worker...")
        try {
            await worker.close() // shutdown
        } catch (err) {
            log("Error while closing worker:", err)
        } finally {
            worker = null
            lastActiveAt = 0
            if (idleTimer) clearInterval(idleTimer)
            idleTimer = null
            log("worker closed")
        }
    }
    return {
        ensureRunning,
        shutdown
    }
}