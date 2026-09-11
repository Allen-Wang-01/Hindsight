import { Router } from 'express'
import {
    cancelAgentRun,
    createAgentRun,
    listAgentInputs,
    streamAgentRun,
} from '../controllers/agentController.js'
import { agentRunRateLimiter, agentStreamRateLimiter, apiRateLimiter } from '../middleware/rateLimiters.js'
import authMiddleware from '../middleware/authMiddleware.js'

const router = Router()

router.use(authMiddleware)
router.get('/inputs', apiRateLimiter, listAgentInputs)
router.post('/runs', agentRunRateLimiter, createAgentRun)
// The path must end in /stream: authMiddleware authenticates those from the
// refresh cookie, which is all EventSource can send.
router.get('/runs/:runId/stream', agentStreamRateLimiter, streamAgentRun)
router.delete('/runs/:runId', apiRateLimiter, cancelAgentRun)

export default router
