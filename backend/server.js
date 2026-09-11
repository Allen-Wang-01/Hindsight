import express from 'express'
import cors from 'cors'
import dotenv from 'dotenv'
import connectDB from './db.js'
import authRoutes from './routes/auth.js'
import cookieParser from 'cookie-parser';
import noteRouter from './routes/note.js'
import reportRoutes from './routes/reports.js'
import { apiRateLimiter, authRateLimiter } from './middleware/rateLimiters.js'
import { env } from './config/env.js'
import { checkConnection } from './config/postgres.js'
import mcpRouter from './routes/mcp.js'
import oauthDiscoveryRouter from './routes/oauthDiscovery.js'
import savedMemoriesRouter from './routes/savedMemories.js'
import agentRouter from './routes/agent.js'
import { installShutdownHandlers } from './lib/shutdown.js'
dotenv.config();

const app = express();

/**
 * Trust the first proxy (required when running behind Fly.io, Vercel, Nginx, etc.)
 * This must be set BEFORE any middleware that relies on client IP,
 * such as rate limiting or authentication.
 */
app.set("trust proxy", 1);

app.use(cookieParser());
app.use(cors({
    origin: env.clientOrigin,
    credentials: true,
}))
app.use(express.json({ limit: '1mb' }));

connectDB()

// /.well-known/* must be at root — no prefix — so OAuth clients can discover them from MCP_BASE_URL
app.use(oauthDiscoveryRouter)
app.use('/api/auth', authRateLimiter, authRoutes);
app.use('/api/notes', noteRouter)
app.use('/api/reports', apiRateLimiter, reportRoutes)
app.use('/api/mcp', mcpRouter)
app.use('/api/saved-memories', savedMemoriesRouter)
app.use('/api/agent', agentRouter)

app.use((err, req, res, next) => {
    res.status(500).json({ message: "Internal Server Error" });
})

app.get('/', (req, res) => {
    res.send('Auth Backend is running')
})



const PORT = process.env.PORT || 3001;
const server = app.listen(PORT, async () => {
    console.log(`Server running on port ${PORT}`);
    await checkConnection() // check postgreSQL connection
    // if (process.env.NODE_ENV === 'production') {
    //     const { startWeeklyReportJob } = await import('./jobs/generateWeeklyReports.js');
    //     startWeeklyReportJob()
    //     console.log('[Server] Cron jobs loaded: weekly (Sun 00:00), monthly (1st 00:05) JST');
    // }
});

installShutdownHandlers(server);

