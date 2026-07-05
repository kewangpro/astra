import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.config import settings
from backend.logging_config import configure_logging, get_logger
from backend.database import init_db
from backend.services.state_recovery import recover_interrupted_missions
from backend.routers import health, registry, recipes, missions, telemetry, agent, approvals, analysis, play

configure_logging()
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("ASTRA backend starting (env=%s, autonomy=%s)", settings.env, settings.autonomy_mode)
    await init_db()
    recovered = await recover_interrupted_missions()
    restart_ids, resume_ids = recovered["restart"], recovered["resume"]
    if restart_ids or resume_ids:
        logger.info("State recovery: acted on %d mission(s) (%d restart, %d resume).",
                    len(restart_ids) + len(resume_ids), len(restart_ids), len(resume_ids))
        for mission_id in restart_ids:
            loop = agent._build_loop()
            task = asyncio.create_task(loop.run(mission_id))
            agent._running_tasks[mission_id] = task
            task.add_done_callback(lambda t, mid=mission_id: agent._running_tasks.pop(mid, None))
            logger.info("State recovery: auto-restarted loop for mission=%s", mission_id)
        for mission_id in resume_ids:
            loop = agent._build_loop()
            task = asyncio.create_task(loop.run(mission_id, resume_existing_sandbox=True))
            agent._running_tasks[mission_id] = task
            task.add_done_callback(lambda t, mid=mission_id: agent._running_tasks.pop(mid, None))
            logger.info("State recovery: reattached loop for mission=%s", mission_id)
    yield
    # Cancel all running mission loops so uvicorn can reload without blocking
    tasks = list(agent._running_tasks.values())
    if tasks:
        logger.info("ASTRA backend shutting down — cancelling %d mission task(s).", len(tasks))
        for t in tasks:
            t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
    logger.info("ASTRA backend shutting down.")


app = FastAPI(
    title="ASTRA API",
    description="Autonomous Strategic Training Agent — backend orchestration layer",
    version="0.4.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router)
app.include_router(registry.router)
app.include_router(recipes.router)
app.include_router(missions.router)
app.include_router(telemetry.router)
app.include_router(agent.router)
app.include_router(approvals.router)
app.include_router(analysis.router)
app.include_router(play.router)


@app.get("/")
async def root():
    return {"name": "ASTRA", "version": "0.4.0", "docs": "/docs"}
