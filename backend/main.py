from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.config import settings
from backend.logging_config import configure_logging, get_logger
from backend.database import init_db
from backend.services.state_recovery import recover_interrupted_missions
from backend.routers import health, registry, recipes, missions, telemetry, agent, approvals, analysis

configure_logging()
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("astra backend starting (env=%s, autonomy=%s)", settings.env, settings.autonomy_mode)
    await init_db()
    recovered = await recover_interrupted_missions()
    if recovered:
        logger.info("State recovery: acted on %d mission(s).", recovered)
    yield
    logger.info("astra backend shutting down.")


app = FastAPI(
    title="astra API",
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


@app.get("/")
async def root():
    return {"name": "astra", "version": "0.4.0", "docs": "/docs"}
