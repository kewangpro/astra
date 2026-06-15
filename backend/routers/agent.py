"""
Agent router — triggers the autonomous loop for a mission.
POST /agent/missions/{id}/run  → launches the loop as a background task.
GET  /agent/missions/{id}/plan → returns the current plan.
"""
from __future__ import annotations

import asyncio
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from backend.config import settings
from backend.database import get_db
from backend.models.mission import Mission
from backend.agent.inference.base import InferenceProvider
from backend.agent.inference.mock_provider import MockProvider
from backend.agent.inference.mlx_provider import MLXProvider
from backend.agent.inference.ollama_provider import OllamaProvider
from backend.agent.lead_agent import LeadAgent
from backend.agent.code_generator import CodeGenerator
from backend.agent.error_analyzer import ErrorAnalyzer
from backend.agent.model_manager import ModelManager
from backend.sandbox.manager import sandbox_manager
from backend.evaluator.specialist import SpecialistEvaluator
from backend.loop.state_machine import LoopStateMachine
from backend.logging_config import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/agent", tags=["agent"])


def _make_provider(provider_type: str, model: str) -> InferenceProvider:
    if provider_type == "ollama":
        return OllamaProvider(model_id=model, base_url=settings.ollama_base_url)
    if provider_type == "mlx":
        return MLXProvider(model_id=model)
    return MockProvider()


def _build_loop() -> LoopStateMachine:
    """
    Build a LoopStateMachine with configured inference providers.

    lead_provider  → LeadAgent (planning, pivots)     default: MLX → local MacBook
    code_provider  → CodeGenerator + ErrorAnalyzer     default: MLX → local MacBook
    """
    lead_provider = _make_provider(settings.lead_provider, settings.lead_model)
    code_provider = _make_provider(settings.code_provider, settings.code_model)

    model_manager = ModelManager()
    model_manager.register("lead", lead_provider)
    model_manager.register("code", code_provider)

    logger.info(
        "Agent: lead=%s/%s  code=%s/%s",
        settings.lead_provider, settings.lead_model,
        settings.code_provider, settings.code_model,
    )

    return LoopStateMachine(
        lead_agent=LeadAgent(lead_provider),
        code_generator=CodeGenerator(code_provider),
        error_analyzer=ErrorAnalyzer(code_provider),
        model_manager=model_manager,
        sandbox_manager=sandbox_manager,
        evaluator=SpecialistEvaluator(),
    )


# Tracks running mission tasks so they can be cancelled on shutdown
_running_tasks: dict[str, asyncio.Task] = {}


@router.post("/missions/{mission_id}/run", status_code=202)
async def run_mission(mission_id: str, db: AsyncSession = Depends(get_db)):
    """Launch the autonomous loop for a mission as a cancellable asyncio task."""
    mission = await db.get(Mission, mission_id)
    if not mission:
        raise HTTPException(status_code=404, detail="Mission not found")
    if mission.status not in ("pending", "paused"):
        raise HTTPException(status_code=409, detail=f"Mission is already in state '{mission.status}'")
    if mission_id in _running_tasks and not _running_tasks[mission_id].done():
        raise HTTPException(status_code=409, detail="Mission loop already running")

    loop = _build_loop()
    task = asyncio.create_task(loop.run(mission_id))
    _running_tasks[mission_id] = task
    task.add_done_callback(lambda t: _running_tasks.pop(mission_id, None))
    logger.info("Agent: launched loop for mission=%s", mission_id)
    return {"mission_id": mission_id, "status": "loop_started"}


@router.get("/missions/{mission_id}/plan")
async def get_plan(mission_id: str, db: AsyncSession = Depends(get_db)):
    mission = await db.get(Mission, mission_id)
    if not mission:
        raise HTTPException(status_code=404, detail="Mission not found")
    return {"mission_id": mission_id, "plan": mission.current_plan}
