"""
Agent router — triggers the autonomous loop for a mission.
POST /agent/missions/{id}/run  → launches the loop as a background task.
GET  /agent/missions/{id}/plan → returns the current plan.
"""
from __future__ import annotations

import asyncio
from typing import Optional

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from backend.database import get_db
from backend.models.mission import Mission
from backend.agent.inference.mock_provider import MockProvider
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


def _build_loop() -> LoopStateMachine:
    """
    Build a LoopStateMachine with the default provider.
    Phase 3 production: swap MockProvider → MLXProvider after model download.
    """
    provider = MockProvider()
    model_manager = ModelManager()
    model_manager.register("lead", provider)

    return LoopStateMachine(
        lead_agent=LeadAgent(provider),
        code_generator=CodeGenerator(provider),
        error_analyzer=ErrorAnalyzer(provider),
        model_manager=model_manager,
        sandbox_manager=sandbox_manager,
        evaluator=SpecialistEvaluator(),
    )


@router.post("/missions/{mission_id}/run", status_code=202)
async def run_mission(mission_id: str, background_tasks: BackgroundTasks, db: AsyncSession = Depends(get_db)):
    """Launch the autonomous loop for a mission in the background."""
    mission = await db.get(Mission, mission_id)
    if not mission:
        raise HTTPException(status_code=404, detail="Mission not found")
    if mission.status not in ("pending", "paused"):
        raise HTTPException(status_code=409, detail=f"Mission is already in state '{mission.status}'")

    loop = _build_loop()
    background_tasks.add_task(loop.run, mission_id)
    logger.info("Agent: launched loop for mission=%s", mission_id)
    return {"mission_id": mission_id, "status": "loop_started"}


@router.get("/missions/{mission_id}/plan")
async def get_plan(mission_id: str, db: AsyncSession = Depends(get_db)):
    mission = await db.get(Mission, mission_id)
    if not mission:
        raise HTTPException(status_code=404, detail="Mission not found")
    return {"mission_id": mission_id, "plan": mission.current_plan}
