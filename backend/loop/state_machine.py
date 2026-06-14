"""
LoopStateMachine — Step 3.3.

Implements the Plan → Implement → Sandbox → Execute → Eval → Refine loop.
Each state transition is persisted atomically in the Mission Store.
Respects autonomy mode (guided/supervised/full_autonomy) for approval gates.
"""
from __future__ import annotations

import asyncio
import os
from enum import Enum
from typing import Optional

from sqlalchemy import select, update

from backend.database import AsyncSessionLocal
from backend.models.mission import Mission, MissionStatus
from backend.models.approval import ApprovalGate, ApprovalStatus, GateType
from backend.agent.lead_agent import LeadAgent
from backend.agent.code_generator import CodeGenerator
from backend.agent.error_analyzer import ErrorAnalyzer
from backend.agent.model_manager import ModelManager
from backend.sandbox.manager import SandboxManager
from backend.evaluator.specialist import SpecialistEvaluator
from backend.loop.pivots import PivotEngine
from backend.config import settings
from backend.logging_config import get_logger

logger = get_logger(__name__)

MAX_RETRIES = 3          # max error-fix iterations before marking FAILED
EVAL_POLL_INTERVAL = 10  # seconds between sandbox liveness checks


class LoopState(str, Enum):
    PLANNING = "planning"
    IMPLEMENTING = "implementing"
    SANDBOXING = "sandboxing"
    EXECUTING = "executing"
    EVALUATING = "evaluating"
    REFINING = "refining"
    COMPLETED = "completed"
    FAILED = "failed"


class LoopStateMachine:
    def __init__(
        self,
        lead_agent: LeadAgent,
        code_generator: CodeGenerator,
        error_analyzer: ErrorAnalyzer,
        model_manager: ModelManager,
        sandbox_manager: SandboxManager,
        evaluator: SpecialistEvaluator,
    ) -> None:
        self._agent = lead_agent
        self._codegen = code_generator
        self._healer = error_analyzer
        self._model_manager = model_manager
        self._sandbox = sandbox_manager
        self._evaluator = evaluator

    async def run(self, mission_id: str) -> None:
        """Entry point: runs the full autonomous loop for a mission."""
        mission = await self._load_mission(mission_id)
        if not mission:
            logger.error("LoopStateMachine: mission %s not found", mission_id)
            return

        pivot_engine = PivotEngine(mission.target_metric)
        script_path: Optional[str] = None
        error_count = 0

        logger.info("LoopStateMachine: starting mission=%s", mission_id)

        while True:
            try:
                # ── PLANNING ──────────────────────────────────────────────
                await self._transition(mission_id, MissionStatus.PLANNING)
                plan = await self._agent.plan(
                    mission.goal, mission.task_type, mission.target_metric
                )
                await self._save_plan(mission_id, plan)

                # ── IMPLEMENTING ─────────────────────────────────────────
                await self._transition(mission_id, MissionStatus.RUNNING)
                script_path = await self._codegen.generate_training_script(mission_id, plan)

                # EXECUTE_CODE approval gate (supervised mode)
                if mission.autonomy_mode == "supervised":
                    approved = await self._request_approval(
                        mission_id, GateType.EXECUTE_CODE,
                        payload={"script_path": script_path},
                    )
                    if not approved:
                        logger.info("LoopStateMachine: EXECUTE_CODE gate rejected — aborting")
                        await self._transition(mission_id, MissionStatus.FAILED)
                        return

                # ── SANDBOXING ────────────────────────────────────────────
                self._model_manager.before_sandbox_launch(plan.get("sandbox_memory_gb", 8.0))
                pid, container_id = self._sandbox.launch(
                    mission_id, script_path,
                    env_vars={"ASTRA_MISSION_ID": mission_id},
                    memory_limit_gb=plan.get("sandbox_memory_gb", 8.0),
                )
                await self._save_sandbox_ids(mission_id, pid, container_id)

                # ── EXECUTING ────────────────────────────────────────────
                await self._transition(mission_id, MissionStatus.RUNNING)
                error_output = await self._wait_for_sandbox(mission_id)

                if error_output:
                    error_count += 1
                    if error_count > MAX_RETRIES:
                        logger.error("LoopStateMachine: max retries exceeded — failing mission")
                        await self._transition(mission_id, MissionStatus.FAILED)
                        return
                    logger.warning("LoopStateMachine: sandbox error (attempt %d/%d) — healing", error_count, MAX_RETRIES)
                    script_path = await self._healer.fix_script(script_path, error_output, error_count)
                    continue   # retry from sandboxing

                self._model_manager.after_sandbox_exit()
                error_count = 0

                # ── EVALUATING ────────────────────────────────────────────
                await self._transition(mission_id, MissionStatus.EVALUATING)
                eval_result = await self._evaluator.evaluate(mission_id, plan)
                current_metrics = eval_result.get("metrics", {})
                pivot_engine.record(mission.current_iteration or 0, current_metrics)
                await self._save_best_metric(mission_id, pivot_engine.best_metric_value())

                # Goal met → done
                if pivot_engine.is_goal_met(current_metrics):
                    logger.info("LoopStateMachine: goal met! mission=%s metrics=%s", mission_id, current_metrics)
                    await self._transition(mission_id, MissionStatus.COMPLETED)
                    return

                # ── REFINING ─────────────────────────────────────────────
                if pivot_engine.needs_pivot():
                    pivot = await self._agent.propose_pivot(current_metrics, pivot_engine.history_snapshot())
                    plan["hyperparameters"].update(pivot.get("adjustments", {}))
                    logger.info("LoopStateMachine: pivot applied: %s", pivot.get("reason"))

                self._agent.flush_iteration_context()
                await self._increment_iteration(mission_id)

            except Exception as e:
                logger.exception("LoopStateMachine: unhandled error in mission=%s: %s", mission_id, e)
                await self._transition(mission_id, MissionStatus.FAILED)
                return

    # ── DB helpers ─────────────────────────────────────────────────────────────

    async def _load_mission(self, mission_id: str) -> Optional[Mission]:
        async with AsyncSessionLocal() as session:
            return await session.get(Mission, mission_id)

    async def _transition(self, mission_id: str, status: MissionStatus) -> None:
        async with AsyncSessionLocal() as session:
            async with session.begin():
                await session.execute(
                    update(Mission)
                    .where(Mission.id == mission_id)
                    .values(status=status.value)
                )
        logger.info("LoopStateMachine: mission=%s → %s", mission_id, status.value)

    async def _save_plan(self, mission_id: str, plan: dict) -> None:
        async with AsyncSessionLocal() as session:
            async with session.begin():
                await session.execute(
                    update(Mission).where(Mission.id == mission_id).values(current_plan=plan)
                )

    async def _save_sandbox_ids(self, mission_id: str, pid: Optional[int], container_id: Optional[str]) -> None:
        async with AsyncSessionLocal() as session:
            async with session.begin():
                await session.execute(
                    update(Mission)
                    .where(Mission.id == mission_id)
                    .values(subprocess_pid=pid, container_id=container_id)
                )

    async def _save_best_metric(self, mission_id: str, value: Optional[float]) -> None:
        if value is None:
            return
        async with AsyncSessionLocal() as session:
            async with session.begin():
                await session.execute(
                    update(Mission)
                    .where(Mission.id == mission_id)
                    .values(best_metric_value=str(value))
                )

    async def _increment_iteration(self, mission_id: str) -> None:
        async with AsyncSessionLocal() as session:
            async with session.begin():
                mission = await session.get(Mission, mission_id)
                if mission:
                    await session.execute(
                        update(Mission)
                        .where(Mission.id == mission_id)
                        .values(current_iteration=(mission.current_iteration or 0) + 1)
                    )

    # ── Sandbox polling ────────────────────────────────────────────────────────

    async def _wait_for_sandbox(self, mission_id: str) -> Optional[str]:
        """Poll until the sandbox exits. Returns error output if it failed, else None."""
        log_path = self._sandbox.get_log_path(mission_id)
        while self._sandbox.is_alive(mission_id):
            await asyncio.sleep(EVAL_POLL_INTERVAL)

        # Read log to check for errors
        if os.path.isfile(log_path):
            with open(log_path, "r") as f:
                content = f.read()
            if "Traceback" in content or "Error" in content:
                return content
        return None

    # ── Approval gate ──────────────────────────────────────────────────────────

    async def _request_approval(
        self, mission_id: str, gate_type: GateType, payload: dict
    ) -> bool:
        """
        Create an approval gate record and poll until approved/rejected.
        In full_autonomy mode this is skipped (returns True immediately).
        In guided mode ALL gates require approval (not just EXECUTE_CODE).
        """
        async with AsyncSessionLocal() as session:
            gate = ApprovalGate(
                mission_id=mission_id,
                gate_type=gate_type.value,
                payload=payload,
                status=ApprovalStatus.PENDING.value,
            )
            session.add(gate)
            await session.commit()
            gate_id = gate.id

        logger.info("LoopStateMachine: waiting for %s approval (gate=%s)", gate_type.value, gate_id)

        # Poll for user decision
        while True:
            await asyncio.sleep(5)
            async with AsyncSessionLocal() as session:
                gate = await session.get(ApprovalGate, gate_id)
                if gate and gate.status == ApprovalStatus.APPROVED.value:
                    return True
                if gate and gate.status == ApprovalStatus.REJECTED.value:
                    return False
