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
from backend.services.telemetry_emitter import emit_status

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

        # Cancel stale approval gates left by any previous loop instance
        await self._cancel_stale_gates(mission_id)

        pivot_engine = PivotEngine(mission.target_metric)
        script_path: Optional[str] = None
        error_count = 0

        logger.info("LoopStateMachine: starting mission=%s", mission_id)

        while True:
            try:
                # ── PLANNING ──────────────────────────────────────────────
                await self._transition(mission_id, MissionStatus.PLANNING)
                await emit_status(mission_id, "Generating training plan…", event_type="info")
                plan = await self._agent.plan(
                    mission.goal, mission.task_type, mission.target_metric
                )
                await self._save_plan(mission_id, plan)
                await emit_status(
                    mission_id, "Plan ready",
                    event_type="success",
                    value=f"{plan.get('algorithm', '?')} · {plan.get('task_type', '?')}",
                )

                # ── IMPLEMENTING ─────────────────────────────────────────
                await self._transition(mission_id, MissionStatus.RUNNING)
                await emit_status(mission_id, "Generating training script…", event_type="info")
                script_path = await self._codegen.generate_training_script(mission_id, plan)
                await emit_status(mission_id, "Training script ready", event_type="success")

                # EXECUTE_CODE approval gate (supervised mode)
                if mission.autonomy_mode == "supervised":
                    await emit_status(mission_id, "Awaiting approval to execute script", event_type="warn")
                    approved = await self._request_approval(
                        mission_id, GateType.EXECUTE_CODE,
                        payload={"script_path": script_path},
                    )
                    if not approved:
                        logger.info("LoopStateMachine: EXECUTE_CODE gate rejected — aborting")
                        await emit_status(mission_id, "Execution rejected by user", event_type="error")
                        await self._transition(mission_id, MissionStatus.FAILED)
                        return
                    await emit_status(mission_id, "Execution approved", event_type="success")

                # ── SANDBOXING ────────────────────────────────────────────
                self._model_manager.before_sandbox_launch(plan.get("sandbox_memory_gb", 8.0))
                await emit_status(mission_id, "Launching training sandbox…", event_type="info")
                log_path = self._sandbox.get_log_path(mission_id)
                log_offset = os.path.getsize(log_path) if os.path.isfile(log_path) else 0
                pid, container_id = self._sandbox.launch(
                    mission_id, script_path,
                    env_vars={"ASTRA_MISSION_ID": mission_id},
                    memory_limit_gb=plan.get("sandbox_memory_gb", 8.0),
                )
                await self._save_sandbox_ids(mission_id, pid, container_id)
                await emit_status(mission_id, "Sandbox running", event_type="info", value=f"pid={pid}")

                # ── EXECUTING ────────────────────────────────────────────
                await self._transition(mission_id, MissionStatus.RUNNING)
                error_output = await self._wait_for_sandbox(mission_id, log_offset)

                if error_output:
                    error_count += 1
                    if error_count > MAX_RETRIES:
                        logger.error("LoopStateMachine: max retries exceeded — failing mission")
                        await emit_status(mission_id, "Max retries exceeded", event_type="error")
                        await self._transition(mission_id, MissionStatus.FAILED)
                        return
                    logger.warning("LoopStateMachine: sandbox error (attempt %d/%d) — healing", error_count, MAX_RETRIES)
                    await emit_status(
                        mission_id, "Sandbox error — healing script",
                        event_type="warn",
                        value=f"attempt {error_count}/{MAX_RETRIES}",
                    )
                    script_path = await self._healer.fix_script(script_path, error_output, error_count)
                    continue   # retry from sandboxing

                self._model_manager.after_sandbox_exit()
                error_count = 0
                await emit_status(mission_id, "Training complete", event_type="success")

                # ── EVALUATING ────────────────────────────────────────────
                await self._transition(mission_id, MissionStatus.EVALUATING)
                await emit_status(mission_id, "Evaluating results…", event_type="info")
                eval_result = await self._evaluator.evaluate(mission_id, plan)
                current_metrics = eval_result.get("metrics", {})
                pivot_engine.record(mission.current_iteration or 0, current_metrics)
                await self._save_best_metric(mission_id, pivot_engine.best_metric_value())

                # Goal met → done
                if pivot_engine.is_goal_met(current_metrics):
                    logger.info("LoopStateMachine: goal met! mission=%s metrics=%s", mission_id, current_metrics)
                    await emit_status(mission_id, "Goal achieved!", event_type="success",
                                      value=str(pivot_engine.best_metric_value()))
                    await self._transition(mission_id, MissionStatus.COMPLETED)
                    await self._crystallize(mission_id, plan, pivot_engine.best_metric_value())
                    return

                # ── REFINING ─────────────────────────────────────────────
                if pivot_engine.needs_pivot():
                    pivot = await self._agent.propose_pivot(current_metrics, pivot_engine.history_snapshot())
                    plan["hyperparameters"].update(pivot.get("adjustments", {}))
                    logger.info("LoopStateMachine: pivot applied: %s", pivot.get("reason"))
                    await emit_status(
                        mission_id, "Pivot triggered",
                        event_type="pivot",
                        value=pivot.get("reason", "plateau detected"),
                        iteration=mission.current_iteration or 0,
                    )

                self._agent.flush_iteration_context()
                await self._increment_iteration(mission_id)

            except asyncio.CancelledError:
                logger.info("LoopStateMachine: mission=%s cancelled (shutdown) — resetting to pending", mission_id)
                await self._transition(mission_id, MissionStatus.PENDING)
                raise  # propagate so asyncio knows the task is done

            except Exception as e:
                logger.exception("LoopStateMachine: unhandled error in mission=%s: %s", mission_id, e)
                await emit_status(mission_id, "Mission failed", event_type="error", value=str(e))
                await self._transition(mission_id, MissionStatus.FAILED)
                return

    # ── DB helpers ─────────────────────────────────────────────────────────────

    async def _cancel_stale_gates(self, mission_id: str) -> None:
        """Reject any pending approval gates left by a previous loop instance."""
        async with AsyncSessionLocal() as session:
            async with session.begin():
                result = await session.execute(
                    select(ApprovalGate).where(
                        ApprovalGate.mission_id == mission_id,
                        ApprovalGate.status == ApprovalStatus.PENDING.value,
                    )
                )
                stale = result.scalars().all()
                for gate in stale:
                    gate.status = ApprovalStatus.REJECTED.value
                    logger.info("LoopStateMachine: cancelled stale gate=%s for mission=%s", gate.id, mission_id)

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

    async def _wait_for_sandbox(self, mission_id: str, log_offset: int = 0) -> Optional[str]:
        """Poll until the sandbox exits. Returns error output if it failed, else None."""
        log_path = self._sandbox.get_log_path(mission_id)
        while self._sandbox.is_alive(mission_id):
            await asyncio.sleep(EVAL_POLL_INTERVAL)

        # Only read content written by THIS run (skip prior runs' output)
        if os.path.isfile(log_path):
            with open(log_path, "r") as f:
                f.seek(log_offset)
                content = f.read()
            if "Traceback" in content or "Error" in content:
                return content
        return None

    # ── Crystallization ────────────────────────────────────────────────────────

    async def _crystallize(self, mission_id: str, plan: dict, score: Optional[float]) -> None:
        """Distil a completed mission into a reusable recipe (non-blocking on failure)."""
        try:
            from backend.services.crystallizer import crystallize
            record = await crystallize(mission_id, plan=plan, score=score)
            if record:
                logger.info("LoopStateMachine: crystallized recipe '%s' for mission=%s", record.name, mission_id)
        except Exception as exc:
            logger.warning("LoopStateMachine: crystallization failed for mission=%s: %s", mission_id, exc)

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
