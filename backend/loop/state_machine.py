"""
LoopStateMachine — Step 3.3.

Implements the Plan → Implement → Sandbox → Execute → Eval → Refine loop.
Each state transition is persisted atomically in the Mission Store.
Respects autonomy mode (guided/supervised/full_autonomy) for approval gates.
"""
from __future__ import annotations

import asyncio
import json
import os
import shutil
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
from backend.evaluator.manifest_evaluator import ManifestEvaluator
from backend.loop.pivots import PivotEngine
from backend.models.manifest import RequirementManifest
from backend.agent.critic_agent import CriticAgent, MAX_REVISIONS as CRITIC_MAX_REVISIONS
from backend.services.manifest_generator import generate_manifest
from backend.services.preflight import PreflightChecker
from backend.services import mission_state, session_summary
from backend.config import settings
from backend.logging_config import get_logger
from backend.services.telemetry_emitter import emit_status, emit_critique

logger = get_logger(__name__)

MAX_RETRIES = 3          # max error-fix iterations before marking FAILED
EVAL_POLL_INTERVAL = 10  # seconds between sandbox liveness checks
ITER_CHECKPOINT_WINDOW = 10  # per-iteration checkpoints to keep (rolling)


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
        critic: Optional[CriticAgent] = None,
    ) -> None:
        self._agent = lead_agent
        self._codegen = code_generator
        self._healer = error_analyzer
        self._model_manager = model_manager
        self._sandbox = sandbox_manager
        self._evaluator = evaluator
        self._critic = critic
        self._manifest_evaluator = ManifestEvaluator()
        self._preflight = PreflightChecker()

    async def run(self, mission_id: str) -> None:
        """Entry point: runs the full autonomous loop for a mission."""
        mission = await self._load_mission(mission_id)
        if not mission:
            logger.error("LoopStateMachine: mission %s not found", mission_id)
            return

        # Cancel stale approval gates left by any previous loop instance
        await self._cancel_stale_gates(mission_id)

        # ── PRE-FLIGHT (Step 7.4) ─────────────────────────────────────────────
        preflight = self._preflight.run(mission_id, mission.task_type)
        await emit_status(
            mission_id, "Pre-flight checks",
            event_type="success" if preflight.passed else "warn",
            value=preflight.summary(),
        )

        pivot_engine = PivotEngine(mission.target_metric)
        # Seed pivot engine with the persisted best so restarts don't lose history
        persisted_best = self._load_persisted_best(mission_id, mission)
        if persisted_best is not None:
            seed_iter = mission.best_metric_iteration if mission.best_metric_iteration is not None else -1
            pivot_engine.record(seed_iter, {next(iter(mission.target_metric), "metric"): persisted_best})
            logger.info(
                "LoopStateMachine: seeded pivot engine with persisted best=%.2f at iter=%d",
                persisted_best, seed_iter,
            )
            # Sync DB if best_score.txt is higher than DB value
            db_best = float(mission.best_metric_value) if mission.best_metric_value else None
            if db_best is None or persisted_best > db_best:
                await self._save_best_metric(mission_id, persisted_best)
        # Restore escalation count so restarts don't reset aggressive pivoting
        if mission.pivot_escalation_count:
            pivot_engine.restore_pivot_count(mission.pivot_escalation_count)
            logger.info(
                "LoopStateMachine: restored pivot_count=%d (escalation=%d) for mission=%s",
                mission.pivot_escalation_count, pivot_engine.escalation_level(), mission_id,
            )
        # Restore _best_at_last_pivot so the first post-restart record_pivot() call
        # doesn't see None and incorrectly reset pivot_count to 0.
        if persisted_best is not None:
            pivot_engine.restore_best_at_last_pivot(persisted_best)
        # Restore best_policy_kwargs so Level 1 pivots after restart still prefer
        # the proven architecture. Use `is not None` — {} (default arch sentinel) is falsy.
        if mission.best_policy_kwargs is not None:
            pivot_engine.restore_best_policy_kwargs(mission.best_policy_kwargs)
            logger.info(
                "LoopStateMachine: restored best_policy_kwargs=%s for mission=%s",
                mission.best_policy_kwargs, mission_id,
            )
        # Replay per-iteration goal metric history from telemetry so needs_pivot()
        # has full context immediately rather than waiting for PLATEAU_WINDOW fresh iters.
        metric_name_for_history = next(iter(mission.target_metric), None)
        if metric_name_for_history:
            history_entries = self._load_goal_metric_history(mission_id, metric_name_for_history)
            if history_entries:
                pivot_engine.restore_history(history_entries)
                logger.info(
                    "LoopStateMachine: replayed %d goal metric history entries for mission=%s",
                    len(history_entries), mission_id,
                )

        manifest = self._load_or_create_manifest(mission_id, mission)
        mission_dir = os.path.abspath(os.path.join(settings.data_path, "missions", mission_id))
        script_path: Optional[str] = None
        error_count = 0
        pivot_reason: Optional[str] = None
        current_iteration = mission.current_iteration or 0
        plan: Optional[dict] = None
        # skip_replan_from_db: set on startup when restarting mid-pivot — load saved plan from DB.
        # skip_replan_in_memory: set after every in-loop pivot — plan already updated in memory.
        # Only one of these can be True at any time.
        skip_replan_from_db = current_iteration > 0 and mission.current_plan is not None
        skip_replan_in_memory = False
        if skip_replan_from_db:
            logger.info(
                "LoopStateMachine: resuming from saved plan at iter=%d for mission=%s",
                current_iteration, mission_id,
            )

        logger.info("LoopStateMachine: starting mission=%s manifest=%d reqs", mission_id, len(manifest.requirements))

        while True:
            try:
                # ── PLANNING ──────────────────────────────────────────────
                await self._transition(mission_id, MissionStatus.PLANNING)
                did_replan = False
                if skip_replan_in_memory:
                    # Pivot fired last iteration — plan already has the updated values in memory.
                    skip_replan_in_memory = False
                    await emit_status(
                        mission_id, "Continuing with pivoted plan",
                        event_type="info",
                        value=f"{plan.get('algorithm', '?')} · {plan.get('task_type', '?')}",  # type: ignore[union-attr]
                    )
                elif skip_replan_from_db:
                    # Restart with saved plan — reload persisted plan from DB.
                    async with AsyncSessionLocal() as _s:
                        _m = await _s.get(Mission, mission_id)
                        plan = dict(_m.current_plan) if _m and _m.current_plan else {}
                    skip_replan_from_db = False
                    await emit_status(
                        mission_id, "Continuing with pivoted plan",
                        event_type="info",
                        value=f"{plan.get('algorithm', '?')} · {plan.get('task_type', '?')}",
                    )
                else:
                    await emit_status(mission_id, "Generating training plan…", event_type="info")
                    plan = await self._agent.plan(
                        mission.goal, mission.task_type, mission.target_metric
                    )
                    await self._save_plan(mission_id, plan)
                    did_replan = True
                    await emit_status(
                        mission_id, "Plan ready",
                        event_type="success",
                        value=f"{plan.get('algorithm', '?')} · {plan.get('task_type', '?')}",
                    )

                # Reconcile manifest artifact pattern if plan task_type differs
                # from the mission's stored task_type (e.g. user left dropdown on "rl")
                if current_iteration == 0:
                    plan_task_type = plan.get("task_type", "").lower()
                    if plan_task_type and plan_task_type != mission.task_type:
                        manifest = generate_manifest(
                            mission_id=mission_id,
                            goal=mission.goal,
                            task_type=plan_task_type,
                            target_metric=mission.target_metric or {},
                        )
                        self._save_manifest(mission_id, manifest)
                        logger.info(
                            "LoopStateMachine: manifest reconciled task_type %s→%s for mission=%s",
                            mission.task_type, plan_task_type, mission_id,
                        )

                # ── CRITIC REVIEW (Step 7.1) ──────────────────────────────
                # Only run on genuine replans (iter 0, or after algo switch that
                # generates a wholly new plan). Skip on resume/pivot — the plan was
                # already reviewed and the change is targeted, not a full replan.
                if self._critic is not None and did_replan:
                    critique = await self._critic.review(plan, mission.goal, revision=0)
                    await emit_critique(mission_id, critique.to_dict())
                    for rev in range(1, CRITIC_MAX_REVISIONS + 1):
                        if critique.approved:
                            break
                        await emit_status(
                            mission_id, "Critic requesting revision",
                            event_type="warn",
                            value=f"score={critique.overall_score:.1f} revision {rev}/{CRITIC_MAX_REVISIONS}",
                        )
                        plan = await self._agent.revise_plan(plan, critique.feedback)
                        await self._save_plan(mission_id, plan)
                        critique = await self._critic.review(plan, mission.goal, revision=rev)
                        await emit_critique(mission_id, critique.to_dict())
                    status = "approved" if critique.approved else "proceeding despite low score"
                    await emit_status(
                        mission_id, f"Critic: {status}",
                        event_type="success" if critique.approved else "warn",
                        value=f"score={critique.overall_score:.1f}",
                    )

                # ── IMPLEMENTING ─────────────────────────────────────────
                await self._transition(mission_id, MissionStatus.RUNNING)
                await emit_status(mission_id, "Generating training script…", event_type="info")
                # _PLAN_SCHEMA doesn't include target_metric, so the LLM never puts it in the plan.
                # Inject it from the mission before code generation so the callback template
                # gets the correct target_metric_name (e.g. "lines_cleared", not "mean_reward").
                plan["target_metric"] = mission.target_metric or {}
                script_path = await self._codegen.generate_training_script(mission_id, plan, current_iteration)
                await emit_status(mission_id, "Training script ready", event_type="success")
                error_history: list[str] = []   # accumulated errors for this script

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
                tel_path = os.path.join(settings.data_path, "missions", mission_id, "telemetry.jsonl")
                tel_offset = os.path.getsize(tel_path) if os.path.isfile(tel_path) else 0
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
                    error_history.append(error_output)
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
                    script_path = await self._healer.fix_script(
                        script_path, error_output, error_count,
                        prior_errors=error_history[:-1],
                        mission_id=mission_id,
                        domain=plan.get("domain"),
                    )
                    continue   # retry from sandboxing

                self._model_manager.after_sandbox_exit()
                error_count = 0
                await emit_status(mission_id, "Training complete", event_type="success")

                # ── EVALUATING ────────────────────────────────────────────
                await self._transition(mission_id, MissionStatus.EVALUATING)
                await emit_status(mission_id, "Evaluating results…", event_type="info")
                eval_result = await self._evaluator.evaluate(mission_id, plan)
                current_metrics = eval_result.get("metrics", {})
                # Merge in metrics the sandbox actually posted to telemetry (mean_reward)
                sandbox_metrics = self._read_telemetry_metrics(mission_id, tel_offset)
                current_metrics = {**current_metrics, **sandbox_metrics}

                # For non-mean_reward goal metrics: run dedicated eval episodes if the
                # benchmark didn't supply the value, then always write to telemetry so the
                # MetricGap sparkline receives it.
                metric_name = next(iter(mission.target_metric), None)
                if metric_name and metric_name != "mean_reward":
                    if metric_name not in current_metrics:
                        await emit_status(
                            mission_id, f"Evaluating {metric_name}…", event_type="info"
                        )
                        goal_val = await asyncio.to_thread(
                            self._run_goal_metric_eval, mission_id, plan, metric_name
                        )
                        if goal_val is not None:
                            current_metrics[metric_name] = goal_val
                    goal_val = current_metrics.get(metric_name)
                    if goal_val is not None:
                        await self._append_telemetry_metric(
                            mission_id, metric_name, goal_val, current_iteration
                        )
                        logger.info(
                            "LoopStateMachine: goal metric mission=%s %s=%.3f iter=%d",
                            mission_id, metric_name, goal_val, current_iteration,
                        )

                _current_policy_kwargs = plan.get("hyperparameters", {}).get("policy_kwargs")
                pivot_engine.record(current_iteration, current_metrics, policy_kwargs=_current_policy_kwargs)
                await self._save_best_policy_kwargs(mission_id, pivot_engine.best_policy_kwargs())
                current_val = current_metrics.get(metric_name) if metric_name else None
                await self._save_best_metric(
                    mission_id,
                    pivot_engine.best_metric_value(),
                    best_iteration=pivot_engine.best_metric_iteration(),
                )
                await self._save_current_metric(mission_id, current_val)
                self._save_iteration_checkpoint(mission_id, current_iteration)

                # ── POST-PIVOT REGRESSION CHECK ───────────────────────────
                # If an arch/algo pivot made things materially worse after
                # PLATEAU_WINDOW iters, restore the pre-pivot checkpoint and
                # de-escalate so HP tuning resumes from the good baseline.
                _pivot_reverted = False
                if pivot_engine.should_revert_pivot():
                    _checkpoint_dir = os.path.join(
                        settings.data_path, "missions", mission_id, "checkpoints"
                    )
                    _best_zip = os.path.join(_checkpoint_dir, "best_model.zip")
                    _pre_hps = plan.pop("_pre_pivot_hps", None)
                    _pre_score = plan.pop("_pre_pivot_best_score", None)
                    _best_iter = pivot_engine.best_metric_iteration()
                    _iter_ckpt = (
                        os.path.join(_checkpoint_dir, "iter", f"checkpoint_iter_{_best_iter}.zip")
                        if _best_iter is not None else None
                    )
                    _restore_src = _iter_ckpt if _iter_ckpt and os.path.exists(_iter_ckpt) else None
                    if _restore_src:
                        try:
                            shutil.copy2(_restore_src, _best_zip)
                            logger.info(
                                "LoopStateMachine: restored checkpoint from %s for mission=%s",
                                os.path.basename(_restore_src), mission_id,
                            )
                        except Exception as _e:
                            logger.warning("LoopStateMachine: could not restore checkpoint: %s", _e)
                    if _pre_score is not None:
                        try:
                            with open(os.path.join(_checkpoint_dir, "best_score.txt"), "w") as _f:
                                _f.write(str(_pre_score))
                        except Exception as _e:
                            logger.warning("LoopStateMachine: could not restore best_score.txt: %s", _e)
                    if _pre_hps is not None:
                        plan["hyperparameters"] = _pre_hps
                    pivot_engine.revert_escalation()
                    await self._save_pivot_count(mission_id, pivot_engine.pivot_count)
                    await self._save_plan(mission_id, plan)
                    skip_replan_in_memory = False
                    _revert_label = (
                        f"iter {_best_iter}" if _best_iter is not None else "pre-pivot backup"
                    )
                    await emit_status(
                        mission_id,
                        f"Pivot reverted — restored checkpoint from {_revert_label}, resuming HP tuning",
                        event_type="warn",
                        iteration=current_iteration,
                    )
                    _pivot_reverted = True

                # ── MANIFEST CHECK ────────────────────────────────────────
                manifest = self._manifest_evaluator.evaluate(
                    manifest, current_metrics, mission_dir, sandbox_ok=True,
                )
                self._save_manifest(mission_id, manifest)
                summary = manifest.summary()
                await emit_status(
                    mission_id, "Requirements checked",
                    event_type="info",
                    value=f"{summary['passed']}/{summary['total']} passed",
                )

                # All requirements met → done
                if manifest.is_complete():
                    best = pivot_engine.best_metric_value()
                    logger.info("LoopStateMachine: manifest complete! mission=%s metrics=%s", mission_id, current_metrics)
                    await emit_status(mission_id, "Goal achieved!", event_type="success",
                                      value=str(best))
                    await self._transition(mission_id, MissionStatus.COMPLETED)
                    await self._crystallize(mission_id, plan, best)
                    return

                # ── MISSION STATE (Step 7.5) ──────────────────────────────
                mission_state.update(
                    mission_id,
                    iteration=current_iteration,
                    plan=plan,
                    metrics=current_metrics,
                )

                # ── REFINING ─────────────────────────────────────────────
                pivot_reason = None
                if not _pivot_reverted and pivot_engine.needs_pivot():
                    escalation = pivot_engine.escalation_level()
                    current_algo = plan.get("algorithm", "PPO")
                    # Detect if the user's goal explicitly names an algorithm.
                    # If so, never switch algorithms — remap level 2 to reward shaping.
                    algo_locked = self._is_algorithm_locked(mission.goal, current_algo)
                    pivot = await self._agent.propose_pivot(
                        current_metrics,
                        pivot_engine.history_snapshot(),
                        escalation_level=escalation,
                        current_algorithm=current_algo,
                        algorithm_locked=algo_locked,
                        current_policy_kwargs=plan.get("hyperparameters", {}).get("policy_kwargs"),
                        current_hyperparameters={
                            k: v for k, v in plan.get("hyperparameters", {}).items()
                            if k != "policy_kwargs"
                        } or None,
                        current_env_kwargs=plan.get("env_kwargs") or None,
                        best_policy_kwargs=pivot_engine.best_policy_kwargs(),
                        best_metric_value=pivot_engine.best_metric_value(),
                        best_metric_iteration=pivot_engine.best_metric_iteration(),
                    )
                    pivot_engine.record_pivot()
                    await self._save_pivot_count(mission_id, pivot_engine.pivot_count)
                    pivot = self._normalize_pivot(pivot)
                    adjustments = self._clamp_rl_adjustments(
                        pivot.get("adjustments", {}), plan.get("task_type", "rl")
                    )

                    # Filter out HP adjustments identical to current values.
                    # Compare as float to handle LLM returning "0.0005" (str) vs 0.0005 (float).
                    def _hp_changed(k: str, proposed) -> bool:
                        current = plan["hyperparameters"].get(k)
                        if current is None:
                            return True
                        try:
                            return float(current) != float(proposed)
                        except (TypeError, ValueError):
                            return current != proposed

                    real_adjustments = {
                        k: v for k, v in adjustments.items()
                        if _hp_changed(k, v)
                    }
                    # Never switch algorithms when the user explicitly named one in the goal.
                    proposed_algo = pivot.get("algorithm")
                    algo_changed = bool(
                        proposed_algo
                        and proposed_algo != current_algo
                        and not algo_locked
                    )
                    if algo_locked and proposed_algo and proposed_algo != current_algo:
                        logger.info(
                            "LoopStateMachine: ignoring algo switch %s→%s — algorithm locked by goal",
                            current_algo, proposed_algo,
                        )
                    _proposed_pky = pivot.get("policy_kwargs")
                    _current_pky = plan.get("hyperparameters", {}).get("policy_kwargs")
                    _recent_arches = plan.get("recent_arches", [])
                    _arch_oscillation = bool(_proposed_pky and _proposed_pky in _recent_arches)
                    if _arch_oscillation:
                        logger.warning(
                            "LoopStateMachine: arch oscillation detected — proposed %s already in recent history %s; suppressing",
                            _proposed_pky, _recent_arches,
                        )
                    arch_changed = bool(
                        _proposed_pky and
                        _proposed_pky != _current_pky and
                        not _arch_oscillation
                    )
                    env_kwargs_changed = bool(
                        pivot.get("env_kwargs") and pivot["env_kwargs"] != plan.get("env_kwargs", {})
                    )

                    # No-op pivot: nothing actually changed — force escalation and skip
                    if not real_adjustments and not algo_changed and not arch_changed and not env_kwargs_changed:
                        logger.warning(
                            "LoopStateMachine: no-op pivot detected (all proposed values identical to current) — "
                            "escalating pivot count without applying; escalation=%d", escalation,
                        )
                        pivot_engine.record_pivot()  # double-count to escalate faster
                        await self._save_pivot_count(mission_id, pivot_engine.pivot_count)
                        await emit_status(
                            mission_id, "Pivot skipped — no changes proposed",
                            event_type="warn",
                            value=f"escalation now {pivot_engine.escalation_level()}",
                            iteration=current_iteration,
                        )
                        pivot_reason = None  # don't regenerate code
                    else:
                        # Snapshot before mutating so display shows old→new correctly
                        old_hps = {k: plan["hyperparameters"].get(k) for k in real_adjustments}
                        # Before any arch/algo change: arm regression detector
                        if arch_changed or algo_changed:
                            plan["_pre_pivot_hps"] = dict(plan.get("hyperparameters", {}))
                            plan["_pre_pivot_best_score"] = pivot_engine.best_metric_value()
                            pivot_engine.record_arch_pivot_baseline()
                        plan["hyperparameters"].update(real_adjustments)
                        if arch_changed:
                            # Track the outgoing arch so future pivots back to it are suppressed
                            _recent = list(plan.get("recent_arches", []))
                            if _current_pky is not None and _current_pky not in _recent:
                                _recent.append(_current_pky)
                            plan["recent_arches"] = _recent[-5:]
                            plan["hyperparameters"]["policy_kwargs"] = pivot["policy_kwargs"]
                            # Reset best_score.txt so the new architecture can save its own
                            # checkpoint. Without this, the old peak score blocks best_model.zip
                            # from ever being written by the new architecture.
                            best_score_path = os.path.join(
                                settings.data_path, "missions", mission_id,
                                "checkpoints", "best_score.txt",
                            )
                            try:
                                with open(best_score_path, "w") as _f:
                                    _f.write("-inf")
                                logger.info(
                                    "LoopStateMachine: reset best_score.txt after net_arch pivot for mission=%s",
                                    mission_id,
                                )
                            except Exception as _e:
                                logger.warning("LoopStateMachine: could not reset best_score.txt: %s", _e)
                        if algo_changed:
                            logger.info(
                                "LoopStateMachine: algorithm switch %s → %s",
                                current_algo, pivot["algorithm"],
                            )
                            plan["algorithm"] = pivot["algorithm"]
                            plan["hyperparameters"] = pivot.get("adjustments", {})
                            # Reset best_score so the new algorithm can save its own checkpoint
                            best_score_path = os.path.join(
                                settings.data_path, "missions", mission_id,
                                "checkpoints", "best_score.txt",
                            )
                            try:
                                with open(best_score_path, "w") as _f:
                                    _f.write("-inf")
                                logger.info(
                                    "LoopStateMachine: reset best_score.txt after algo switch for mission=%s",
                                    mission_id,
                                )
                            except Exception as _e:
                                logger.warning("LoopStateMachine: could not reset best_score.txt: %s", _e)
                        if env_kwargs_changed:
                            _cur_env = plan.get("env_kwargs") or {}
                            _merged = dict(_cur_env, **pivot["env_kwargs"])
                            plan["env_kwargs"] = self._clamp_env_kwargs(_merged)
                            logger.info(
                                "LoopStateMachine: reward reshape applied: %s",
                                plan["env_kwargs"],
                            )
                        pivot_reason = pivot.get("reason", "plateau detected")
                        # Persist the pivot-modified plan so a restart resumes with
                        # the new HPs/algo/env_kwargs rather than re-planning fresh.
                        await self._save_plan(mission_id, plan)
                        skip_replan_in_memory = True
                        # Build a compact changes summary for the event stream
                        change_parts = []
                        if algo_changed:
                            change_parts.append(f"algo: {current_algo}→{pivot['algorithm']}")
                        if real_adjustments:
                            hp_strs = []
                            for k, v in real_adjustments.items():
                                old_v = old_hps.get(k)
                                hp_strs.append(f"{k}: {old_v}→{v}" if old_v is not None else f"{k}={v}")
                            change_parts.append(", ".join(hp_strs))
                        if arch_changed:
                            arch = pivot["policy_kwargs"].get("net_arch")
                            if arch:
                                change_parts.append(f"net_arch: {arch}")
                        if env_kwargs_changed:
                            env_strs = [f"{k}={v}" for k, v in pivot["env_kwargs"].items()]
                            change_parts.append(f"env_kwargs: {{{', '.join(env_strs)}}}")
                        changes_summary = " | ".join(change_parts) if change_parts else "hyperparameter adjustment"
                        pivot_value = f"{pivot_reason} | changes: {changes_summary}"
                        logger.info(
                            "LoopStateMachine: pivot applied (escalation=%d): %s | algo=%s | adjustments: %s | policy_kwargs: %s",
                            escalation, pivot_reason, plan.get("algorithm"), real_adjustments, pivot.get("policy_kwargs"),
                        )
                        await emit_status(
                            mission_id, "Pivot triggered",
                            event_type="pivot",
                            value=pivot_value,
                            iteration=current_iteration,
                        )

                # Preserve current plan for next iteration — only re-plan when a
                # pivot fires (skip_replan_in_memory already True in that path)
                # or on the very first iteration (plan freshly generated above).
                if not skip_replan_in_memory:
                    skip_replan_in_memory = True

                # Always persist the current plan so a service restart loads the
                # most recent state (including env_kwargs from prior pivots).
                # Pivot iterations already call _save_plan above; this covers the
                # non-pivot case where env_kwargs / HPs haven't changed.
                if plan is not None:
                    await self._save_plan(mission_id, plan)

                # ── SESSION SUMMARY (Step 7.3) ────────────────────────────
                session_summary.write_session_summary(
                    mission_id=mission_id,
                    iteration=current_iteration,
                    goal=mission.goal,
                    algorithm=plan.get("algorithm", "unknown"),
                    current_metrics=current_metrics,
                    manifest_summary=manifest.summary(),
                    pivot_applied=pivot_reason,
                )

                self._agent.flush_iteration_context()
                await self._increment_iteration(mission_id)
                current_iteration += 1

            except asyncio.CancelledError:
                logger.info("LoopStateMachine: mission=%s cancelled (shutdown) — terminating sandbox and resetting to pending", mission_id)
                try:
                    self._sandbox.terminate(mission_id)
                except Exception as _term_e:
                    logger.warning("LoopStateMachine: sandbox terminate on cancel failed for mission=%s: %s", mission_id, _term_e)
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

    def _load_persisted_best(self, mission_id: str, mission) -> Optional[float]:
        """Return the highest known best metric from all available sources."""
        candidates = []
        metric_name = next(iter(mission.target_metric), None) if mission.target_metric else None
        # best_score.txt is written by the training callback and always stores mean_reward.
        # Only use it when the target metric IS mean_reward; otherwise it would corrupt
        # custom targets like lines_cleared with a negative reward value.
        if metric_name in (None, "mean_reward"):
            score_file = os.path.join(
                settings.data_path, "missions", mission_id, "checkpoints", "best_score.txt"
            )
            try:
                candidates.append(float(open(score_file).read().strip()))
            except Exception:
                pass
        # From DB — for custom (non-mean_reward) targets, negative values indicate
        # contamination from a prior mean_reward seed; discard them.
        try:
            if mission.best_metric_value:
                db_val = float(mission.best_metric_value)
                if metric_name == "mean_reward" or db_val >= 0:
                    candidates.append(db_val)
        except Exception:
            pass
        # From full telemetry scan — authoritative for the actual target metric key
        if metric_name:
            all_telem = self._read_telemetry_metrics(mission_id, offset=0)
            if metric_name in all_telem:
                candidates.append(all_telem[metric_name])
        return max(candidates) if candidates else None

    def _load_goal_metric_history(self, mission_id: str, metric_name: str) -> list[dict]:
        """Read per-iteration goal metric values from telemetry.jsonl for history replay."""
        tel_path = os.path.join(settings.data_path, "missions", mission_id, "telemetry.jsonl")
        if not os.path.isfile(tel_path):
            return []
        entries: dict[int, float] = {}
        try:
            with open(tel_path) as f:
                for line in f:
                    try:
                        e = json.loads(line)
                    except Exception:
                        continue
                    if e.get("type") == "metric" and e.get("name") == metric_name:
                        it = e.get("iteration")
                        val = e.get("value")
                        if it is not None and val is not None:
                            # Keep the value from the last entry per iteration
                            entries[int(it)] = float(val)
        except Exception:
            return []
        return [{"iteration": it, metric_name: val} for it, val in sorted(entries.items())]

    def _save_iteration_checkpoint(self, mission_id: str, iteration: int) -> None:
        """Copy best_model.zip → iter/checkpoint_iter_{N}.zip and prune old ones."""
        import glob
        checkpoint_dir = os.path.join(settings.data_path, "missions", mission_id, "checkpoints")
        best_zip = os.path.join(checkpoint_dir, "best_model.zip")
        if not os.path.exists(best_zip):
            return
        iter_dir = os.path.join(checkpoint_dir, "iter")
        os.makedirs(iter_dir, exist_ok=True)
        dest = os.path.join(iter_dir, f"checkpoint_iter_{iteration}.zip")
        try:
            shutil.copy2(best_zip, dest)
            logger.info(
                "LoopStateMachine: saved checkpoint_iter_%d for mission=%s", iteration, mission_id
            )
        except Exception as _e:
            logger.warning("LoopStateMachine: could not save iter checkpoint: %s", _e)
            return
        # Prune checkpoints beyond the rolling window
        try:
            all_iter = sorted(
                glob.glob(os.path.join(iter_dir, "checkpoint_iter_*.zip")),
                key=lambda p: int(os.path.basename(p).replace("checkpoint_iter_", "").replace(".zip", "")),
            )
            for old in all_iter[:-ITER_CHECKPOINT_WINDOW]:
                os.remove(old)
                logger.info("LoopStateMachine: pruned %s", os.path.basename(old))
        except Exception as _e:
            logger.warning("LoopStateMachine: could not prune iter checkpoints: %s", _e)

    async def _save_best_metric(
        self,
        mission_id: str,
        value: Optional[float],
        best_iteration: Optional[int] = None,
    ) -> None:
        if value is None:
            return
        updates: dict = {"best_metric_value": str(value)}
        if best_iteration is not None:
            updates["best_metric_iteration"] = best_iteration
        async with AsyncSessionLocal() as session:
            async with session.begin():
                await session.execute(
                    update(Mission).where(Mission.id == mission_id).values(**updates)
                )

    async def _save_current_metric(self, mission_id: str, value: Optional[float]) -> None:
        if value is None:
            return
        async with AsyncSessionLocal() as session:
            async with session.begin():
                await session.execute(
                    update(Mission)
                    .where(Mission.id == mission_id)
                    .values(current_metric_value=str(value))
                )

    async def _save_pivot_count(self, mission_id: str, count: int) -> None:
        async with AsyncSessionLocal() as session:
            async with session.begin():
                await session.execute(
                    update(Mission)
                    .where(Mission.id == mission_id)
                    .values(pivot_escalation_count=count)
                )

    async def _save_best_policy_kwargs(self, mission_id: str, kwargs: Optional[dict]) -> None:
        if kwargs is None:
            return
        async with AsyncSessionLocal() as session:
            async with session.begin():
                await session.execute(
                    update(Mission)
                    .where(Mission.id == mission_id)
                    .values(best_policy_kwargs=kwargs)
                )

    @staticmethod
    def _is_algorithm_locked(goal: str, current_algorithm: str) -> bool:
        """Return True if the goal explicitly names the current algorithm.

        When a user writes "Train a Snake-v0 DQN agent …", switching to PPO
        would violate their intent. We detect this by checking whether the
        algorithm name appears as a word in the goal string (case-insensitive).
        """
        import re
        return bool(re.search(rf"\b{re.escape(current_algorithm)}\b", goal, re.IGNORECASE))

    @staticmethod
    def _clamp_env_kwargs(env_kwargs: dict) -> dict:
        """Clamp LLM-proposed env_kwargs to sane ranges.

        distance_weight=0 removes the navigation shaping signal entirely and
        causes the agent to get stuck after eating the first food item.
        """
        out = dict(env_kwargs)
        if "distance_weight" in out:
            out["distance_weight"] = max(0.1, float(out["distance_weight"]))
        return out

    @staticmethod
    def _normalize_pivot(pivot: dict) -> dict:
        """Fix common LLM schema deviations in pivot responses.

        The LLM sometimes nests HP adjustments under adjustments.hyperparameters
        and env_kwargs under adjustments.env_kwargs instead of as flat scalars
        in adjustments and a top-level env_kwargs key. Flatten those here so
        the rest of the pipeline always sees a consistent structure.
        """
        raw = pivot.get("adjustments", {})
        nested_hps = raw.get("hyperparameters") if isinstance(raw.get("hyperparameters"), dict) else None
        nested_env = raw.get("env_kwargs") if isinstance(raw.get("env_kwargs"), dict) else None
        # LLM sometimes puts policy_kwargs inside adjustments instead of top-level.
        # Promote it so the arch-change path sees it and best_model.zip is not
        # overwritten with a mismatched architecture.
        nested_pky = raw.get("policy_kwargs") if isinstance(raw.get("policy_kwargs"), dict) else None

        if nested_hps is not None or nested_env is not None or nested_pky is not None:
            # Rebuild adjustments: scalar HP keys only — no nested dicts
            flat = {k: v for k, v in raw.items() if k not in ("hyperparameters", "env_kwargs", "policy_kwargs")}
            if nested_hps:
                flat.update(nested_hps)
            pivot = {**pivot, "adjustments": flat}
            # Promote nested env_kwargs to top-level if not already set
            if nested_env and not pivot.get("env_kwargs"):
                pivot = {**pivot, "env_kwargs": nested_env}
            # Promote nested policy_kwargs to top-level if not already set
            if nested_pky and not pivot.get("policy_kwargs"):
                pivot = {**pivot, "policy_kwargs": nested_pky}

        return pivot

    @staticmethod
    def _clamp_rl_adjustments(adjustments: dict, task_type: str) -> dict:
        """Clamp LLM-proposed pivot hyperparameters to safe RL ranges."""
        if task_type != "rl":
            return adjustments
        _RANGES = {
            "learning_rate":  (1e-5,  1e-2),
            "n_steps":        (1024,  4096),
            "batch_size":     (64,    512),
            "n_epochs":       (3,     20),
            "gamma":          (0.90,  0.999),
            "gae_lambda":     (0.80,  0.99),
            "clip_range":     (0.1,   0.4),
            "clip_range_vf":  (0.1,   0.4),
            "ent_coef":       (0.0,   0.1),
            "vf_coef":        (0.1,   1.0),
            "max_grad_norm":  (0.3,   1.0),
            "target_kl":      (0.01,  0.05),
        }
        clamped = {}
        for k, v in adjustments.items():
            if k in _RANGES and isinstance(v, (int, float)):
                lo, hi = _RANGES[k]
                clamped[k] = max(lo, min(hi, v))
            else:
                clamped[k] = v
        # batch_size must not exceed n_steps
        if "batch_size" in clamped and "n_steps" in clamped:
            clamped["batch_size"] = min(clamped["batch_size"], clamped["n_steps"])
        if clamped != adjustments:
            logger.info("LoopStateMachine: clamped pivot adjustments %s → %s", adjustments, clamped)
        return clamped

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

    # ── Manifest helpers ───────────────────────────────────────────────────────

    def _manifest_path(self, mission_id: str) -> str:
        return os.path.join(settings.data_path, "missions", mission_id, "requirements.json")

    def _load_or_create_manifest(self, mission_id: str, mission: Mission) -> RequirementManifest:
        path = self._manifest_path(mission_id)
        if os.path.isfile(path):
            try:
                return RequirementManifest.load(path)
            except Exception as exc:
                logger.warning("LoopStateMachine: could not load manifest for %s: %s — regenerating", mission_id, exc)
        manifest = generate_manifest(
            mission_id=mission_id,
            goal=mission.goal,
            task_type=mission.task_type,
            target_metric=mission.target_metric or {},
        )
        manifest.save(path)
        return manifest

    def _save_manifest(self, mission_id: str, manifest: RequirementManifest) -> None:
        manifest.save(self._manifest_path(mission_id))

    # ── Goal metric evaluation ─────────────────────────────────────────────────

    def _run_goal_metric_eval(self, mission_id: str, plan: dict, metric_name: str) -> Optional[float]:
        """Run deterministic rollouts with the best checkpoint and return MAX goal metric.

        Returns the best single-episode value across 10 episodes — reflects the agent's
        peak capability rather than its average, which is what "achieve X" goals require.
        Called in a thread (via asyncio.to_thread) so it doesn't block the event loop.
        """
        import sys
        import numpy as np

        env_id = plan.get("env_id", "")
        algorithm = plan.get("algorithm", "PPO").upper()
        checkpoint_dir = os.path.join(settings.data_path, "missions", mission_id, "checkpoints")
        checkpoint_path = os.path.join(checkpoint_dir, "best_model.zip")
        if not os.path.isfile(checkpoint_path):
            checkpoint_path = os.path.join(checkpoint_dir, "last_model.zip")
        if not os.path.isfile(checkpoint_path):
            logger.warning("LoopStateMachine: no checkpoint for goal metric eval mission=%s", mission_id)
            return None

        try:
            project_root = os.path.abspath(os.path.join(settings.data_path, ".."))
            if project_root not in sys.path:
                sys.path.insert(0, project_root)

            if env_id == "Tetris-v0":
                from envs.tetris_env import register
                register()
            elif env_id == "Snake-v0":
                from envs.snake_env import register
                register()

            import gymnasium as gym
            from stable_baselines3 import PPO, SAC, A2C, DQN, TD3

            algo_cls = {"PPO": PPO, "SAC": SAC, "A2C": A2C, "DQN": DQN, "TD3": TD3}.get(algorithm, PPO)
            env = gym.make(env_id)
            model = algo_cls.load(checkpoint_path, env=env)

            values = []
            for _ in range(10):
                obs, _ = env.reset()
                done = False
                ep_val = 0.0
                while not done:
                    action, _ = model.predict(obs, deterministic=True)
                    obs, _, terminated, truncated, info = env.step(action)
                    done = terminated or truncated
                    if done:
                        ep_val = float(info.get(metric_name, 0))
                values.append(ep_val)

            env.close()
            return float(max(values)) if values else None
        except Exception as exc:
            logger.warning("LoopStateMachine: goal metric eval error: %s", exc)
            return None

    async def _append_telemetry_metric(
        self, mission_id: str, name: str, value: float, iteration: int
    ) -> None:
        """Write a metric event to telemetry.jsonl and broadcast to connected HUD clients."""
        import json as _json
        from backend.services.connection_manager import manager

        payload = {
            "type": "metric",
            "mission_id": mission_id,
            "name": name,
            "value": value,
            "step": iteration * 500000,  # synthetic step so sparkline orders correctly
            "iteration": iteration,
        }
        path = os.path.join(settings.data_path, "missions", mission_id, "telemetry.jsonl")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "a") as f:
            f.write(_json.dumps(payload) + "\n")
        await manager.broadcast(mission_id, payload)

    # ── Sandbox polling ────────────────────────────────────────────────────────

    def _read_telemetry_metrics(self, mission_id: str, offset: int = 0) -> dict:
        """Return {metric_name: peak_value} for metric events written to telemetry.jsonl since offset.

        Uses the MAX value seen per metric key so that the state machine always
        records the iteration's best performance, not just the last eval snapshot.
        """
        import json as _json
        path = os.path.join(settings.data_path, "missions", mission_id, "telemetry.jsonl")
        metrics: dict = {}
        if not os.path.isfile(path):
            return metrics
        with open(path, "r") as f:
            f.seek(offset)
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    event = _json.loads(line)
                    if event.get("type") == "metric" and "name" in event and "value" in event:
                        name, val = event["name"], event["value"]
                        if name not in metrics or val > metrics[name]:
                            metrics[name] = val
                except Exception:
                    pass
        return metrics

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
            # Ignore benign warnings (telemetry timeouts, warm-start mismatches)
            # and only flag real Python errors with a traceback.
            fatal_lines = [
                line for line in content.splitlines()
                if ("Traceback" in line or "Error" in line)
                and "Telemetry error" not in line
                and "Warm-start skipped" not in line
            ]
            if fatal_lines:
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
