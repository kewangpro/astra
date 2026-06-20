"""
Integration tests for LoopStateMachine.

All external I/O is mocked:
  - LeadAgent / CodeGenerator / ErrorAnalyzer / ModelManager — deterministic fakes
  - SandboxManager — configurable in-memory mock (no real processes)
  - SpecialistEvaluator — returns scripted metrics
  - AsyncSessionLocal — in-memory SQLite (via conftest.patch_db)
  - asyncio.sleep — patched to zero duration
  - LoopStateMachine._crystallize — no-op (crystallizer deps not available in CI)
"""
from __future__ import annotations

import os
import tempfile
import uuid
from typing import Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio
from sqlalchemy import select

from backend.loop.state_machine import LoopStateMachine
from backend.models.mission import Mission, MissionStatus
from backend.models.approval import ApprovalGate, ApprovalStatus
from backend.evaluator.manifest_evaluator import ManifestEvaluator


# ── Mock helpers ───────────────────────────────────────────────────────────────

class _MockLeadAgent:
    async def plan(self, goal, task_type, target_metric):
        return {
            "task_type": "rl",
            "algorithm": "PPO",
            "hyperparameters": {"learning_rate": 3e-4, "gamma": 0.99},
            "sandbox_memory_gb": 4.0,
        }

    async def propose_pivot(self, current_metrics, history):
        return {"reason": "plateau_detected", "adjustments": {"learning_rate": 1e-4}}

    def flush_iteration_context(self):
        pass


class _MockCodeGen:
    def __init__(self, tmp_dir: str):
        self._tmp_dir = tmp_dir
        self._path = os.path.join(tmp_dir, "train.py")
        with open(self._path, "w") as f:
            f.write("print('mock training script')\n")

    async def generate_training_script(self, mission_id, plan, current_iteration=0):
        return self._path


class _MockHealer:
    def __init__(self, tmp_dir: str):
        self._tmp_dir = tmp_dir

    async def fix_script(self, script_path, error_output, iteration=0, prior_errors=None, mission_id=None, domain=None):
        fixed = os.path.join(self._tmp_dir, f"train_fixed_{iteration}.py")
        with open(fixed, "w") as f:
            f.write("print('fixed script')\n")
        return fixed


class _MockSandbox:
    """Sandbox that exits immediately. Writes configurable log content."""
    def __init__(self, tmp_dir: str, error_content: str = ""):
        self._tmp_dir = tmp_dir
        self._error_content = error_content
        self._log_paths: dict = {}

    def launch(self, mission_id, script_path, **kwargs):
        log_path = os.path.join(self._tmp_dir, f"{mission_id}.log")
        with open(log_path, "w") as f:
            f.write(self._error_content)
        self._log_paths[mission_id] = log_path
        return (1234, None)

    def is_alive(self, mission_id):
        return False

    def get_log_path(self, mission_id):
        return self._log_paths.get(mission_id, "")

    def terminate(self, mission_id):
        pass


class _ErrorThenSuccessSandbox:
    """First launch writes an error log; subsequent launches are clean."""
    def __init__(self, tmp_dir: str):
        self._tmp_dir = tmp_dir
        self._call_count = 0
        self._log_paths: dict = {}

    def launch(self, mission_id, script_path, **kwargs):
        self._call_count += 1
        content = "Traceback\nNameError: mock error" if self._call_count == 1 else "Training complete."
        log_path = os.path.join(self._tmp_dir, f"{mission_id}_{self._call_count}.log")
        with open(log_path, "w") as f:
            f.write(content)
        self._log_paths[mission_id] = log_path
        return (1234, None)

    def is_alive(self, mission_id):
        return False

    def get_log_path(self, mission_id):
        return self._log_paths.get(mission_id, "")

    def terminate(self, mission_id):
        pass


class _AlwaysErrorSandbox:
    def __init__(self, tmp_dir: str):
        self._tmp_dir = tmp_dir
        self._log_paths: dict = {}

    def launch(self, mission_id, script_path, **kwargs):
        # Always use the same log file (append) so offset tracking works across retries
        log_path = os.path.join(self._tmp_dir, f"{mission_id}.log")
        with open(log_path, "a") as f:
            f.write("Traceback\nRuntimeError: always fails\n")
        self._log_paths[mission_id] = log_path
        return (1234, None)

    def is_alive(self, mission_id):
        return False

    def get_log_path(self, mission_id):
        return self._log_paths.get(mission_id, "")

    def terminate(self, mission_id):
        pass


class _SequenceEvaluator:
    """Returns metrics from a scripted sequence; repeats the last entry."""
    def __init__(self, sequence: list):
        self._seq = sequence
        self._idx = 0

    async def evaluate(self, mission_id, plan):
        metrics = self._seq[min(self._idx, len(self._seq) - 1)]
        self._idx += 1
        return {"metrics": metrics, "verdict": "pass"}


async def _noop_crystallize(self, *args, **kwargs):
    pass


class _SkipFileExistsManifestEvaluator(ManifestEvaluator):
    """Auto-passes file_exists requirements (no real checkpoints in tests)
    but evaluates metric_threshold and no_sandbox_error checks normally."""
    def evaluate(self, manifest, metrics, mission_dir, sandbox_ok):
        from datetime import datetime, timezone
        for r in manifest.requirements:
            if r.check_type == "file_exists" and not r.passed:
                r.passed = True
                r.passed_at = datetime.now(timezone.utc).isoformat()
                r.evidence = "test-skip-file-check"
        return super().evaluate(manifest, metrics, mission_dir, sandbox_ok)


def _build_sm(agent, codegen, healer, sandbox, evaluator):
    mm = MagicMock()  # ModelManager — no-op
    sm = LoopStateMachine(
        lead_agent=agent,
        code_generator=codegen,
        error_analyzer=healer,
        model_manager=mm,
        sandbox_manager=sandbox,
        evaluator=evaluator,
    )
    # Override manifest evaluator so tests aren't blocked by missing checkpoint files
    sm._manifest_evaluator = _SkipFileExistsManifestEvaluator()
    return sm


# ── Fixtures ───────────────────────────────────────────────────────────────────

@pytest_asyncio.fixture
async def seeded_mission(db_session, patch_db):
    mission = Mission(
        id=str(uuid.uuid4()),
        goal="Train a Snake RL agent",
        task_type="rl",
        target_metric={"mean_reward": 100.0},
        autonomy_mode="full_autonomy",
        status=MissionStatus.PENDING.value,
        current_iteration=0,
    )
    db_session.add(mission)
    await db_session.commit()
    return mission


# ── Tests ──────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_happy_path_goal_met(seeded_mission, db_session, patch_db, monkeypatch):
    monkeypatch.setattr("backend.loop.state_machine.EVAL_POLL_INTERVAL", 0)

    with tempfile.TemporaryDirectory() as tmp:
        evaluator = _SequenceEvaluator([{"mean_reward": 200.0}])
        sm = _build_sm(
            _MockLeadAgent(),
            _MockCodeGen(tmp),
            _MockHealer(tmp),
            _MockSandbox(tmp),
            evaluator,
        )
        with patch.object(LoopStateMachine, "_crystallize", _noop_crystallize):
            await sm.run(seeded_mission.id)

    await db_session.refresh(seeded_mission)
    assert seeded_mission.status == MissionStatus.COMPLETED.value


@pytest.mark.asyncio
async def test_error_recovery_then_goal_met(seeded_mission, db_session, patch_db, monkeypatch):
    monkeypatch.setattr("backend.loop.state_machine.EVAL_POLL_INTERVAL", 0)

    with tempfile.TemporaryDirectory() as tmp:
        evaluator = _SequenceEvaluator([
            {"mean_reward": 0.0},    # not returned (sandbox errors before eval on attempt 1)
            {"mean_reward": 200.0},  # attempt 2 succeeds → goal met
        ])
        sm = _build_sm(
            _MockLeadAgent(),
            _MockCodeGen(tmp),
            _MockHealer(tmp),
            _ErrorThenSuccessSandbox(tmp),
            evaluator,
        )
        with patch.object(LoopStateMachine, "_crystallize", _noop_crystallize):
            await sm.run(seeded_mission.id)

    await db_session.refresh(seeded_mission)
    assert seeded_mission.status == MissionStatus.COMPLETED.value


@pytest.mark.asyncio
async def test_max_retries_exceeded_marks_failed(seeded_mission, db_session, patch_db, monkeypatch):
    monkeypatch.setattr("backend.loop.state_machine.EVAL_POLL_INTERVAL", 0)

    with tempfile.TemporaryDirectory() as tmp:
        sm = _build_sm(
            _MockLeadAgent(),
            _MockCodeGen(tmp),
            _MockHealer(tmp),
            _AlwaysErrorSandbox(tmp),
            _SequenceEvaluator([{"mean_reward": 0.0}]),
        )
        await sm.run(seeded_mission.id)

    await db_session.refresh(seeded_mission)
    assert seeded_mission.status == MissionStatus.FAILED.value


@pytest.mark.asyncio
async def test_plateau_triggers_pivot_then_goal_met(seeded_mission, db_session, patch_db, monkeypatch):
    monkeypatch.setattr("backend.loop.state_machine.EVAL_POLL_INTERVAL", 0)

    with tempfile.TemporaryDirectory() as tmp:
        evaluator = _SequenceEvaluator([
            {"mean_reward": 10.0},   # iter 0 — below target, no pivot yet
            {"mean_reward": 10.0},   # iter 1 — still stalled
            {"mean_reward": 10.0},   # iter 2 — plateau → pivot applied
            {"mean_reward": 200.0},  # iter 3 — goal met
        ])
        agent = _MockLeadAgent()
        propose_calls = []
        _orig = agent.propose_pivot
        async def _track_pivot(m, h, escalation_level=0, current_algorithm="PPO", algorithm_locked=False):
            propose_calls.append((m, h))
            return await _orig(m, h)
        agent.propose_pivot = _track_pivot

        sm = _build_sm(agent, _MockCodeGen(tmp), _MockHealer(tmp), _MockSandbox(tmp), evaluator)
        with patch.object(LoopStateMachine, "_crystallize", _noop_crystallize):
            await sm.run(seeded_mission.id)

    await db_session.refresh(seeded_mission)
    assert seeded_mission.status == MissionStatus.COMPLETED.value
    assert len(propose_calls) >= 1


@pytest.mark.asyncio
async def test_manifest_reconciled_when_plan_task_type_differs(db_session, patch_db, monkeypatch):
    """Mission created with task_type='rl' (UI default) but plan identifies 'ml'.
    Manifest artifact pattern should be reconciled to checkpoints/model.* after iter 0."""
    monkeypatch.setattr("backend.loop.state_machine.EVAL_POLL_INTERVAL", 0)

    mission = Mission(
        id=str(uuid.uuid4()),
        goal="Train a sklearn classifier on iris to 95% accuracy",
        task_type="rl",  # wrong — simulates user leaving dropdown on default
        target_metric={"accuracy": 0.95},
        autonomy_mode="full_autonomy",
        status=MissionStatus.PENDING.value,
        current_iteration=0,
    )
    db_session.add(mission)
    await db_session.commit()

    class _MLLeadAgent:
        async def plan(self, goal, task_type, target_metric):
            return {
                "task_type": "ml",   # LeadAgent correctly identifies ml
                "algorithm": "RandomForestClassifier",
                "hyperparameters": {"n_estimators": 100},
                "sandbox_memory_gb": 1.0,
            }
        async def propose_pivot(self, current_metrics, history):
            return {"reason": "plateau", "adjustments": {}}
        def flush_iteration_context(self):
            pass

    with tempfile.TemporaryDirectory() as tmp:
        monkeypatch.setattr("backend.config.settings.data_path", tmp)
        evaluator = _SequenceEvaluator([{"accuracy": 1.0}])
        sm = _build_sm(
            _MLLeadAgent(),
            _MockCodeGen(tmp),
            _MockHealer(tmp),
            _MockSandbox(tmp),
            evaluator,
        )
        with patch.object(LoopStateMachine, "_crystallize", _noop_crystallize):
            await sm.run(mission.id)

        # Check that the saved manifest uses the ml artifact pattern
        import json, glob as _glob
        manifest_path = os.path.join(tmp, "missions", mission.id, "requirements.json")
        assert os.path.isfile(manifest_path), "requirements.json not written"
        reqs = json.load(open(manifest_path))["requirements"]
        artifact = next(r for r in reqs if r["check_type"] == "file_exists")
        assert artifact["path_pattern"] == "checkpoints/model.*", (
            f"Expected checkpoints/model.* but got {artifact['path_pattern']}"
        )

    await db_session.refresh(mission)
    assert mission.status == MissionStatus.COMPLETED.value


@pytest.mark.asyncio
async def test_supervised_gate_rejected_marks_failed(db_session, patch_db, monkeypatch):
    monkeypatch.setattr("backend.loop.state_machine.EVAL_POLL_INTERVAL", 0)

    mission = Mission(
        id=str(uuid.uuid4()),
        goal="Fine-tune an LLM",
        task_type="sft",
        target_metric={"eval_loss": 1.5},
        autonomy_mode="supervised",
        status=MissionStatus.PENDING.value,
        current_iteration=0,
    )
    db_session.add(mission)
    await db_session.commit()

    with tempfile.TemporaryDirectory() as tmp:
        sm = _build_sm(
            _MockLeadAgent(),
            _MockCodeGen(tmp),
            _MockHealer(tmp),
            _MockSandbox(tmp),
            _SequenceEvaluator([{"eval_loss": 0.5}]),
        )
        # Simulate gate rejection without polling
        with patch.object(LoopStateMachine, "_request_approval", new=AsyncMock(return_value=False)):
            await sm.run(mission.id)

    await db_session.refresh(mission)
    assert mission.status == MissionStatus.FAILED.value
