"""Unit tests for LoopStateMachine static helpers."""
from __future__ import annotations

import json
import pytest

from backend.loop.state_machine import LoopStateMachine


# ── _clamp_rl_adjustments ─────────────────────────────────────────────────────

def test_clamp_noop_for_non_rl():
    adj = {"learning_rate": 999, "n_steps": 1}
    result = LoopStateMachine._clamp_rl_adjustments(adj, "ml")
    assert result == adj


def test_clamp_learning_rate_too_high():
    result = LoopStateMachine._clamp_rl_adjustments({"learning_rate": 1.0}, "rl")
    assert result["learning_rate"] == 1e-2


def test_clamp_learning_rate_too_low():
    result = LoopStateMachine._clamp_rl_adjustments({"learning_rate": 1e-10}, "rl")
    assert result["learning_rate"] == 1e-5


def test_clamp_n_steps_too_low():
    result = LoopStateMachine._clamp_rl_adjustments({"n_steps": 64}, "rl")
    assert result["n_steps"] == 1024


def test_clamp_n_steps_too_high():
    result = LoopStateMachine._clamp_rl_adjustments({"n_steps": 99999}, "rl")
    assert result["n_steps"] == 4096


def test_clamp_n_epochs_too_high():
    result = LoopStateMachine._clamp_rl_adjustments({"n_epochs": 80}, "rl")
    assert result["n_epochs"] == 20


def test_clamp_n_epochs_too_low():
    result = LoopStateMachine._clamp_rl_adjustments({"n_epochs": 1}, "rl")
    assert result["n_epochs"] == 3


def test_clamp_batch_size_capped_by_n_steps():
    result = LoopStateMachine._clamp_rl_adjustments(
        {"n_steps": 512, "batch_size": 1024}, "rl"
    )
    assert result["batch_size"] <= result["n_steps"]


def test_clamp_valid_values_unchanged():
    adj = {
        "learning_rate": 3e-4,
        "n_steps": 2048,
        "batch_size": 64,
        "n_epochs": 10,
        "gamma": 0.99,
        "ent_coef": 0.01,
    }
    result = LoopStateMachine._clamp_rl_adjustments(adj, "rl")
    assert result == adj


def test_clamp_unknown_keys_passed_through():
    adj = {"custom_flag": True, "learning_rate": 3e-4}
    result = LoopStateMachine._clamp_rl_adjustments(adj, "rl")
    assert result["custom_flag"] is True


def test_clamp_ent_coef_bounds():
    assert LoopStateMachine._clamp_rl_adjustments({"ent_coef": -0.5}, "rl")["ent_coef"] == 0.0
    assert LoopStateMachine._clamp_rl_adjustments({"ent_coef": 0.5}, "rl")["ent_coef"] == 0.1


def test_clamp_gamma_bounds():
    assert LoopStateMachine._clamp_rl_adjustments({"gamma": 0.5}, "rl")["gamma"] == 0.90
    assert LoopStateMachine._clamp_rl_adjustments({"gamma": 1.0}, "rl")["gamma"] == 0.999


def test_clamp_empty_adjustments():
    result = LoopStateMachine._clamp_rl_adjustments({}, "rl")
    assert result == {}


# ── _read_telemetry_metrics ───────────────────────────────────────────────────

def _write_telemetry(path, events):
    with open(path, "w") as f:
        for e in events:
            f.write(json.dumps(e) + "\n")


def test_read_telemetry_returns_max_not_last(tmp_path, monkeypatch):
    monkeypatch.setattr("backend.config.settings.data_path", str(tmp_path))
    mission_id = "test-mission"
    tel_path = tmp_path / "missions" / mission_id / "telemetry.jsonl"
    tel_path.parent.mkdir(parents=True)
    _write_telemetry(tel_path, [
        {"type": "metric", "name": "mean_reward", "value": 30.0},
        {"type": "metric", "name": "mean_reward", "value": 164.24},
        {"type": "metric", "name": "mean_reward", "value": 116.7},  # last is not the max
    ])
    sm = object.__new__(LoopStateMachine)
    result = sm._read_telemetry_metrics(mission_id)
    assert result["mean_reward"] == 164.24


def test_read_telemetry_max_wins_across_multiple_metrics(tmp_path, monkeypatch):
    monkeypatch.setattr("backend.config.settings.data_path", str(tmp_path))
    mission_id = "test-multi"
    tel_path = tmp_path / "missions" / mission_id / "telemetry.jsonl"
    tel_path.parent.mkdir(parents=True)
    _write_telemetry(tel_path, [
        {"type": "metric", "name": "mean_reward", "value": 10.0},
        {"type": "metric", "name": "accuracy", "value": 0.75},
        {"type": "metric", "name": "mean_reward", "value": 50.0},
        {"type": "metric", "name": "accuracy", "value": 0.60},  # lower, should not win
    ])
    sm = object.__new__(LoopStateMachine)
    result = sm._read_telemetry_metrics(mission_id)
    assert result["mean_reward"] == 50.0
    assert result["accuracy"] == 0.75


def test_read_telemetry_respects_offset(tmp_path, monkeypatch):
    monkeypatch.setattr("backend.config.settings.data_path", str(tmp_path))
    mission_id = "test-offset"
    tel_path = tmp_path / "missions" / mission_id / "telemetry.jsonl"
    tel_path.parent.mkdir(parents=True)
    line1 = json.dumps({"type": "metric", "name": "mean_reward", "value": 999.0}) + "\n"
    line2 = json.dumps({"type": "metric", "name": "mean_reward", "value": 20.0}) + "\n"
    with open(tel_path, "w") as f:
        f.write(line1)
    offset = tel_path.stat().st_size
    with open(tel_path, "a") as f:
        f.write(line2)
    sm = object.__new__(LoopStateMachine)
    # With offset, should only see line2 (value=20), not line1 (value=999)
    result = sm._read_telemetry_metrics(mission_id, offset=offset)
    assert result["mean_reward"] == 20.0


def test_read_telemetry_empty_file(tmp_path, monkeypatch):
    monkeypatch.setattr("backend.config.settings.data_path", str(tmp_path))
    mission_id = "test-empty"
    tel_path = tmp_path / "missions" / mission_id / "telemetry.jsonl"
    tel_path.parent.mkdir(parents=True)
    tel_path.write_text("")
    sm = object.__new__(LoopStateMachine)
    assert sm._read_telemetry_metrics(mission_id) == {}


def test_read_telemetry_missing_file(tmp_path, monkeypatch):
    monkeypatch.setattr("backend.config.settings.data_path", str(tmp_path))
    sm = object.__new__(LoopStateMachine)
    assert sm._read_telemetry_metrics("nonexistent-mission") == {}


# ── _hp_changed (no-op pivot filter) ─────────────────────────────────────────

def _make_noop_filter(plan_hps: dict):
    """Return a closure that mimics the _hp_changed logic defined inside _step."""
    def _hp_changed(k, proposed):
        current = plan_hps.get(k)
        if current is None:
            return True
        try:
            return float(current) != float(proposed)
        except (TypeError, ValueError):
            return current != proposed
    return _hp_changed


def test_noop_filter_same_float_values():
    f = _make_noop_filter({"learning_rate": 0.0005, "batch_size": 128})
    assert not f("learning_rate", 0.0005)
    assert not f("batch_size", 128)


def test_noop_filter_string_vs_float():
    """LLM often returns strings; should still compare equal to float in plan."""
    f = _make_noop_filter({"learning_rate": 0.0005, "batch_size": 128})
    assert not f("learning_rate", "0.0005")
    assert not f("batch_size", "128")


def test_noop_filter_detects_real_change():
    f = _make_noop_filter({"learning_rate": 0.0005})
    assert f("learning_rate", 0.001)
    assert f("learning_rate", "0.001")


def test_noop_filter_unknown_key_treated_as_change():
    f = _make_noop_filter({"learning_rate": 0.0005})
    assert f("n_steps", 2048)


def test_noop_filter_non_numeric_equality():
    f = _make_noop_filter({"mode": "fast"})
    assert not f("mode", "fast")
    assert f("mode", "slow")


# ── old_hps snapshot (display shows old→new, not new→new) ────────────────────

def test_display_uses_pre_update_value():
    """Verify that snapshotting before update produces old→new, not new→new."""
    plan_hps = {"learning_rate": 0.001, "batch_size": 64}
    real_adjustments = {"learning_rate": 0.0005, "batch_size": 128}

    old_hps = {k: plan_hps.get(k) for k in real_adjustments}
    plan_hps.update(real_adjustments)  # mutate, as state_machine does

    hp_strs = []
    for k, v in real_adjustments.items():
        old_v = old_hps.get(k)
        hp_strs.append(f"{k}: {old_v}→{v}" if old_v is not None else f"{k}={v}")

    assert "learning_rate: 0.001→0.0005" in hp_strs
    assert "batch_size: 64→128" in hp_strs


# ── _is_algorithm_locked ─────────────────────────────────────────────────────

_locked = LoopStateMachine._is_algorithm_locked


def test_algo_locked_explicit_name_in_goal():
    assert _locked("Train a Snake-v0 DQN agent to achieve mean_reward of 100", "DQN")


def test_algo_locked_case_insensitive():
    assert _locked("train a snake dqn agent", "DQN")


def test_algo_locked_different_algo_not_locked():
    assert not _locked("Train a Snake-v0 DQN agent", "PPO")


def test_algo_locked_no_algo_in_goal():
    assert not _locked("Train a Snake-v0 agent to achieve mean_reward of 100", "DQN")


def test_algo_locked_partial_word_not_matched():
    # "DQNX" in goal should not lock "DQN"
    assert not _locked("Use DQNX strategy", "DQN")


def test_algo_locked_ppo_named():
    assert _locked("Train PPO on CartPole-v1", "PPO")


# ── _normalize_pivot ─────────────────────────────────────────────────────────

_normalize = LoopStateMachine._normalize_pivot


def test_normalize_flat_adjustments_unchanged():
    pivot = {"reason": "plateau", "adjustments": {"learning_rate": 0.001, "batch_size": 64}}
    assert _normalize(pivot) == pivot


def test_normalize_nested_hyperparameters_flattened():
    pivot = {
        "reason": "plateau",
        "adjustments": {"hyperparameters": {"learning_rate": 0.0005, "batch_size": 128}},
    }
    result = _normalize(pivot)
    assert result["adjustments"] == {"learning_rate": 0.0005, "batch_size": 128}


def test_normalize_nested_env_kwargs_promoted():
    pivot = {
        "reason": "plateau",
        "adjustments": {
            "hyperparameters": {"learning_rate": 0.0005},
            "env_kwargs": {"food_reward": 20.0},
        },
    }
    result = _normalize(pivot)
    assert result["adjustments"] == {"learning_rate": 0.0005}
    assert result["env_kwargs"] == {"food_reward": 20.0}


def test_normalize_top_level_env_kwargs_not_overwritten():
    """If pivot already has top-level env_kwargs, don't clobber it."""
    pivot = {
        "reason": "plateau",
        "adjustments": {"env_kwargs": {"food_reward": 10.0}},
        "env_kwargs": {"food_reward": 20.0},
    }
    result = _normalize(pivot)
    assert result["env_kwargs"] == {"food_reward": 20.0}


def test_normalize_mixed_flat_and_nested():
    """Flat HP keys alongside nested hyperparameters dict are all merged."""
    pivot = {
        "reason": "plateau",
        "adjustments": {
            "batch_size": 64,
            "hyperparameters": {"learning_rate": 0.0005, "gamma": 0.99},
        },
    }
    result = _normalize(pivot)
    assert result["adjustments"] == {"batch_size": 64, "learning_rate": 0.0005, "gamma": 0.99}


def test_display_does_not_show_noop_arrow():
    """A key that passes through unchanged (filtered by _hp_changed) never appears."""
    plan_hps = {"learning_rate": 0.0005, "batch_size": 128}
    adjustments = {"learning_rate": 0.0005, "batch_size": 128}

    def _hp_changed(k, proposed):
        current = plan_hps.get(k)
        if current is None:
            return True
        try:
            return float(current) != float(proposed)
        except (TypeError, ValueError):
            return current != proposed

    real_adjustments = {k: v for k, v in adjustments.items() if _hp_changed(k, v)}
    assert real_adjustments == {}, "identical HPs must be filtered out"


# ── startup seed iteration ────────────────────────────────────────────────────

def test_seed_uses_best_metric_iteration_when_set():
    """When mission has a known best_metric_iteration, the pivot engine seed uses that iter."""
    from backend.loop.pivots import PivotEngine

    # Simulate: DB reports best=164.24 was achieved at iteration 45
    engine = PivotEngine({"mean_reward": 200})
    engine.record(45, {"mean_reward": 164.24})

    # The best entry should report iter 45, not None
    assert engine.best_metric_iteration() == 45


def test_seed_falls_back_to_minus_one_when_iteration_none():
    """When mission.best_metric_iteration is None, seed at -1 → best_metric_iteration returns None."""
    from backend.loop.pivots import PivotEngine

    # Simulate the old path: seed at -1
    engine = PivotEngine({"mean_reward": 200})
    engine.record(-1, {"mean_reward": 164.24})

    # -1 sentinel maps to None in the UI
    assert engine.best_metric_iteration() is None


# ── _load_persisted_best contamination guard ──────────────────────────────────

class _MockMission:
    def __init__(self, target_metric, best_metric_value=None, best_metric_iteration=None):
        self.target_metric = target_metric
        self.best_metric_value = best_metric_value
        self.best_metric_iteration = best_metric_iteration


def _make_sm():
    """Return a LoopStateMachine with telemetry stubbed to return empty dict."""
    sm = LoopStateMachine.__new__(LoopStateMachine)
    sm._read_telemetry_metrics = lambda mission_id, offset: {}
    return sm


def test_load_persisted_best_ignores_negative_db_for_custom_metric():
    """Negative DB value (mean_reward contamination) must be discarded for non-mean_reward targets."""
    sm = _make_sm()
    mission = _MockMission(
        target_metric={"lines_cleared": 20},
        best_metric_value="-120.124",
    )
    result = sm._load_persisted_best("fake-id", mission)
    assert result is None


def test_load_persisted_best_accepts_positive_db_for_custom_metric():
    """A positive DB value is a valid best for a custom metric like lines_cleared."""
    sm = _make_sm()
    mission = _MockMission(
        target_metric={"lines_cleared": 20},
        best_metric_value="5.0",
    )
    result = sm._load_persisted_best("fake-id", mission)
    assert result == 5.0


def test_load_persisted_best_accepts_negative_db_for_mean_reward():
    """Negative DB value is valid when the target metric IS mean_reward."""
    sm = _make_sm()
    mission = _MockMission(
        target_metric={"mean_reward": 200},
        best_metric_value="-100.5",
    )
    result = sm._load_persisted_best("fake-id", mission)
    assert result == -100.5


def test_load_persisted_best_returns_none_when_all_sources_empty():
    sm = _make_sm()
    mission = _MockMission(target_metric={"lines_cleared": 20}, best_metric_value=None)
    result = sm._load_persisted_best("fake-id", mission)
    assert result is None


def test_load_persisted_best_prefers_telemetry_over_negative_db():
    """Telemetry scan wins over a negative DB value for custom targets."""
    sm = _make_sm()
    sm._read_telemetry_metrics = lambda mission_id, offset: {"lines_cleared": 3.0}
    mission = _MockMission(
        target_metric={"lines_cleared": 20},
        best_metric_value="-99.0",
    )
    result = sm._load_persisted_best("fake-id", mission)
    assert result == 3.0


# ── _wait_for_sandbox error detection ─────────────────────────────────────────

import asyncio
from unittest.mock import MagicMock, patch


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


def _make_sandbox_sm(log_content: str, tmp_path):
    sm = _make_sm()
    log_path = tmp_path / "sandbox.log"
    log_path.write_text(log_content)
    sandbox = MagicMock()
    sandbox.is_alive.return_value = False
    sandbox.get_log_path.return_value = str(log_path)
    sm._sandbox = sandbox
    return sm


def test_wait_sandbox_real_traceback_returns_content(tmp_path):
    """A Python Traceback is correctly flagged as a fatal error."""
    content = "Traceback (most recent call last):\n  File train.py, line 5\nKeyError: 'foo'\n"
    sm = _make_sandbox_sm(content, tmp_path)
    result = _run(sm._wait_for_sandbox("mid"))
    assert result == content


def test_wait_sandbox_telemetry_error_not_flagged(tmp_path):
    """Telemetry timeout warnings do not trigger the healer."""
    content = "WARNING:root:Telemetry error: HTTPConnectionPool(host='127.0.0.1', port=8200): Read timed out.\n"
    sm = _make_sandbox_sm(content, tmp_path)
    result = _run(sm._wait_for_sandbox("mid"))
    assert result is None


def test_wait_sandbox_warm_start_skipped_not_flagged(tmp_path):
    """Architecture mismatch warm-start warning does not trigger the healer."""
    content = "WARNING:root:Warm-start skipped (architecture mismatch or load error): Error(s) in loading state_dict\n"
    sm = _make_sandbox_sm(content, tmp_path)
    result = _run(sm._wait_for_sandbox("mid"))
    assert result is None


def test_wait_sandbox_clean_exit_returns_none(tmp_path):
    """Clean sandbox output with no errors returns None."""
    content = "Training complete. Steps=500000\n"
    sm = _make_sandbox_sm(content, tmp_path)
    result = _run(sm._wait_for_sandbox("mid"))
    assert result is None


def test_wait_sandbox_mixed_benign_and_fatal_returns_content(tmp_path):
    """Benign warnings alongside a real Traceback still flags as fatal."""
    content = (
        "WARNING:root:Telemetry error: Read timed out.\n"
        "WARNING:root:Warm-start skipped (architecture mismatch or load error): ...\n"
        "Traceback (most recent call last):\nValueError: bad value\n"
    )
    sm = _make_sandbox_sm(content, tmp_path)
    result = _run(sm._wait_for_sandbox("mid"))
    assert result == content


# ── propose_pivot current plan context ───────────────────────────────────────

def test_propose_pivot_passes_current_plan_context(monkeypatch):
    """propose_pivot receives current policy_kwargs, hyperparameters, and env_kwargs
    so the LLM can avoid re-proposing already-applied changes."""
    import asyncio
    from unittest.mock import AsyncMock, MagicMock, patch

    captured = {}

    async def _fake_propose(
        self,
        current_metrics,
        history,
        escalation_level=0,
        current_algorithm="PPO",
        algorithm_locked=False,
        current_policy_kwargs=None,
        current_hyperparameters=None,
        current_env_kwargs=None,
    ):
        captured["policy_kwargs"] = current_policy_kwargs
        captured["hyperparameters"] = current_hyperparameters
        captured["env_kwargs"] = current_env_kwargs
        return {"reason": "test", "adjustments": {}}

    from backend.agent.lead_agent import LeadAgent
    monkeypatch.setattr(LeadAgent, "propose_pivot", _fake_propose)

    from backend.loop.pivots import PivotEngine
    engine = PivotEngine({"food_eaten": 30})
    # seed 3 identical values to trigger plateau
    for i in range(3):
        engine.record(i, {"food_eaten": 1.0})

    assert engine.needs_pivot()

    plan = {
        "algorithm": "PPO",
        "hyperparameters": {
            "policy_kwargs": {"net_arch": [256, 256]},
            "learning_rate": 3e-4,
            "n_steps": 2048,
        },
        "env_kwargs": {"food_reward": 15.0, "death_penalty": -2.0},
    }

    sm = LoopStateMachine.__new__(LoopStateMachine)
    sm._agent = LeadAgent.__new__(LeadAgent)

    async def _run_pivot():
        current_algo = plan.get("algorithm", "PPO")
        return await sm._agent.propose_pivot(
            {"food_eaten": 1.0},
            engine.history_snapshot(),
            escalation_level=engine.escalation_level(),
            current_algorithm=current_algo,
            algorithm_locked=False,
            current_policy_kwargs=plan.get("hyperparameters", {}).get("policy_kwargs"),
            current_hyperparameters={
                k: v for k, v in plan.get("hyperparameters", {}).items()
                if k != "policy_kwargs"
            } or None,
            current_env_kwargs=plan.get("env_kwargs") or None,
        )

    asyncio.get_event_loop().run_until_complete(_run_pivot())

    assert captured["policy_kwargs"] == {"net_arch": [256, 256]}
    assert captured["hyperparameters"] == {"learning_rate": 3e-4, "n_steps": 2048}
    assert captured["env_kwargs"] == {"food_reward": 15.0, "death_penalty": -2.0}


# ── _load_goal_metric_history ──────────────────────────────────────────────────

def test_load_goal_metric_history_reads_per_iter_values(tmp_path):
    """Reads food_eaten events from telemetry.jsonl and returns one entry per iteration."""
    import json as _json
    from unittest.mock import patch

    tel = tmp_path / "telemetry.jsonl"
    events = [
        {"type": "metric", "name": "food_eaten", "value": 1.0, "iteration": 0},
        {"type": "metric", "name": "mean_reward", "value": 30.0, "iteration": 0},
        {"type": "metric", "name": "food_eaten", "value": 3.0, "iteration": 1},
        {"type": "metric", "name": "food_eaten", "value": 2.0, "iteration": 2},
    ]
    tel.write_text("\n".join(_json.dumps(e) for e in events))

    with patch("backend.config.settings") as mock_settings:
        mock_settings.data_path = str(tmp_path.parent)
        # Patch the path to point at our tmp file
        import backend.loop.state_machine as sm_mod
        original = sm_mod.settings
        sm_mod.settings = mock_settings
        mock_settings.data_path = str(tmp_path)
        # Use the actual method via a bare instance (no __init__ needed for static-style call)
        sm = LoopStateMachine.__new__(LoopStateMachine)

        import os
        with patch.object(sm_mod.os.path, "join", side_effect=lambda *a: str(tel) if "telemetry" in str(a) else os.path.join(*a)):
            result = sm._load_goal_metric_history("any-id", "food_eaten")

        sm_mod.settings = original

    assert result == [
        {"iteration": 0, "food_eaten": 1.0},
        {"iteration": 1, "food_eaten": 3.0},
        {"iteration": 2, "food_eaten": 2.0},
    ]


def test_load_goal_metric_history_missing_file_returns_empty(tmp_path):
    """Returns empty list when telemetry file does not exist."""
    from unittest.mock import patch
    import backend.loop.state_machine as sm_mod

    sm = LoopStateMachine.__new__(LoopStateMachine)
    with patch("backend.loop.state_machine.settings") as mock_settings:
        mock_settings.data_path = str(tmp_path / "nonexistent")
        result = sm._load_goal_metric_history("any-id", "food_eaten")
    assert result == []


# ── env_kwargs merge (not replace) ───────────────────────────────────────────

def _apply_env_kwargs(plan, pivot_env_kwargs):
    """Mirror the merge logic from state_machine._step."""
    current = plan.get("env_kwargs") or {}
    plan["env_kwargs"] = dict(current, **pivot_env_kwargs)


def test_env_kwargs_pivot_merges_not_replaces():
    """A pivot that only specifies one env_kwarg must not erase others already in the plan."""
    plan = {
        "algorithm": "PPO",
        "env_kwargs": {"food_reward": 22.0, "distance_weight": 1.0, "survival_bonus": 0.05},
    }
    _apply_env_kwargs(plan, {"survival_bonus": 0.07})

    assert plan["env_kwargs"]["food_reward"] == 22.0      # preserved
    assert plan["env_kwargs"]["distance_weight"] == 1.0   # preserved — was destroyed by old bug
    assert plan["env_kwargs"]["survival_bonus"] == 0.07   # updated


def test_env_kwargs_pivot_replaces_specific_key():
    """A pivot that sets distance_weight=0.0 does override that specific key."""
    plan = {
        "env_kwargs": {"food_reward": 22.0, "distance_weight": 1.0},
    }
    _apply_env_kwargs(plan, {"distance_weight": 0.0})

    assert plan["env_kwargs"]["food_reward"] == 22.0      # preserved
    assert plan["env_kwargs"]["distance_weight"] == 0.0   # overridden


def test_env_kwargs_merge_when_plan_has_none():
    """Merge still works when plan has no prior env_kwargs (None or missing)."""
    plan = {}  # type: dict
    _apply_env_kwargs(plan, {"food_reward": 25.0})

    assert plan["env_kwargs"] == {"food_reward": 25.0}


# ── _clamp_env_kwargs ─────────────────────────────────────────────────────────

def test_clamp_env_kwargs_distance_weight_zero_raised_to_min():
    """distance_weight=0 must be clamped to 0.1 — zeroing disables navigation shaping."""
    result = LoopStateMachine._clamp_env_kwargs({"distance_weight": 0.0, "food_reward": 20.0})
    assert result["distance_weight"] == 0.1
    assert result["food_reward"] == 20.0  # other keys unchanged


def test_clamp_env_kwargs_distance_weight_negative_raised_to_min():
    result = LoopStateMachine._clamp_env_kwargs({"distance_weight": -1.0})
    assert result["distance_weight"] == 0.1


def test_clamp_env_kwargs_distance_weight_valid_unchanged():
    result = LoopStateMachine._clamp_env_kwargs({"distance_weight": 1.5})
    assert result["distance_weight"] == 1.5


def test_clamp_env_kwargs_no_distance_weight_passthrough():
    """Keys other than distance_weight pass through without modification."""
    result = LoopStateMachine._clamp_env_kwargs({"food_reward": 25.0, "survival_bonus": 0.05})
    assert result == {"food_reward": 25.0, "survival_bonus": 0.05}


def test_load_goal_metric_history_last_value_per_iter_wins(tmp_path):
    """When multiple entries exist for the same iteration, last one wins."""
    import json as _json
    from unittest.mock import patch
    import backend.loop.state_machine as sm_mod

    tel = tmp_path / "telemetry.jsonl"
    events = [
        {"type": "metric", "name": "food_eaten", "value": 1.0, "iteration": 0},
        {"type": "metric", "name": "food_eaten", "value": 5.0, "iteration": 0},
    ]
    tel.write_text("\n".join(_json.dumps(e) for e in events))

    sm = LoopStateMachine.__new__(LoopStateMachine)
    import os
    with patch("backend.loop.state_machine.settings") as mock_settings:
        mock_settings.data_path = str(tmp_path)
        with patch.object(sm_mod.os.path, "join", side_effect=lambda *a: str(tel) if "telemetry" in str(a) else os.path.join(*a)):
            result = sm._load_goal_metric_history("any-id", "food_eaten")

    assert result == [{"iteration": 0, "food_eaten": 5.0}]
