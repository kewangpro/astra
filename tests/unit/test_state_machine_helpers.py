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


def test_clamp_exploration_initial_eps_floor():
    """Real incident: successive pivots drifted exploration_initial_eps down to
    0.01 (near-pure-greedy from the start of training) since DQN's exploration
    keys were never added to the clamp table. Must be floored to a value that
    still guarantees real exploration."""
    result = LoopStateMachine._clamp_rl_adjustments({"exploration_initial_eps": 0.01}, "rl")
    assert result["exploration_initial_eps"] == 0.5


def test_clamp_exploration_initial_eps_ceiling():
    result = LoopStateMachine._clamp_rl_adjustments({"exploration_initial_eps": 5.0}, "rl")
    assert result["exploration_initial_eps"] == 1.0


def test_clamp_exploration_fraction_floor():
    """Real incident: exploration_fraction drifted back to 0.3, which combined
    with _inject_curriculum's cumulative total_timesteps across phases decays
    epsilon to its floor before the curriculum's hardest phase even begins."""
    result = LoopStateMachine._clamp_rl_adjustments({"exploration_fraction": 0.3}, "rl")
    assert result["exploration_fraction"] == 0.4


def test_clamp_exploration_fraction_ceiling():
    result = LoopStateMachine._clamp_rl_adjustments({"exploration_fraction": 1.0}, "rl")
    assert result["exploration_fraction"] == 0.8


def test_clamp_exploration_final_eps_floor():
    result = LoopStateMachine._clamp_rl_adjustments({"exploration_final_eps": 0.01}, "rl")
    assert result["exploration_final_eps"] == 0.05


def test_clamp_exploration_final_eps_ceiling():
    result = LoopStateMachine._clamp_rl_adjustments({"exploration_final_eps": 0.9}, "rl")
    assert result["exploration_final_eps"] == 0.2


def test_clamp_gradient_steps_ceiling():
    """Real incident: a pivot drifted gradient_steps to 1000 combined with
    train_freq=1 (train after every single env step) — ~8000x the intended
    compute per env step vs a sane default, burning 10+ hours of CPU on a
    single iteration without finishing."""
    result = LoopStateMachine._clamp_rl_adjustments({"gradient_steps": 1000}, "rl")
    assert result["gradient_steps"] == 4


def test_clamp_gradient_steps_floor():
    result = LoopStateMachine._clamp_rl_adjustments({"gradient_steps": 0}, "rl")
    assert result["gradient_steps"] == 1


def test_clamp_train_freq_bounds():
    assert LoopStateMachine._clamp_rl_adjustments({"train_freq": 0}, "rl")["train_freq"] == 1
    assert LoopStateMachine._clamp_rl_adjustments({"train_freq": 999}, "rl")["train_freq"] == 16


def test_clamp_learning_starts_bounds():
    assert LoopStateMachine._clamp_rl_adjustments({"learning_starts": 0}, "rl")["learning_starts"] == 1000
    assert LoopStateMachine._clamp_rl_adjustments({"learning_starts": 999999}, "rl")["learning_starts"] == 50000


def test_clamp_target_update_interval_bounds():
    assert LoopStateMachine._clamp_rl_adjustments({"target_update_interval": 1}, "rl")["target_update_interval"] == 100
    assert LoopStateMachine._clamp_rl_adjustments({"target_update_interval": 99999}, "rl")["target_update_interval"] == 5000


def test_clamp_tau_bounds():
    assert LoopStateMachine._clamp_rl_adjustments({"tau": 0.0}, "rl")["tau"] == 0.005
    assert LoopStateMachine._clamp_rl_adjustments({"tau": 2.0}, "rl")["tau"] == 1.0


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


# ── _should_force_actor_critic ────────────────────────────────────────────────
# Real incident: this override was previously unconditional — every Tetris-v0
# mission ever trained the custom Actor-Critic model regardless of what
# algorithm was actually requested, with zero indication anywhere (mission
# goal, UI, crystallized recipe) that the choice wasn't honored. Confirmed
# live against two completed missions ("...DQN agent..." and "...PPO
# agent...") whose generated train.py scripts both imported ActorCriticNet.

_force_ac = LoopStateMachine._should_force_actor_critic


def test_force_actor_critic_when_no_algorithm_requested():
    """Default case: goal doesn't name a specific algorithm — use the
    empirically stronger Actor-Critic trainer."""
    assert _force_ac("Tetris-v0", "PPO", "", "Train a Tetris-v0 agent to clear 200 lines")


def test_does_not_force_when_dqn_explicitly_requested():
    """The exact real-incident reproduction."""
    assert not _force_ac("Tetris-v0", "DQN", "", "Train a Tetris-v0 DQN agent to achieve 200 lines_cleared")


def test_does_not_force_when_ppo_explicitly_requested():
    assert not _force_ac("Tetris-v0", "PPO", "", "Train a Tetris-v0 PPO agent to achieve 200 lines_cleared")


def test_does_not_force_when_a2c_explicitly_requested():
    assert not _force_ac("Tetris-v0", "A2C", "", "Train a Tetris-v0 A2C agent to achieve 200 lines_cleared")


def test_does_not_force_for_non_tetris_env():
    assert not _force_ac("Snake-v0", "PPO", "", "Train a Snake-v0 agent")


def test_does_not_force_when_trainer_type_already_set():
    """Recipe-locked trainer_type (e.g. tetris_actor_critic_v1.yaml) is never overridden."""
    assert not _force_ac("Tetris-v0", "PPO", "actor_critic", "Train a Tetris-v0 agent")


def test_does_not_force_when_algorithm_field_empty():
    """No algorithm in the plan at all — nothing to honor, use the default."""
    assert _force_ac("Tetris-v0", "", "", "Train a Tetris-v0 agent to clear lines")


# ── _lookahead_trainer_type_for ─────────────────────────────────────────────────
# Real incident: vanilla SB3 DQN/PPO/A2C structurally cannot compete on
# Tetris-v0 (confirmed live: 130+ DQN pivots, dozens of PPO/A2C pivots, all
# plateaued at lines_cleared≈0-1, vs. Actor-Critic hitting 394 in 3
# iterations). An explicit DQN/PPO/A2C request for Tetris-v0 must still be
# honored (the algorithm choice is preserved) but routed to the matching
# lookahead-augmented custom trainer instead of hopeless vanilla SB3.

_lookahead_for = LoopStateMachine._lookahead_trainer_type_for


def test_lookahead_routes_explicit_dqn_request():
    assert _lookahead_for("Tetris-v0", "DQN", "", "Train a Tetris-v0 DQN agent to clear 200 lines") == "lookahead_dqn"


def test_lookahead_routes_explicit_ppo_request():
    assert _lookahead_for("Tetris-v0", "PPO", "", "Train a Tetris-v0 PPO agent to clear 200 lines") == "lookahead_ppo"


def test_lookahead_routes_explicit_a2c_request():
    assert _lookahead_for("Tetris-v0", "A2C", "", "Train a Tetris-v0 A2C agent to clear 200 lines") == "lookahead_a2c"


def test_lookahead_none_when_algorithm_not_named_in_goal():
    """Algorithm set in the plan but not explicitly requested by the user —
    nothing to honor, no lookahead routing (falls through to the actor_critic
    default instead)."""
    assert _lookahead_for("Tetris-v0", "PPO", "", "Train a Tetris-v0 agent to clear lines") is None


def test_lookahead_none_for_non_tetris_env():
    assert _lookahead_for("Snake-v0", "DQN", "", "Train a Snake-v0 DQN agent") is None


def test_lookahead_none_when_trainer_type_already_set():
    assert _lookahead_for("Tetris-v0", "DQN", "actor_critic", "Train a Tetris-v0 DQN agent") is None


def test_lookahead_none_for_unrecognized_algorithm():
    """Only DQN/PPO/A2C have lookahead variants — an unrecognized/unsupported
    algorithm name falls through untouched rather than crashing."""
    assert _lookahead_for("Tetris-v0", "SAC", "", "Train a Tetris-v0 SAC agent") is None


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
    """DB value is valid for a custom metric only when best_metric_iteration is set (eval ran)."""
    sm = _make_sm()
    mission = _MockMission(
        target_metric={"lines_cleared": 20},
        best_metric_value="5.0",
        best_metric_iteration=0,
    )
    result = sm._load_persisted_best("fake-id", mission)
    assert result == 5.0


def test_load_persisted_best_ignores_db_for_custom_metric_without_eval():
    """DB value is discarded when best_metric_iteration is None (no eval has run yet)."""
    sm = _make_sm()
    mission = _MockMission(
        target_metric={"lines_cleared": 20},
        best_metric_value="1.32",  # contaminated by training telemetry scan
        best_metric_iteration=None,
    )
    result = sm._load_persisted_best("fake-id", mission)
    assert result is None


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


def test_load_persisted_best_uses_telemetry_for_mean_reward():
    """Telemetry scan is used for mean_reward targets (training metric name matches goal)."""
    sm = _make_sm()
    sm._read_telemetry_metrics = lambda mission_id, offset: {"mean_reward": 150.0}
    mission = _MockMission(
        target_metric={"mean_reward": 200},
        best_metric_value=None,
    )
    result = sm._load_persisted_best("fake-id", mission)
    assert result == 150.0


def test_load_persisted_best_ignores_telemetry_for_custom_metric():
    """Telemetry scan is skipped for custom goal metrics to avoid training-signal contamination."""
    sm = _make_sm()
    sm._read_telemetry_metrics = lambda mission_id, offset: {"lines_cleared": 3.0}
    mission = _MockMission(
        target_metric={"lines_cleared": 20},
        best_metric_value=None,
    )
    result = sm._load_persisted_best("fake-id", mission)
    assert result is None


# ── _wait_for_sandbox error detection ─────────────────────────────────────────

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from sqlalchemy import select
from backend.models.approval import ApprovalGate, ApprovalStatus, GateType


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


def test_wait_sandbox_torn_utf8_at_seek_offset_does_not_crash(tmp_path):
    """A seek() landing mid-multibyte-character (e.g. tqdm's block-char progress
    bar, \\xe2\\x96\\x88) must not raise UnicodeDecodeError and kill an otherwise
    clean/successful run — confirmed via a real incident where this crashed a
    DPO mission after all 3 training epochs had already completed."""
    sm = _make_sm()
    log_path = tmp_path / "sandbox.log"
    # Byte 0x96 is the *middle* byte of the 3-byte UTF-8 sequence for '█'
    # (\xe2\x96\x88) — write raw bytes and seek to right before it so the read
    # starts on a continuation byte, exactly like the real incident.
    raw = "Training complete\n".encode() + b"\xe2\x96\x88\xe2\x96\x88" + b"\n"
    log_path.write_bytes(raw)
    sandbox = MagicMock()
    sandbox.is_alive.return_value = False
    sandbox.get_log_path.return_value = str(log_path)
    sm._sandbox = sandbox

    result = _run(sm._wait_for_sandbox("mid", log_offset=len("Training complete\n".encode()) + 1))
    assert result is None  # no Traceback/Error — must not raise


# ── propose_pivot current plan context ───────────────────────────────────────

def test_propose_pivot_passes_current_plan_context(monkeypatch):
    """propose_pivot receives current policy_kwargs, hyperparameters, env_kwargs,
    and best_policy_kwargs so the LLM can avoid re-proposing already-applied changes
    and knows which architecture performed best."""
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
        best_policy_kwargs=None,
        best_metric_value=None,
        best_metric_iteration=None,
    ):
        captured["policy_kwargs"] = current_policy_kwargs
        captured["hyperparameters"] = current_hyperparameters
        captured["env_kwargs"] = current_env_kwargs
        captured["best_policy_kwargs"] = best_policy_kwargs
        captured["best_metric_value"] = best_metric_value
        captured["best_metric_iteration"] = best_metric_iteration
        return {"reason": "test", "adjustments": {}}

    from backend.agent.lead_agent import LeadAgent
    monkeypatch.setattr(LeadAgent, "propose_pivot", _fake_propose)

    from backend.loop.pivots import PivotEngine
    engine = PivotEngine({"food_eaten": 30})
    best_arch = {"net_arch": [256, 256, 128]}
    # iter 0: best so far with the arch we want to track
    engine.record(0, {"food_eaten": 16.0}, policy_kwargs=best_arch)
    # iters 1–3: plateau at 6 to trigger needs_pivot
    for i in range(1, 4):
        engine.record(i, {"food_eaten": 6.0}, policy_kwargs={"net_arch": [256, 256]})

    assert engine.needs_pivot()
    assert engine.best_policy_kwargs() == best_arch

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
            {"food_eaten": 6.0},
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
            best_policy_kwargs=engine.best_policy_kwargs(),
            best_metric_value=engine.best_metric_value(),
            best_metric_iteration=engine.best_metric_iteration(),
        )

    asyncio.get_event_loop().run_until_complete(_run_pivot())

    assert captured["policy_kwargs"] == {"net_arch": [256, 256]}
    assert captured["hyperparameters"] == {"learning_rate": 3e-4, "n_steps": 2048}
    assert captured["env_kwargs"] == {"food_reward": 15.0, "death_penalty": -2.0}
    assert captured["best_policy_kwargs"] == {"net_arch": [256, 256, 128]}
    assert captured["best_metric_value"] == 16.0
    assert captured["best_metric_iteration"] == 0


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


# ── arch oscillation detection ────────────────────────────────────────────────

def _arch_changed(pivot_pky, current_pky, recent_arches):
    """Mirror the oscillation-aware arch_changed logic from state_machine._step."""
    oscillation = bool(pivot_pky and pivot_pky in recent_arches)
    return bool(pivot_pky and pivot_pky != current_pky and not oscillation), oscillation


def test_arch_oscillation_suppressed_when_in_recent_history():
    """Proposing an arch that was used 1-2 pivots ago must be suppressed."""
    current = {"net_arch": [256, 256, 128]}
    recent = [{"net_arch": [256, 256]}]  # LLM oscillating back to [256,256]
    proposed = {"net_arch": [256, 256]}
    changed, oscillation = _arch_changed(proposed, current, recent)
    assert not changed
    assert oscillation


def test_arch_not_suppressed_when_not_in_recent_history():
    """A genuinely new arch (not in recent_arches) proceeds normally."""
    current = {"net_arch": [256, 256]}
    recent = [{"net_arch": [64, 64]}]
    proposed = {"net_arch": [512, 512]}
    changed, oscillation = _arch_changed(proposed, current, recent)
    assert changed
    assert not oscillation


def test_arch_oscillation_window_is_three():
    """recent_arches is capped at 3 — oldest entry falls off."""
    recent = [{"net_arch": [64, 64]}, {"net_arch": [128, 128]}, {"net_arch": [256, 256]}]
    # [64,64] is in the window — still blocked
    changed, osc = _arch_changed({"net_arch": [64, 64]}, {"net_arch": [512, 512]}, recent)
    assert not changed and osc
    # After sliding window drops [64,64] (4th entry added), it would no longer block
    new_recent = (recent + [{"net_arch": [512, 512]}])[-3:]
    assert {"net_arch": [64, 64]} not in new_recent


def test_arch_unchanged_when_same_as_current_regardless_of_history():
    """Proposing the same arch as current is already filtered by != check, not oscillation."""
    current = {"net_arch": [256, 256]}
    changed, oscillation = _arch_changed(current, current, [])
    assert not changed
    assert not oscillation  # not oscillation — just a no-op


def test_recent_arches_updated_on_arch_change():
    """When arch changes, the outgoing arch is prepended to recent_arches (capped at 3)."""
    plan = {
        "hyperparameters": {"policy_kwargs": {"net_arch": [256, 256]}},
        "recent_arches": [{"net_arch": [64, 64]}],
    }
    outgoing = plan["hyperparameters"]["policy_kwargs"]
    _recent = list(plan.get("recent_arches", []))
    if outgoing not in _recent:
        _recent.append(outgoing)
    plan["recent_arches"] = _recent[-3:]
    plan["hyperparameters"]["policy_kwargs"] = {"net_arch": [512, 512]}

    assert {"net_arch": [256, 256]} in plan["recent_arches"]
    assert {"net_arch": [64, 64]} in plan["recent_arches"]
    assert plan["hyperparameters"]["policy_kwargs"] == {"net_arch": [512, 512]}


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
    """Values already within range pass through unmodified."""
    result = LoopStateMachine._clamp_env_kwargs({"food_reward": 25.0, "survival_bonus": 0.05})
    assert result == {"food_reward": 25.0, "survival_bonus": 0.05}


def test_clamp_env_kwargs_death_penalty_floor():
    """Real incident: death_penalty drifted from -10 to -3 (weakened 3.3x) with
    zero resistance, plausibly teaching the agent to prioritize survival over
    food-seeking since dying barely hurt anymore."""
    result = LoopStateMachine._clamp_env_kwargs({"death_penalty": -0.5})
    assert result["death_penalty"] == -1.0


def test_clamp_env_kwargs_death_penalty_ceiling():
    result = LoopStateMachine._clamp_env_kwargs({"death_penalty": -999.0})
    assert result["death_penalty"] == -20.0


def test_clamp_env_kwargs_food_reward_bounds():
    assert LoopStateMachine._clamp_env_kwargs({"food_reward": 0.5})["food_reward"] == 5.0
    assert LoopStateMachine._clamp_env_kwargs({"food_reward": 9999.0})["food_reward"] == 50.0


def test_clamp_env_kwargs_survival_bonus_ceiling():
    """Real incident: survival_bonus drifted from 0.01 to 0.05 (5x) with zero
    resistance, combined with the weakened death_penalty above."""
    result = LoopStateMachine._clamp_env_kwargs({"survival_bonus": 999.0})
    assert result["survival_bonus"] == 0.2


def test_clamp_env_kwargs_survival_bonus_floor():
    result = LoopStateMachine._clamp_env_kwargs({"survival_bonus": -1.0})
    assert result["survival_bonus"] == 0.0


# ── _clamp_net_arch ──────────────────────────────────────────────────────────

def test_clamp_net_arch_none_passthrough():
    assert LoopStateMachine._clamp_net_arch(None) is None


def test_clamp_net_arch_no_net_arch_key_passthrough():
    pky = {"foo": "bar"}
    assert LoopStateMachine._clamp_net_arch(pky) == pky


def test_clamp_net_arch_list_width_ceiling():
    """Real incident class: gradient_steps drifted to 1000 with no bound,
    burning 10+ hours of CPU on one iteration — an oversized net_arch is the
    same failure mode (OOM/catastrophic slowdown) for architecture."""
    result = LoopStateMachine._clamp_net_arch({"net_arch": [99999, 99999]})
    assert result["net_arch"] == [1024, 1024]


def test_clamp_net_arch_list_width_floor():
    result = LoopStateMachine._clamp_net_arch({"net_arch": [1, 1]})
    assert result["net_arch"] == [8, 8]


def test_clamp_net_arch_list_depth_capped():
    result = LoopStateMachine._clamp_net_arch({"net_arch": [64] * 20})
    assert len(result["net_arch"]) == 6


def test_clamp_net_arch_list_valid_unchanged():
    result = LoopStateMachine._clamp_net_arch({"net_arch": [256, 256]})
    assert result["net_arch"] == [256, 256]


def test_clamp_net_arch_dict_n_layers_and_layer_size_bounded():
    """DPO/GRPO shared-net dict shape: {'type': ..., 'n_layers': N, 'layer_size': M}."""
    result = LoopStateMachine._clamp_net_arch(
        {"net_arch": {"type": "shared", "n_layers": 999, "layer_size": 999999}}
    )
    assert result["net_arch"]["n_layers"] == 6
    assert result["net_arch"]["layer_size"] == 2048
    assert result["net_arch"]["type"] == "shared"  # sibling keys preserved


# Real incident: a pivot proposed net_arch as a list of dicts (confusing the
# RL list convention with the DPO/GRPO dict convention). The old code took
# the list branch, int(dict) raised TypeError, and a bare `except: pass` left
# the malformed list completely unclamped — it reached SB3's create_mlp() as
# net_arch[0] being a dict and crashed every relaunch in a self-healing loop
# with no way out (mission fa203f6b).

def test_clamp_net_arch_list_of_dicts_dropped():
    result = LoopStateMachine._clamp_net_arch(
        {"net_arch": [{"n_layers": 3, "n_neurons": 256}, {"n_layers": 2, "n_neurons": 128}]}
    )
    assert "net_arch" not in result


def test_clamp_net_arch_list_of_dicts_preserves_sibling_keys():
    result = LoopStateMachine._clamp_net_arch(
        {"net_arch": [{"n_layers": 3}], "activation_fn": "relu"}
    )
    assert "net_arch" not in result
    assert result["activation_fn"] == "relu"


def test_clamp_net_arch_list_with_one_bad_element_dropped():
    """Even a single non-numeric element invalidates the whole list."""
    result = LoopStateMachine._clamp_net_arch({"net_arch": [256, {"bad": 1}, 128]})
    assert "net_arch" not in result


def test_clamp_net_arch_dict_with_bad_n_layers_dropped():
    result = LoopStateMachine._clamp_net_arch(
        {"net_arch": {"type": "shared", "n_layers": {"nested": "garbage"}, "layer_size": 256}}
    )
    assert "net_arch" not in result


def test_clamp_net_arch_scalar_dropped():
    """Not a list or dict at all (e.g. a bare string) — same reject rule."""
    result = LoopStateMachine._clamp_net_arch({"net_arch": "256,256"})
    assert "net_arch" not in result


# ── _pick_untried_net_arch ───────────────────────────────────────────────────

def test_pick_untried_net_arch_skips_all_tried():
    """Real incident: the LLM proposed the identical already-abandoned
    architecture 11 pivots in a row despite an explicit instruction not to —
    this is the deterministic fallback that breaks the stall."""
    tried = [{"net_arch": [128]}, {"net_arch": [512]}, {"net_arch": [128, 128]}]
    result = LoopStateMachine._pick_untried_net_arch(tried, current_policy_kwargs=None)
    assert result is not None
    assert result not in ([128], [512], [128, 128])


def test_pick_untried_net_arch_also_excludes_current():
    all_but_last = LoopStateMachine._CANDIDATE_NET_ARCHES[:-1]
    tried = [{"net_arch": a} for a in all_but_last]
    result = LoopStateMachine._pick_untried_net_arch(
        tried, current_policy_kwargs={"net_arch": LoopStateMachine._CANDIDATE_NET_ARCHES[-1]}
    )
    assert result is None  # every candidate exhausted


def test_pick_untried_net_arch_ignores_dict_shaped_history():
    """DPO/GRPO dict-shaped net_arch entries must not crash the RL-only picker."""
    tried = [{"net_arch": {"type": "shared", "n_layers": 2, "layer_size": 300}}]
    result = LoopStateMachine._pick_untried_net_arch(tried, current_policy_kwargs=None)
    assert result == LoopStateMachine._CANDIDATE_NET_ARCHES[0]


# ── _save_iteration_checkpoint ───────────────────────────────────────────────

def _make_checkpoint_dir(tmp_path, mission_id="test-mission"):
    ckpt_dir = tmp_path / "missions" / mission_id / "checkpoints"
    ckpt_dir.mkdir(parents=True)
    best_zip = ckpt_dir / "best_model.zip"
    best_zip.write_bytes(b"fake-model-weights")
    return ckpt_dir


def test_save_iteration_checkpoint_creates_iter_subdir(tmp_path, monkeypatch):
    monkeypatch.setattr("backend.loop.state_machine.settings.data_path", str(tmp_path))
    ckpt_dir = _make_checkpoint_dir(tmp_path)
    sm = LoopStateMachine.__new__(LoopStateMachine)
    sm._save_iteration_checkpoint("test-mission", 47)
    assert (ckpt_dir / "iter" / "checkpoint_iter_47.zip").exists()


def test_save_iteration_checkpoint_content_matches_best_model(tmp_path, monkeypatch):
    monkeypatch.setattr("backend.loop.state_machine.settings.data_path", str(tmp_path))
    ckpt_dir = _make_checkpoint_dir(tmp_path)
    sm = LoopStateMachine.__new__(LoopStateMachine)
    sm._save_iteration_checkpoint("test-mission", 47)
    content = (ckpt_dir / "iter" / "checkpoint_iter_47.zip").read_bytes()
    assert content == b"fake-model-weights"


def test_save_iteration_checkpoint_prunes_beyond_window(tmp_path, monkeypatch):
    monkeypatch.setattr("backend.loop.state_machine.settings.data_path", str(tmp_path))
    from backend.loop.state_machine import ITER_CHECKPOINT_WINDOW
    ckpt_dir = _make_checkpoint_dir(tmp_path)
    iter_dir = ckpt_dir / "iter"
    iter_dir.mkdir()
    # Pre-create ITER_CHECKPOINT_WINDOW existing checkpoints (iters 0..9)
    for i in range(ITER_CHECKPOINT_WINDOW):
        (iter_dir / f"checkpoint_iter_{i}.zip").write_bytes(b"old")
    sm = LoopStateMachine.__new__(LoopStateMachine)
    sm._save_iteration_checkpoint("test-mission", ITER_CHECKPOINT_WINDOW)
    remaining = sorted(iter_dir.glob("checkpoint_iter_*.zip"),
                       key=lambda p: int(p.stem.replace("checkpoint_iter_", "")))
    # Should have exactly WINDOW files; oldest (iter 0) pruned
    assert len(remaining) == ITER_CHECKPOINT_WINDOW
    assert not (iter_dir / "checkpoint_iter_0.zip").exists()
    assert (iter_dir / f"checkpoint_iter_{ITER_CHECKPOINT_WINDOW}.zip").exists()


def test_save_iteration_checkpoint_noop_when_best_model_missing(tmp_path, monkeypatch):
    monkeypatch.setattr("backend.loop.state_machine.settings.data_path", str(tmp_path))
    # No best_model.zip created
    (tmp_path / "missions" / "test-mission" / "checkpoints").mkdir(parents=True)
    sm = LoopStateMachine.__new__(LoopStateMachine)
    sm._save_iteration_checkpoint("test-mission", 5)
    iter_dir = tmp_path / "missions" / "test-mission" / "checkpoints" / "iter"
    assert not iter_dir.exists()


def test_save_iteration_checkpoint_no_pre_pivot_backup_created(tmp_path, monkeypatch):
    """best_model_pre_pivot.zip must never be created — iter window replaces it."""
    monkeypatch.setattr("backend.loop.state_machine.settings.data_path", str(tmp_path))
    ckpt_dir = _make_checkpoint_dir(tmp_path)
    sm = LoopStateMachine.__new__(LoopStateMachine)
    sm._save_iteration_checkpoint("test-mission", 5)
    assert not (ckpt_dir / "best_model_pre_pivot.zip").exists()


def test_normalize_pivot_promotes_nested_policy_kwargs():
    """policy_kwargs nested inside adjustments must be promoted to top-level."""
    pivot = {
        "reason": "test",
        "adjustments": {
            "policy_kwargs": {"net_arch": [512, 512]},
            "learning_rate": 0.0001,
        },
    }
    result = LoopStateMachine._normalize_pivot(pivot)
    assert result["policy_kwargs"] == {"net_arch": [512, 512]}
    assert "policy_kwargs" not in result["adjustments"]
    assert result["adjustments"]["learning_rate"] == 0.0001


def test_normalize_pivot_does_not_overwrite_existing_top_level_policy_kwargs():
    """If top-level policy_kwargs already exists, nested one is discarded."""
    pivot = {
        "reason": "test",
        "policy_kwargs": {"net_arch": [256, 256]},
        "adjustments": {"policy_kwargs": {"net_arch": [512, 512]}},
    }
    result = LoopStateMachine._normalize_pivot(pivot)
    assert result["policy_kwargs"] == {"net_arch": [256, 256]}


def test_normalize_pivot_unwraps_doubly_nested_net_arch():
    """LLM sometimes returns policy_kwargs: {net_arch: {net_arch: [...]}} — must flatten."""
    pivot = {
        "reason": "test",
        "policy_kwargs": {"net_arch": {"net_arch": [256, 256, 128]}},
        "adjustments": {},
    }
    result = LoopStateMachine._normalize_pivot(pivot)
    assert result["policy_kwargs"] == {"net_arch": [256, 256, 128]}


def test_normalize_pivot_leaves_correct_net_arch_unchanged():
    """Correctly-formatted policy_kwargs must not be modified."""
    pivot = {
        "reason": "test",
        "policy_kwargs": {"net_arch": [256, 256]},
        "adjustments": {},
    }
    result = LoopStateMachine._normalize_pivot(pivot)
    assert result["policy_kwargs"] == {"net_arch": [256, 256]}


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


# ── _run_goal_metric_eval env_kwargs passthrough ──────────────────────────────

def test_run_goal_metric_eval_passes_env_kwargs_to_gym_make(tmp_path):
    """eval must pass env_kwargs from train_config.json to gym.make so obs_type=features models work."""
    import json as _json
    import sys
    import types
    from unittest.mock import patch, MagicMock

    mission_id = "test-mission-eval"
    checkpoint_dir = tmp_path / "missions" / mission_id / "checkpoints"
    checkpoint_dir.mkdir(parents=True)
    (checkpoint_dir / "best_model.zip").write_text("fake")
    (checkpoint_dir / "train_config.json").write_text(
        _json.dumps({"env_id": "Snake-v0", "env_kwargs": {"obs_type": "features", "max_steps": 2000}})
    )

    sm = LoopStateMachine.__new__(LoopStateMachine)
    plan = {"env_id": "Snake-v0", "algorithm": "PPO", "hyperparameters": {"eval_episodes": 1}}

    mock_model = MagicMock()
    mock_model.predict.return_value = (0, None)
    mock_env = MagicMock()
    mock_env.reset.return_value = ([0.0] * 25, {})
    mock_env.step.return_value = ([0.0] * 25, 0.0, True, False, {"food_eaten": 7})

    mock_ppo_cls = MagicMock()
    mock_ppo_cls.load.return_value = mock_model

    make_calls = []

    def fake_gym_make(env_id, **kwargs):
        make_calls.append((env_id, kwargs))
        return mock_env

    fake_gym = types.ModuleType("gymnasium")
    fake_gym.make = fake_gym_make
    fake_sb3 = types.ModuleType("stable_baselines3")
    fake_sb3.PPO = mock_ppo_cls
    fake_sb3.SAC = MagicMock()
    fake_sb3.A2C = MagicMock()
    fake_sb3.DQN = MagicMock()
    fake_sb3.TD3 = MagicMock()
    fake_snake = types.ModuleType("envs.snake_env")
    fake_snake.register = MagicMock()

    with patch("backend.loop.state_machine.settings") as mock_settings, \
         patch.dict(sys.modules, {
             "gymnasium": fake_gym,
             "stable_baselines3": fake_sb3,
             "envs.snake_env": fake_snake,
         }):
        mock_settings.data_path = str(tmp_path)
        sm._run_goal_metric_eval(mission_id, plan, "food_eaten")

    assert any(kw.get("obs_type") == "features" for _, kw in make_calls), \
        f"obs_type=features not passed to gym.make; calls={make_calls}"


# ── _request_approval: auto-approve vs manual-approve flag ─────────────────────

@pytest.mark.usefixtures("patch_db")
def test_request_approval_flags_auto_approve(db_session, test_mission):
    """Inline classifier resolution must return is_auto=True, so the caller
    can display 'auto-approve' rather than a generic message that looks the
    same as a manual click."""
    sm = _make_sm()
    sm._model_manager = MagicMock()
    sm._model_manager._providers = {"code": MagicMock()}

    async def _fake_try_auto_approve(gate_id, code_provider, db=None):
        result = MagicMock()
        result.action = "approved"
        result.classifier = "static"
        return result

    with patch("backend.services.auto_approver.try_auto_approve", new=_fake_try_auto_approve):
        approved, is_auto = asyncio.get_event_loop().run_until_complete(
            sm._request_approval(test_mission.id, GateType.EXECUTE_CODE, payload={"script_path": "/x.py"})
        )

    assert approved is True
    assert is_auto is True


@pytest.mark.usefixtures("patch_db")
def test_request_approval_guided_mode_skips_inline_auto_approve(db_session, test_mission):
    """allow_inline_auto_approve=False (guided mode) must never even attempt
    the classifier shortcut — every gate requires an explicit decision, not
    just one the classifier happens to bless. Distinct from supervised mode,
    where the classifier is tried first and only falls back to manual on a
    'blocked' verdict."""
    sm = _make_sm()
    sm._model_manager = MagicMock()
    sm._model_manager._providers = {"code": MagicMock()}

    auto_approve_called = False

    async def _fake_try_auto_approve(gate_id, code_provider, db=None):
        nonlocal auto_approve_called
        auto_approve_called = True
        result = MagicMock()
        result.action = "approved"
        return result

    async def _resolve_as_manually_approved(*_args, **_kwargs):
        from backend.loop.state_machine import AsyncSessionLocal as PatchedSessionLocal
        async with PatchedSessionLocal() as session:
            result = await session.execute(
                select(ApprovalGate).where(ApprovalGate.mission_id == test_mission.id)
            )
            gate = result.scalars().first()
            gate.status = ApprovalStatus.APPROVED.value
            gate.reviewer_note = "manually reviewed and approved"
            await session.commit()

    async def _run():
        with patch("backend.services.auto_approver.try_auto_approve", new=_fake_try_auto_approve), \
             patch("backend.loop.state_machine.asyncio.sleep", new=AsyncMock(side_effect=_resolve_as_manually_approved)):
            return await sm._request_approval(
                test_mission.id, GateType.EXECUTE_CODE, payload={"script_path": "/x.py"},
                allow_inline_auto_approve=False,
            )

    approved, is_auto = asyncio.get_event_loop().run_until_complete(_run())

    assert auto_approve_called is False
    assert approved is True
    assert is_auto is False


@pytest.mark.usefixtures("patch_db")
def test_request_approval_flags_manual_approve(db_session, test_mission):
    """A gate resolved by a human clicking Approve (reviewer_note has no
    '[auto-approved]' prefix) must return is_auto=False. Deterministic:
    asyncio.sleep's mock itself performs the DB write as a side effect on its
    first call, instead of racing a background task against the poll loop."""
    sm = _make_sm()
    sm._model_manager = MagicMock()
    sm._model_manager._providers = {"code": MagicMock()}

    async def _fake_try_auto_approve(gate_id, code_provider, db=None):
        result = MagicMock()
        result.action = "blocked"  # classifier declines -> falls through to manual poll
        return result

    async def _resolve_as_manually_approved(*_args, **_kwargs):
        from backend.loop.state_machine import AsyncSessionLocal as PatchedSessionLocal
        async with PatchedSessionLocal() as session:
            result = await session.execute(
                select(ApprovalGate).where(ApprovalGate.mission_id == test_mission.id)
            )
            gate = result.scalars().first()
            gate.status = ApprovalStatus.APPROVED.value
            gate.reviewer_note = "looks fine to me"
            await session.commit()

    async def _run():
        with patch("backend.services.auto_approver.try_auto_approve", new=_fake_try_auto_approve), \
             patch("backend.loop.state_machine.asyncio.sleep", new=AsyncMock(side_effect=_resolve_as_manually_approved)):
            return await sm._request_approval(
                test_mission.id, GateType.EXECUTE_CODE, payload={"script_path": "/x.py"}
            )

    approved, is_auto = asyncio.get_event_loop().run_until_complete(_run())

    assert approved is True
    assert is_auto is False


@pytest.mark.usefixtures("patch_db")
def test_request_approval_rejected_returns_none_flag(db_session, test_mission):
    sm = _make_sm()
    sm._model_manager = MagicMock()
    sm._model_manager._providers = {"code": MagicMock()}

    async def _fake_try_auto_approve(gate_id, code_provider, db=None):
        result = MagicMock()
        result.action = "blocked"
        return result

    async def _resolve_as_rejected(*_args, **_kwargs):
        from backend.loop.state_machine import AsyncSessionLocal as PatchedSessionLocal
        async with PatchedSessionLocal() as session:
            result = await session.execute(
                select(ApprovalGate).where(ApprovalGate.mission_id == test_mission.id)
            )
            gate = result.scalars().first()
            gate.status = ApprovalStatus.REJECTED.value
            await session.commit()

    async def _run():
        with patch("backend.services.auto_approver.try_auto_approve", new=_fake_try_auto_approve), \
             patch("backend.loop.state_machine.asyncio.sleep", new=AsyncMock(side_effect=_resolve_as_rejected)):
            return await sm._request_approval(
                test_mission.id, GateType.EXECUTE_CODE, payload={"script_path": "/x.py"}
            )

    approved, is_auto = asyncio.get_event_loop().run_until_complete(_run())

    assert approved is False
    assert is_auto is None
