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
