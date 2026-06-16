"""Unit tests for LoopStateMachine static helpers."""
from __future__ import annotations

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
    assert result["n_steps"] == 512


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
