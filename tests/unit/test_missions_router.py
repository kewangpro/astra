"""Unit tests for missions router helpers (backend/routers/missions.py)."""
from __future__ import annotations

import pytest
from backend.routers.missions import _parse_target_metric


# ── RL reward patterns ────────────────────────────────────────────────────────

def test_parse_reward_of_integer():
    result = _parse_target_metric("Train a CartPole-v1 PPO agent to achieve mean_reward of 475")
    assert result == {"mean_reward": 475.0}


def test_parse_reward_of_float():
    result = _parse_target_metric("achieve reward of 475.5")
    assert result == {"mean_reward": 475.5}


def test_parse_reward_case_insensitive():
    result = _parse_target_metric("Achieve REWARD OF 200")
    assert result == {"mean_reward": 200.0}


# ── ML accuracy patterns ──────────────────────────────────────────────────────

def test_parse_accuracy_percentage():
    result = _parse_target_metric("Train an iris classifier with 95% accuracy")
    assert result == {"accuracy": 0.95}


def test_parse_accuracy_of_fraction():
    result = _parse_target_metric("Train model to achieve accuracy of 0.92")
    assert result == {"accuracy": 0.92}


def test_parse_accuracy_of_integer_converts_to_fraction():
    result = _parse_target_metric("achieve accuracy of 92")
    assert result == {"accuracy": 0.92}


def test_parse_accuracy_case_insensitive():
    result = _parse_target_metric("90% ACCURACY")
    assert result == {"accuracy": 0.90}


# ── Loss patterns ─────────────────────────────────────────────────────────────

def test_parse_eval_loss():
    result = _parse_target_metric("fine-tune until eval_loss <= 0.5")
    assert result == {"eval_loss": 0.5}


def test_parse_loss_of():
    result = _parse_target_metric("reduce loss of 0.3")
    assert result == {"eval_loss": 0.3}


# ── No match ──────────────────────────────────────────────────────────────────

def test_parse_unrecognized_goal():
    result = _parse_target_metric("Train a model to do something cool")
    assert result == {}


def test_parse_empty_string():
    result = _parse_target_metric("")
    assert result == {}


def test_parse_no_numeric_target():
    result = _parse_target_metric("achieve high accuracy")
    assert result == {}
