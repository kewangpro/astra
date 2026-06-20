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


# ── Generic "achieve {metric} of {value}" catch-all ──────────────────────────

def test_parse_lines_cleared():
    result = _parse_target_metric(
        "Train a Tetris-v0 PPO agent to achieve lines_cleared of 20"
    )
    assert result == {"lines_cleared": 20.0}


def test_parse_generic_metric_name():
    result = _parse_target_metric("achieve f1_score of 0.85")
    assert result == {"f1_score": 0.85}


def test_parse_generic_case_insensitive():
    result = _parse_target_metric("Achieve Mean_Reward of 100")
    assert result == {"mean_reward": 100.0}


def test_parse_generic_integer_value():
    result = _parse_target_metric("achieve episodes of 500")
    assert result == {"episodes": 500.0}


def test_parse_generic_does_not_match_without_achieve():
    # Pattern requires "achieve" keyword to avoid greedy false matches
    result = _parse_target_metric("lines_cleared of 20 is the goal")
    assert result == {}


def test_parse_multi_word_metric_food_eaten():
    result = _parse_target_metric(
        "Train a Snake-v0 PPO agent to achieve food eaten of 30"
    )
    assert result == {"food_eaten": 30.0}


def test_parse_multi_word_metric_spaces_to_underscores():
    result = _parse_target_metric("achieve avg episode length of 200")
    assert result == {"avg_episode_length": 200.0}


def test_parse_multi_word_metric_case_insensitive():
    result = _parse_target_metric("Achieve Food Eaten of 15")
    assert result == {"food_eaten": 15.0}
