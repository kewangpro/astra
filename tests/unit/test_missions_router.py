"""Unit tests for missions router helpers (backend/routers/missions.py)."""
from __future__ import annotations

import pytest
from fastapi import HTTPException

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


def test_parse_number_first_food_eaten_in_game():
    result = _parse_target_metric(
        "Train a Snake-v0 PPO agent to achieve 20 food eaten in one game"
    )
    assert result == {"food_eaten": 20.0}


def test_parse_number_first_lines_cleared():
    result = _parse_target_metric(
        "Train a Tetris PPO agent to achieve 30 lines cleared per episode"
    )
    assert result == {"lines_cleared": 30.0}


def test_parse_number_first_no_trailing_clause():
    result = _parse_target_metric("achieve 50 food eaten")
    assert result == {"food_eaten": 50.0}


def test_parse_number_first_does_not_clobber_metric_of_value():
    # "achieve food eaten of 30" should still use the existing pattern (metric-first)
    result = _parse_target_metric("achieve food eaten of 30")
    assert result == {"food_eaten": 30.0}


# ── pass_rate and reach patterns ──────────────────────────────────────────────

def test_parse_rejection_sampling_reach_pass_rate():
    result = _parse_target_metric("Rejection-sampling fine-tuning to reach 90% pass rate")
    assert result == {"pass_rate": 0.9}


def test_parse_pass_rate_percentage():
    assert _parse_target_metric("achieve 92% pass_rate") == {"pass_rate": 0.92}
    assert _parse_target_metric("reach 95% pass rate") == {"pass_rate": 0.95}


def test_parse_pass_rate_of():
    assert _parse_target_metric("pass rate of 0.88") == {"pass_rate": 0.88}
    assert _parse_target_metric("pass_rate of 85%") == {"pass_rate": 0.85}


# ── Task type inference ───────────────────────────────────────────────────────

def test_infer_task_type_from_goal():
    from backend.routers.missions import _infer_task_type_from_goal
    assert _infer_task_type_from_goal("Rejection-sampling fine-tuning to reach 90% pass rate") == "rft"
    assert _infer_task_type_from_goal("Train a scikit-learn classifier on iris to 95% accuracy") == "ml"
    assert _infer_task_type_from_goal("Fine-tune the Ensemble routing model with DPO") == "dpo"
    assert _infer_task_type_from_goal("Distill conductor_gemma to 82% pass rate") == "distill"
    assert _infer_task_type_from_goal("Prompt optimization for conductor prompt") == "prompt"
    assert _infer_task_type_from_goal("Train a Snake-v0 PPO agent to achieve 100 food eaten") == "rl"



# ── incoherent task_type / target_metric ─────────────────────────────────────

def test_rl_with_pass_rate_target_is_rejected():
    """task_type became Optional[str]="rl" on 2026-09-08, so an omitted field and
    an explicit "rl" both arrive as "rl". _infer_task_type_from_goal catches
    goals that NAME their method, but an unnamed one falls through to the default
    — and an RL mission is scored by rollout in a Gym env, so it can never
    produce a routing pass_rate. Before task_type was optional this was a 422 on
    the missing field; keep it loud rather than dispatching down a path that
    cannot report the goal's own number (mission 6d999c84 did exactly that:
    a rejection-sampling goal, dispatched as rl, "completed" in 9 minutes with
    no metric)."""
    from backend.routers.missions import _reject_incoherent_task_type
    with pytest.raises(HTTPException) as exc:
        _reject_incoherent_task_type("rl", {"pass_rate": 0.9})
    assert exc.value.status_code == 422
    assert "cannot produce" in exc.value.detail
    # The message must say how to fix it, not just what is wrong.
    assert "task_type explicitly" in exc.value.detail


def test_rl_with_an_rl_metric_is_accepted():
    from backend.routers.missions import _reject_incoherent_task_type
    for metric in ({"mean_reward": 200.0}, {"food_eaten": 100.0}, {"lines_cleared": 300.0}):
        _reject_incoherent_task_type("rl", metric)      # must not raise


def test_finetune_types_may_target_pass_rate():
    """The guard is rl-only — pass_rate is exactly what these types produce."""
    from backend.routers.missions import _reject_incoherent_task_type
    for tt in ("rft", "distill", "dpo", "grpo", "prompt"):
        _reject_incoherent_task_type(tt, {"pass_rate": 0.9})   # must not raise


def test_guard_is_inert_without_a_target():
    from backend.routers.missions import _reject_incoherent_task_type
    _reject_incoherent_task_type("rl", {})
