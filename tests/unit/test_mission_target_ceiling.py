"""Unit tests for recipe metric_ceiling enforcement at mission creation."""
from __future__ import annotations

import pytest
from fastapi import HTTPException

from backend.agent.code_generator import recipe_metric_ceiling
from backend.routers.missions import _reject_unreachable_target


def test_dpo_recipe_declares_pass_rate_ceiling():
    ceiling = recipe_metric_ceiling("dpo")
    assert ceiling.get("pass_rate") == pytest.approx(0.84)


def test_grpo_recipe_declares_pass_rate_ceiling():
    assert recipe_metric_ceiling("grpo").get("pass_rate") == pytest.approx(0.84)


def test_unknown_task_type_has_no_ceiling():
    assert recipe_metric_ceiling("rl") == {}


def test_target_above_ceiling_is_rejected():
    with pytest.raises(HTTPException) as exc:
        _reject_unreachable_target("dpo", {"pass_rate": 0.85})
    assert exc.value.status_code == 422
    assert "ceiling" in exc.value.detail


def test_target_at_ceiling_is_allowed():
    _reject_unreachable_target("dpo", {"pass_rate": 0.84})


def test_target_within_noise_margin_is_allowed():
    _reject_unreachable_target("dpo", {"pass_rate": 0.843})


def test_target_below_ceiling_is_allowed():
    _reject_unreachable_target("dpo", {"pass_rate": 0.80})


def test_no_ceiling_declared_is_noop():
    _reject_unreachable_target("rl", {"mean_reward": 1e9})


def test_empty_target_is_noop():
    _reject_unreachable_target("dpo", {})


def test_non_numeric_target_is_skipped():
    _reject_unreachable_target("dpo", {"pass_rate": "high"})
