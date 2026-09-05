"""Unit tests for recipe metric_ceiling enforcement at mission creation."""
from __future__ import annotations

import pytest
from fastapi import HTTPException

from backend.agent.code_generator import recipe_metric_ceiling
from backend.routers.missions import _CEILING_TARGET_MARGIN, _reject_unreachable_target

# The exact ceiling values are recipe-tunable (dpo moved 0.84 → 0.87 on
# 2026-09-02 after the eval-oracle split showed the warm-start already at 0.859
# — see recipes/ensemble_dpo_v1.yaml). The "declares X" tests below pin the
# current values so an accidental recipe edit is caught; the enforcement tests
# derive from the live ceiling so they don't rebreak on a deliberate re-tune.
#
# 2026-09-05: all three rebased from the 71-case static scale to the blended
# all-78-case scale when pass_rate's meaning changed (see the comment on
# _BARE_EVAL_BLENDED_RE in backend/loop/state_machine.py). These are NOT a
# re-tune of the same quantity — they measure a different population, and the
# old values would have become near-unreachable targets had they been left.
_DPO_CEILING = 0.85
_GRPO_CEILING = 0.83
_DISTILL_CEILING = 0.92


def test_dpo_recipe_declares_pass_rate_ceiling():
    ceiling = recipe_metric_ceiling("dpo")
    assert ceiling.get("pass_rate") == pytest.approx(_DPO_CEILING)


def test_grpo_recipe_declares_pass_rate_ceiling():
    assert recipe_metric_ceiling("grpo").get("pass_rate") == pytest.approx(_GRPO_CEILING)


def test_distill_recipe_declares_pass_rate_ceiling():
    assert recipe_metric_ceiling("distill").get("pass_rate") == pytest.approx(_DISTILL_CEILING)


def test_distill_target_above_ceiling_is_rejected():
    ceiling = recipe_metric_ceiling("distill")["pass_rate"]
    with pytest.raises(HTTPException) as exc:
        _reject_unreachable_target("distill", {"pass_rate": ceiling + 0.03})
    assert exc.value.status_code == 422


def test_unknown_task_type_has_no_ceiling():
    assert recipe_metric_ceiling("rl") == {}


def test_target_above_ceiling_is_rejected():
    ceiling = recipe_metric_ceiling("dpo")["pass_rate"]
    with pytest.raises(HTTPException) as exc:
        _reject_unreachable_target("dpo", {"pass_rate": ceiling + 0.05})
    assert exc.value.status_code == 422
    assert "ceiling" in exc.value.detail


def test_target_at_ceiling_is_allowed():
    ceiling = recipe_metric_ceiling("dpo")["pass_rate"]
    _reject_unreachable_target("dpo", {"pass_rate": ceiling})


def test_target_within_noise_margin_is_allowed():
    ceiling = recipe_metric_ceiling("dpo")["pass_rate"]
    _reject_unreachable_target("dpo", {"pass_rate": ceiling + _CEILING_TARGET_MARGIN * 0.5})


def test_target_below_ceiling_is_allowed():
    ceiling = recipe_metric_ceiling("dpo")["pass_rate"]
    _reject_unreachable_target("dpo", {"pass_rate": ceiling - 0.04})


def test_no_ceiling_declared_is_noop():
    _reject_unreachable_target("rl", {"mean_reward": 1e9})


def test_empty_target_is_noop():
    _reject_unreachable_target("dpo", {})


def test_non_numeric_target_is_skipped():
    _reject_unreachable_target("dpo", {"pass_rate": "high"})
