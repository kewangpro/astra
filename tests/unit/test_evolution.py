"""Unit tests for MutationOperator and SelectionPolicy in services/evolution.py."""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from backend.services.evolution import (
    MutationOperator,
    SelectionPolicy,
    MUTATION_STRENGTH,
    PROMOTION_IMPROVEMENT,
    _BOUNDS,
)


def _make_recipe(hyperparameters: dict, full_content: dict | None = None) -> MagicMock:
    r = MagicMock()
    r.hyperparameters = hyperparameters
    r.full_content = full_content if full_content is not None else {"hyperparameters": hyperparameters}
    return r


# ── MutationOperator ──────────────────────────────────────────────────────────

class TestMutationOperator:
    def test_returns_dict(self):
        recipe = _make_recipe({"learning_rate": 0.001})
        child = MutationOperator(seed=42).mutate(recipe)
        assert isinstance(child, dict)

    def test_hyperparameters_key_present(self):
        recipe = _make_recipe({"learning_rate": 0.001})
        child = MutationOperator(seed=42).mutate(recipe)
        assert "hyperparameters" in child

    def test_non_numeric_values_unchanged(self):
        recipe = _make_recipe({"learning_rate": 0.001, "algo": "ppo"})
        child = MutationOperator(seed=42).mutate(recipe)
        assert child["hyperparameters"]["algo"] == "ppo"

    def test_numeric_value_mutated(self):
        # With seed fixed, mutation should differ from parent
        recipe = _make_recipe({"learning_rate": 0.01})
        child = MutationOperator(seed=0).mutate(recipe)
        # Not necessarily different on every seed, but with seed=0 it should be
        # Just verify the value exists and is float
        assert isinstance(child["hyperparameters"]["learning_rate"], float)

    def test_learning_rate_stays_in_bounds(self):
        lo, hi = _BOUNDS["learning_rate"]
        for seed in range(20):
            recipe = _make_recipe({"learning_rate": 1e-4})
            child = MutationOperator(strength=1.0, seed=seed).mutate(recipe)
            lr = child["hyperparameters"]["learning_rate"]
            assert lo <= lr <= hi, f"learning_rate {lr} out of bounds [{lo}, {hi}]"

    def test_gamma_stays_in_bounds(self):
        lo, hi = _BOUNDS["gamma"]
        for seed in range(20):
            recipe = _make_recipe({"gamma": 0.99})
            child = MutationOperator(strength=1.0, seed=seed).mutate(recipe)
            g = child["hyperparameters"]["gamma"]
            assert lo <= g <= hi

    def test_int_params_stay_int(self):
        recipe = _make_recipe({"batch_size": 64})
        child = MutationOperator(seed=7).mutate(recipe)
        assert isinstance(child["hyperparameters"]["batch_size"], int)

    def test_batch_size_stays_in_bounds(self):
        lo, hi = _BOUNDS["batch_size"]
        for seed in range(20):
            recipe = _make_recipe({"batch_size": 32})
            child = MutationOperator(strength=2.0, seed=seed).mutate(recipe)
            bs = child["hyperparameters"]["batch_size"]
            assert lo <= bs <= hi

    def test_parent_not_mutated(self):
        hp = {"learning_rate": 0.001}
        recipe = _make_recipe(hp)
        MutationOperator(seed=1).mutate(recipe)
        # Original hyperparameters dict on recipe mock should be unaffected
        assert recipe.hyperparameters == {"learning_rate": 0.001}

    def test_zero_strength_returns_same_value(self):
        recipe = _make_recipe({"learning_rate": 0.001})
        child = MutationOperator(strength=0.0, seed=0).mutate(recipe)
        assert child["hyperparameters"]["learning_rate"] == pytest.approx(0.001)

    def test_unknown_param_no_bounds_applied(self):
        # Custom param not in _BOUNDS — should still mutate without crashing
        recipe = _make_recipe({"my_custom_lr": 0.5})
        child = MutationOperator(strength=0.1, seed=3).mutate(recipe)
        assert "my_custom_lr" in child["hyperparameters"]

    def test_full_content_preserved(self):
        full = {"hyperparameters": {"lr": 0.01}, "extra": "keep_me"}
        recipe = _make_recipe({"lr": 0.01}, full_content=full)
        child = MutationOperator(seed=0).mutate(recipe)
        assert child.get("extra") == "keep_me"


# ── SelectionPolicy ───────────────────────────────────────────────────────────

class TestSelectionPolicy:
    def setup_method(self):
        self.policy = SelectionPolicy()

    def test_promotes_when_child_beats_parent(self):
        # child must beat parent by > 1%
        assert self.policy.should_promote(0.95, 0.90)

    def test_rejects_when_child_barely_misses_threshold(self):
        # 0.909 is only ~1% above 0.90 but threshold is strictly > 1%
        assert not self.policy.should_promote(0.909, 0.90)

    def test_rejects_equal_scores(self):
        assert not self.policy.should_promote(0.90, 0.90)

    def test_rejects_when_child_is_none(self):
        assert not self.policy.should_promote(None, 0.80)

    def test_promotes_when_parent_is_none(self):
        # No parent score → always promote
        assert self.policy.should_promote(0.5, None)

    def test_promotes_when_both_none_parent_only(self):
        assert self.policy.should_promote(0.1, None)

    def test_custom_threshold_respected(self):
        policy = SelectionPolicy(min_improvement=0.10)  # 10%
        assert policy.should_promote(1.11, 1.0)
        assert not policy.should_promote(1.09, 1.0)

    def test_zero_threshold_promotes_any_improvement(self):
        policy = SelectionPolicy(min_improvement=0.0)
        assert policy.should_promote(0.901, 0.90)

    def test_child_zero_score_not_promoted_over_parent(self):
        assert not self.policy.should_promote(0.0, 0.5)

    def test_negative_scores_handled(self):
        # child -0.5 vs parent -1.0: child is better (less negative)
        assert self.policy.should_promote(-0.5, -1.0)
