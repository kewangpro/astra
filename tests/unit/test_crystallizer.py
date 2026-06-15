"""Unit tests for pure helpers in services/crystallizer.py."""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from backend.services.crystallizer import _slugify, _next_version, _build_recipe_content


# ── _slugify ──────────────────────────────────────────────────────────────────

class TestSlugify:
    def test_lowercases(self):
        assert _slugify("CartPole") == "cartpole"

    def test_replaces_spaces_with_underscores(self):
        assert _slugify("cart pole") == "cart_pole"

    def test_strips_special_chars(self):
        assert _slugify("cart-pole!v2") == "cart_pole_v2"

    def test_trims_leading_trailing_underscores(self):
        assert _slugify("__hello__") == "hello"

    def test_truncates_to_60_chars(self):
        long = "a" * 100
        result = _slugify(long)
        assert len(result) <= 60

    def test_empty_string(self):
        result = _slugify("")
        assert isinstance(result, str)

    def test_numbers_preserved(self):
        assert _slugify("ppo v2 3") == "ppo_v2_3"

    def test_multiple_separators_collapsed(self):
        result = _slugify("a---b   c")
        assert "__" not in result
        assert result == "a_b_c"


# ── _next_version ─────────────────────────────────────────────────────────────

class TestNextVersion:
    def test_returns_one_when_no_existing(self):
        assert _next_version([], "cartpole_rl") == 1

    def test_returns_next_after_existing(self):
        names = ["cartpole_rl_v1", "cartpole_rl_v2"]
        assert _next_version(names, "cartpole_rl") == 3

    def test_ignores_non_matching_names(self):
        names = ["other_v5", "cartpole_rl_v1"]
        assert _next_version(names, "cartpole_rl") == 2

    def test_ignores_non_digit_suffix(self):
        names = ["cartpole_rl_vabc", "cartpole_rl_v1"]
        assert _next_version(names, "cartpole_rl") == 2

    def test_handles_gaps_in_versions(self):
        names = ["cartpole_rl_v1", "cartpole_rl_v5"]
        assert _next_version(names, "cartpole_rl") == 6

    def test_does_not_match_partial_base(self):
        names = ["cartpole_rl_extended_v3"]
        # base is "cartpole_rl", not "cartpole_rl_extended"
        assert _next_version(names, "cartpole_rl") == 1

    def test_single_existing(self):
        assert _next_version(["foo_v1"], "foo") == 2


# ── _build_recipe_content ─────────────────────────────────────────────────────

def _make_mission(
    id: str = "abc12345-0000",
    goal: str = "Train CartPole agent",
    task_type: str = "rl",
    current_plan: dict | None = None,
    best_metric_value: float | None = None,
    current_iteration: int = 5,
    target_metric: dict | None = None,
) -> MagicMock:
    m = MagicMock()
    m.id = id
    m.goal = goal
    m.task_type = task_type
    m.current_plan = current_plan or {}
    m.best_metric_value = best_metric_value
    m.current_iteration = current_iteration
    m.target_metric = target_metric
    return m


class TestBuildRecipeContent:
    def test_task_type_uppercased(self):
        mission = _make_mission(current_plan={"task_type": "rl", "algorithm": "ppo", "hyperparameters": {}})
        content = _build_recipe_content(mission, score=None, lessons=[])
        assert content["task_type"] == "RL"

    def test_algorithm_included(self):
        mission = _make_mission(current_plan={"algorithm": "sac", "task_type": "rl", "hyperparameters": {}})
        content = _build_recipe_content(mission, score=None, lessons=[])
        assert content["algorithm"] == "sac"

    def test_score_in_description_when_present(self):
        mission = _make_mission(current_plan={"task_type": "rl", "algorithm": "ppo", "hyperparameters": {}})
        content = _build_recipe_content(mission, score=0.9876, lessons=[])
        assert "0.9876" in content["description"]

    def test_description_without_score(self):
        mission = _make_mission(current_plan={"task_type": "rl", "algorithm": "ppo", "hyperparameters": {}})
        content = _build_recipe_content(mission, score=None, lessons=[])
        assert "mission" in content["description"].lower()

    def test_hyperparameters_extracted(self):
        hp = {"lr": 0.001, "gamma": 0.99}
        mission = _make_mission(current_plan={"task_type": "rl", "algorithm": "ppo", "hyperparameters": hp})
        content = _build_recipe_content(mission, score=None, lessons=[])
        assert content["hyperparameters"] == hp

    def test_curriculum_included_when_present(self):
        plan = {"task_type": "rl", "algorithm": "ppo", "hyperparameters": {},
                "curriculum_phases": [{"steps": 1000}]}
        mission = _make_mission(current_plan=plan)
        content = _build_recipe_content(mission, score=None, lessons=[])
        assert "curriculum" in content
        assert content["curriculum"]["phases"] == [{"steps": 1000}]

    def test_curriculum_absent_when_not_in_plan(self):
        mission = _make_mission(current_plan={"task_type": "rl", "algorithm": "ppo", "hyperparameters": {}})
        content = _build_recipe_content(mission, score=None, lessons=[])
        assert "curriculum" not in content

    def test_lessons_truncated_to_three(self):
        lessons = [{"text": f"lesson {i}" * 5} for i in range(10)]
        mission = _make_mission(current_plan={"task_type": "rl", "algorithm": "ppo", "hyperparameters": {}})
        content = _build_recipe_content(mission, score=None, lessons=lessons)
        assert len(content["lessons"]) == 3

    def test_no_lessons_key_when_empty(self):
        mission = _make_mission(current_plan={"task_type": "rl", "algorithm": "ppo", "hyperparameters": {}})
        content = _build_recipe_content(mission, score=None, lessons=[])
        assert "lessons" not in content

    def test_target_metric_included_when_set(self):
        mission = _make_mission(
            current_plan={"task_type": "rl", "algorithm": "ppo", "hyperparameters": {}},
            target_metric={"mean_reward": 200.0},
        )
        content = _build_recipe_content(mission, score=None, lessons=[])
        assert content["target_metric"] == {"mean_reward": 200.0}

    def test_provenance_contains_mission_id(self):
        mission = _make_mission(id="test-mission-id-xyz")
        content = _build_recipe_content(mission, score=5.0, lessons=[])
        assert content["provenance"]["mission_id"] == "test-mission-id-xyz"

    def test_provenance_contains_iterations(self):
        mission = _make_mission(current_iteration=7)
        content = _build_recipe_content(mission, score=None, lessons=[])
        assert content["provenance"]["iterations"] == 7
