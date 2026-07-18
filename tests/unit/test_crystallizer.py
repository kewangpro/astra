"""Unit tests for pure helpers in services/crystallizer.py."""
from __future__ import annotations

import uuid
from unittest.mock import MagicMock, patch

import pytest

from backend.services.crystallizer import _slugify, _next_version, _build_recipe_content, crystallize


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
        # Uses valid SB3 PPO kwargs; invalid keys (e.g. 'lr') are filtered out
        hp = {"learning_rate": 0.001, "gamma": 0.99}
        mission = _make_mission(current_plan={"task_type": "rl", "algorithm": "ppo", "hyperparameters": hp})
        content = _build_recipe_content(mission, score=None, lessons=[])
        assert content["hyperparameters"] == hp

    def test_invalid_rl_kwargs_stripped(self):
        hp = {"learning_rate": 0.001, "gamma": 0.99, "entropy_coeff": 0.01, "dataset_path": "CartPole-v1"}
        mission = _make_mission(current_plan={"task_type": "rl", "algorithm": "ppo", "hyperparameters": hp})
        content = _build_recipe_content(mission, score=None, lessons=[])
        assert "dataset_path" not in content["hyperparameters"]
        assert "entropy_coeff" not in content["hyperparameters"]
        assert content["hyperparameters"].get("ent_coef") == 0.01

    def test_env_id_surfaced_for_rl(self):
        plan = {"task_type": "rl", "algorithm": "PPO", "env_id": "CartPole-v1", "hyperparameters": {}}
        mission = _make_mission(current_plan=plan)
        content = _build_recipe_content(mission, score=None, lessons=[])
        assert content.get("env_id") == "CartPole-v1"

    def test_domain_inferred_from_env_id(self):
        plan = {"task_type": "rl", "algorithm": "PPO", "env_id": "LunarLander-v2", "hyperparameters": {}}
        mission = _make_mission(current_plan=plan)
        content = _build_recipe_content(mission, score=None, lessons=[])
        assert content["domain"] == "LunarLander"

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

    def test_actor_critic_sets_algorithm_field(self):
        """trainer_type=actor_critic overrides algorithm to 'actor_critic'."""
        plan = {
            "task_type": "rl", "algorithm": "PPO",
            "trainer_type": "actor_critic", "hyperparameters": {},
        }
        mission = _make_mission(current_plan=plan)
        content = _build_recipe_content(mission, score=None, lessons=[])
        assert content["algorithm"] == "actor_critic"

    def test_actor_critic_strips_ppo_kwargs(self):
        """PPO-specific keys (n_steps, gae_lambda, etc.) are removed for actor_critic."""
        hp = {"learning_rate": 0.001, "gamma": 0.99, "n_steps": 512, "gae_lambda": 0.9}
        plan = {
            "task_type": "rl", "algorithm": "PPO",
            "trainer_type": "actor_critic", "hyperparameters": hp,
        }
        mission = _make_mission(current_plan=plan)
        content = _build_recipe_content(mission, score=None, lessons=[])
        assert "n_steps" not in content["hyperparameters"]
        assert "gae_lambda" not in content["hyperparameters"]
        assert content["hyperparameters"]["learning_rate"] == 0.001

    def test_actor_critic_surfaces_trainer_type(self):
        """trainer_type appears as a top-level field in the recipe content."""
        plan = {
            "task_type": "rl", "algorithm": "PPO",
            "trainer_type": "actor_critic", "hyperparameters": {},
        }
        mission = _make_mission(current_plan=plan)
        content = _build_recipe_content(mission, score=None, lessons=[])
        assert content.get("trainer_type") == "actor_critic"


# ── crystallize() — the actual DB entry point ──────────────────────────────────
# Real incident: crystallize() computed its own domain independently
# (`resolved_plan.get("domain") or mission.goal.split()[0]`) instead of
# reusing _infer_domain() — already correctly used inside _build_recipe_content()
# above — and then overwrote that correct value with the buggy one. A mission
# goal starting with "Train" (every mission goal in this codebase does — see
# _make_mission's own default) produced domain="Train" and a recipe filename
# literally named after the bug (train_rl_v13.yaml). None of the tests above
# exercise crystallize() itself, only the pure _build_recipe_content() helper,
# which is why this shipped.

class TestCrystallizeDomain:
    @pytest.fixture(autouse=True)
    def _patch_crystallizer_db_and_io(self, db_engine, monkeypatch, tmp_path):
        from sqlalchemy.ext.asyncio import async_sessionmaker, AsyncSession
        maker = async_sessionmaker(db_engine, expire_on_commit=False, class_=AsyncSession)
        monkeypatch.setattr("backend.services.crystallizer.AsyncSessionLocal", maker)
        monkeypatch.setattr("backend.services.crystallizer.settings.recipes_path", str(tmp_path))
        monkeypatch.setattr(
            "backend.services.crystallizer.vector_memory.query_lessons", lambda *a, **kw: []
        )
        monkeypatch.setattr(
            "backend.services.crystallizer.recipe_library.index_recipe", lambda *a, **kw: None
        )

    @pytest.fixture
    async def seeded_mission(self, db_session):
        from backend.models.mission import Mission, MissionStatus
        mission = Mission(
            id=str(uuid.uuid4()),
            goal="Train a Tetris-v0 DQN agent to achieve 200 lines_cleared",
            task_type="rl",
            target_metric={"lines_cleared": 200.0},
            autonomy_mode="supervised",
            status=MissionStatus.COMPLETED.value,
            current_iteration=1,
            best_metric_value="300.0",
            current_plan={
                "task_type": "rl", "algorithm": "DQN", "env_id": "Tetris-v0",
                "trainer_type": "actor_critic", "hyperparameters": {"learning_rate": 0.001},
            },
        )
        db_session.add(mission)
        await db_session.commit()
        return mission

    async def test_domain_not_first_word_of_goal(self, seeded_mission, db_session):
        """The exact real-incident reproduction: a goal starting with 'Train'
        must not produce domain='Train' when a real env_id is available."""
        record = await crystallize(seeded_mission.id, score=300.0)
        assert record is not None
        assert record.domain == "Tetris"
        assert record.domain != "Train"

    async def test_recipe_name_reflects_correct_domain(self, seeded_mission, db_session):
        """The buggy path also produced filenames like train_rl_v13.yaml —
        the name must be built from the corrected domain."""
        record = await crystallize(seeded_mission.id, score=300.0)
        assert record.name.startswith("tetris_")
        assert not record.name.startswith("train_")
