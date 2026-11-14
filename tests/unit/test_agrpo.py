"""Unit tests for Agent GRPO (agrpo) Astra wiring."""
from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import MagicMock, patch

from backend.agent.code_generator import CodeGenerator, _load_recipe_for_env
from backend.agent.task_catalog import task_type_enum
from backend.routers.missions import _infer_task_type_from_goal
from backend.sandbox.manager import _FINETUNE_REMOTE_TASK_TYPES


def test_agrpo_in_catalog_and_remote_set():
    assert "agrpo" in task_type_enum()
    assert "agrpo" in _FINETUNE_REMOTE_TASK_TYPES


def test_infer_agrpo_from_goal():
    assert _infer_task_type_from_goal("Train agentic RL with Agent GRPO") == "agrpo"
    assert _infer_task_type_from_goal("Run agrpo on MultiTurnAgentGym") == "agrpo"
    assert _infer_task_type_from_goal("Train GRPO routing adapter") == "grpo"


def test_agent_gym_agrpo_recipe_and_codegen():
    recipe = _load_recipe_for_env("agrpo", "")
    assert recipe["name"] == "agent_gym_agrpo_v1"
    assert recipe["task_type"] == "agrpo"

    generator = CodeGenerator(provider=MagicMock())
    plan = {
        "task_type": "agrpo",
        "recipe": "agent_gym_agrpo_v1",
        "hyperparameters": {},
        "target_metric": {"pass_rate": 0.5},
    }
    prompt = generator._build_user_prompt(
        task_type="agrpo",
        mission_id="test-agrpo-1",
        plan=plan,
        checkpoint_dir="/tmp/adapters/astra_test",
    )
    assert "astrarl.make" in prompt
    assert "mlx_grpo_traj" in prompt
    assert "use_execv=True" in prompt
    assert "astra_pkg_root" in prompt or "/Users/kewang/astra-pkg" in prompt
    assert "reward_mode" in prompt
    assert "agent_gym" in prompt
    assert "agent_grpo_train.py" in prompt  # facade docs still name the CLI
    # Live heal prompt must not teach a hand-built argv list.
    assert '"--reward-mode"' not in prompt


def test_generate_training_script_agrpo_canned_facade(tmp_path, monkeypatch):
    monkeypatch.setattr("backend.config.settings.data_path", str(tmp_path))
    monkeypatch.setattr("backend.config.settings.api_port", 8200)
    monkeypatch.setattr("backend.config.settings.sandbox_host", "mac-mini.local")

    provider = MagicMock()
    gen = CodeGenerator(provider)
    plan = {
        "task_type": "agrpo",
        "recipe": "agent_gym_agrpo_v1",
        "hyperparameters": {},
        "target_metric": {"pass_rate": 0.5},
    }
    with patch.object(CodeGenerator, "_query_lessons", return_value=[]):
        path = asyncio.get_event_loop().run_until_complete(
            gen.generate_training_script("mission-agrpo-facade", plan)
        )
    text = Path(path).read_text()
    assert "astrarl.make" in text
    assert "mlx_grpo_traj" in text
    assert "use_execv=True" in text
    assert "/Users/kewang/finetune" in text
    assert "/Users/kewang/finetune-env" in text
    assert "/Users/kewang/astra-pkg" in text
    assert "expanduser" not in text
    assert "agent_grpo_train.py" not in text  # argv list gone; facade only
    assert "no_warmstart=True" in text or 'adapter="' in text
    provider.generate.assert_not_called()


def test_ensemble_agent_grpo_recipe_e2e_mode():
    recipe = _load_recipe_for_env("ensemble_agent_grpo_v1", "")
    assert recipe["task_type"] == "agrpo"
    assert recipe["hyperparameters"]["reward_mode"] == "e2e"
    assert "agent_e2e_stubs.yaml" in recipe["hyperparameters"]["e2e_cases"]
