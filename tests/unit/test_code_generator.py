"""Unit tests for CodeGenerator (backend/agent/code_generator.py)."""
from __future__ import annotations

import asyncio
import json
import pytest
from unittest.mock import AsyncMock, patch

from backend.agent.code_generator import CodeGenerator


# ── helpers ───────────────────────────────────────────────────────────────────

def _make_provider(response: str = "import gymnasium as gym\npass") -> AsyncMock:
    provider = AsyncMock()
    provider.generate = AsyncMock(return_value=response)
    return provider


def _make_rl_plan(**overrides) -> dict:
    base = {
        "task_type": "rl",
        "algorithm": "PPO",
        "env_id": "CartPole-v1",
        "target_metric": {"mean_reward": 475},
        "hyperparameters": {"learning_rate": 0.001, "n_steps": 2048},
    }
    base.update(overrides)
    return base


def _make_ml_plan(**overrides) -> dict:
    base = {
        "task_type": "ml",
        "target_metric": {"accuracy": 0.92},
        "hyperparameters": {"model_params": {}},
    }
    base.update(overrides)
    return base


# ── _strip_fences (static helper) ────────────────────────────────────────────

def test_strip_fences_removes_python_block():
    code = "```python\nimport gymnasium\n```"
    result = CodeGenerator._strip_fences(code)
    assert "```" not in result
    assert "import gymnasium" in result


def test_strip_fences_passthrough_clean():
    code = "import gymnasium as gym\nenv = gym.make('CartPole-v1')\n"
    assert CodeGenerator._strip_fences(code).strip() == code.strip()


def test_strip_fences_removes_stop_tokens():
    code = "pass\n<|endoftext|>"
    result = CodeGenerator._strip_fences(code)
    assert "<|endoftext|>" not in result


# ── _build_user_prompt ────────────────────────────────────────────────────────

def test_build_user_prompt_rl_contains_env_id(tmp_path, monkeypatch):
    monkeypatch.setattr("backend.config.settings.data_path", str(tmp_path))
    monkeypatch.setattr("backend.config.settings.api_port", 8200)
    monkeypatch.setattr("backend.config.settings.sandbox_host", None)

    gen = CodeGenerator(_make_provider())
    plan = _make_rl_plan()
    prompt = gen._build_user_prompt("rl", "test-id", plan, str(tmp_path / "ckpt"))

    assert 'gym.make("CartPole-v1")' in prompt
    assert "PPO" in prompt
    assert "475" in prompt                # target_reward


def test_build_user_prompt_rl_contains_telemetry_guard(tmp_path, monkeypatch):
    monkeypatch.setattr("backend.config.settings.data_path", str(tmp_path))
    monkeypatch.setattr("backend.config.settings.api_port", 8200)
    monkeypatch.setattr("backend.config.settings.sandbox_host", None)

    gen = CodeGenerator(_make_provider())
    plan = _make_rl_plan()
    prompt = gen._build_user_prompt("rl", "test-id", plan, str(tmp_path / "ckpt"))

    assert "n_calls % 2048 == 0" in prompt


def test_build_user_prompt_rl_reads_env_id_from_plan(tmp_path, monkeypatch):
    monkeypatch.setattr("backend.config.settings.data_path", str(tmp_path))
    monkeypatch.setattr("backend.config.settings.api_port", 8200)
    monkeypatch.setattr("backend.config.settings.sandbox_host", None)

    gen = CodeGenerator(_make_provider())
    plan = _make_rl_plan(env_id="LunarLander-v2")
    prompt = gen._build_user_prompt("rl", "test-id", plan, str(tmp_path / "ckpt"))

    assert "LunarLander-v2" in prompt


def test_build_user_prompt_ml(tmp_path, monkeypatch):
    monkeypatch.setattr("backend.config.settings.data_path", str(tmp_path))
    monkeypatch.setattr("backend.config.settings.api_port", 8200)
    monkeypatch.setattr("backend.config.settings.sandbox_host", None)

    gen = CodeGenerator(_make_provider())
    plan = _make_ml_plan()
    prompt = gen._build_user_prompt("ml", "test-id", plan, str(tmp_path / "ckpt"))

    assert "RandomForestClassifier" in prompt
    assert "sklearn" in prompt


# ── _query_lessons ────────────────────────────────────────────────────────────

def test_query_lessons_returns_empty_on_exception():
    with patch("backend.services.vector_memory.query_lessons", side_effect=RuntimeError("db down")):
        result = CodeGenerator._query_lessons({"task_type": "rl", "algorithm": "PPO"})
    assert result == []


def test_query_lessons_filters_missing_text():
    mock_results = [{"text": "avoid actor_lr"}, {"text": None}, {"text": "use gymnasium"}]
    with patch("backend.services.vector_memory.query_lessons", return_value=mock_results):
        result = CodeGenerator._query_lessons({"task_type": "rl", "algorithm": "PPO"})
    assert result == ["avoid actor_lr", "use gymnasium"]


def test_query_lessons_returns_empty_list_on_import_error():
    with patch("backend.services.vector_memory.query_lessons", side_effect=ImportError("no chroma")):
        result = CodeGenerator._query_lessons({"task_type": "rl"})
    assert result == []


# ── generate_training_script (integration-light) ─────────────────────────────

def test_generate_training_script_writes_file(tmp_path, monkeypatch):
    monkeypatch.setattr("backend.config.settings.data_path", str(tmp_path))
    monkeypatch.setattr("backend.config.settings.api_port", 8200)
    monkeypatch.setattr("backend.config.settings.sandbox_host", None)

    code = "import gymnasium as gym\nenv = gym.make('CartPole-v1')\n"
    gen = CodeGenerator(_make_provider(code))

    with patch.object(CodeGenerator, "_query_lessons", return_value=[]):
        path = asyncio.get_event_loop().run_until_complete(
            gen.generate_training_script("mission-123", _make_rl_plan())
        )

    assert path.endswith("train.py")
    assert "gymnasium" in open(path).read()


# ── _patch_rl_imports ─────────────────────────────────────────────────────────

def test_patch_rl_imports_adds_ppo():
    code = "import numpy as np\nmodel = PPO('MlpPolicy', env)"
    result = CodeGenerator._patch_rl_imports(code)
    assert "from stable_baselines3 import PPO" in result


def test_patch_rl_imports_adds_basecallback():
    code = "import numpy as np\nclass Cb(BaseCallback): pass"
    result = CodeGenerator._patch_rl_imports(code)
    assert "from stable_baselines3.common.callbacks import BaseCallback" in result


def test_patch_rl_imports_strips_bad_callback_kwargs():
    code = (
        "from stable_baselines3.common.callbacks import BaseCallback\n"
        "class CustomCallback(BaseCallback): pass\n"
        "callback = CustomCallback(checkpoint_freq=2048, save_path='./ckpt')"
    )
    result = CodeGenerator._patch_rl_imports(code)
    assert "checkpoint_freq" not in result
    assert "CustomCallback()" in result


def test_patch_rl_imports_noop_when_clean():
    code = (
        "from stable_baselines3 import PPO\n"
        "from stable_baselines3.common.callbacks import BaseCallback\n"
        "callback = CustomCallback()"
    )
    result = CodeGenerator._patch_rl_imports(code)
    assert result.count("from stable_baselines3 import PPO") == 1


def test_patch_rl_imports_adds_checkpoint_callback():
    code = "import numpy as np\ncb = CheckpointCallback(save_freq=1000)"
    result = CodeGenerator._patch_rl_imports(code)
    assert "from stable_baselines3.common.callbacks import CheckpointCallback" in result


def test_patch_rl_imports_replaces_sb3_alias():
    code = "import stable_baselines3 as sb3\nmodel = sb3.PPO('MlpPolicy', env)"
    result = CodeGenerator._patch_rl_imports(code)
    assert "import stable_baselines3 as sb3" not in result
    assert "from stable_baselines3 import PPO" in result
    assert "sb3.PPO" not in result


def test_build_user_prompt_rl_includes_policy_kwargs(tmp_path, monkeypatch):
    monkeypatch.setattr("backend.config.settings.data_path", str(tmp_path))
    monkeypatch.setattr("backend.config.settings.api_port", 8200)
    monkeypatch.setattr("backend.config.settings.sandbox_host", None)

    gen = CodeGenerator(_make_provider())
    plan = _make_rl_plan(hyperparameters={
        "learning_rate": 0.001,
        "n_steps": 2048,
        "policy_kwargs": {"net_arch": [256, 256]},
    })
    prompt = gen._build_user_prompt("rl", "test-id", plan, str(tmp_path / "ckpt"))

    assert "[256, 256]" in prompt
    assert "policy_kwargs" in prompt


def test_build_user_prompt_rl_policy_kwargs_none_when_absent(tmp_path, monkeypatch):
    monkeypatch.setattr("backend.config.settings.data_path", str(tmp_path))
    monkeypatch.setattr("backend.config.settings.api_port", 8200)
    monkeypatch.setattr("backend.config.settings.sandbox_host", None)

    gen = CodeGenerator(_make_provider())
    plan = _make_rl_plan()
    prompt = gen._build_user_prompt("rl", "test-id", plan, str(tmp_path / "ckpt"))

    assert "Policy kwargs (network architecture): none" in prompt


def test_generate_training_script_injects_lessons(tmp_path, monkeypatch):
    monkeypatch.setattr("backend.config.settings.data_path", str(tmp_path))
    monkeypatch.setattr("backend.config.settings.api_port", 8200)
    monkeypatch.setattr("backend.config.settings.sandbox_host", None)

    captured = {}

    async def mock_generate(messages, config):
        captured["system"] = messages[0].content
        return "pass"

    provider = AsyncMock()
    provider.generate = mock_generate
    gen = CodeGenerator(provider)

    lessons = ["avoid actor_lr kwarg", "use gymnasium not gym"]
    with patch.object(CodeGenerator, "_query_lessons", return_value=lessons):
        asyncio.get_event_loop().run_until_complete(
            gen.generate_training_script("mission-456", _make_rl_plan())
        )

    assert "avoid actor_lr kwarg" in captured["system"]
    assert "use gymnasium not gym" in captured["system"]
