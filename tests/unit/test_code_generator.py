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

    assert "Policy kwargs (network architecture): None" in prompt


def test_build_user_prompt_rl_includes_best_model_save(tmp_path, monkeypatch):
    monkeypatch.setattr("backend.config.settings.data_path", str(tmp_path))
    monkeypatch.setattr("backend.config.settings.api_port", 8200)
    monkeypatch.setattr("backend.config.settings.sandbox_host", None)

    gen = CodeGenerator(_make_provider())
    plan = _make_rl_plan()
    prompt = gen._build_user_prompt("rl", "test-id", plan, str(tmp_path / "ckpt"))

    assert "best_model" in prompt
    assert "_best_reward" in prompt


def test_build_user_prompt_rl_includes_last_model_save(tmp_path, monkeypatch):
    monkeypatch.setattr("backend.config.settings.data_path", str(tmp_path))
    monkeypatch.setattr("backend.config.settings.api_port", 8200)
    monkeypatch.setattr("backend.config.settings.sandbox_host", None)

    gen = CodeGenerator(_make_provider())
    plan = _make_rl_plan()
    prompt = gen._build_user_prompt("rl", "test-id", plan, str(tmp_path / "ckpt"))

    assert "last_model" in prompt


def test_build_user_prompt_rl_hardcodes_hyperparameters(tmp_path, monkeypatch):
    monkeypatch.setattr("backend.config.settings.data_path", str(tmp_path))
    monkeypatch.setattr("backend.config.settings.api_port", 8200)
    monkeypatch.setattr("backend.config.settings.sandbox_host", None)

    gen = CodeGenerator(_make_provider())
    plan = _make_rl_plan(hyperparameters={"learning_rate": 0.0005, "n_steps": 1024, "batch_size": 128})
    prompt = gen._build_user_prompt("rl", "test-id", plan, str(tmp_path / "ckpt"))

    # Exact pivot values must appear verbatim in the code block
    assert "_hp = " in prompt
    assert "0.0005" in prompt
    assert "_VALID_PPO_KEYS" in prompt
    assert "_filtered" in prompt
    assert "_policy_kwargs" in prompt


def test_build_user_prompt_rl_policy_kwargs_none_renders_as_python_none(tmp_path, monkeypatch):
    monkeypatch.setattr("backend.config.settings.data_path", str(tmp_path))
    monkeypatch.setattr("backend.config.settings.api_port", 8200)
    monkeypatch.setattr("backend.config.settings.sandbox_host", None)

    gen = CodeGenerator(_make_provider())
    plan = _make_rl_plan()
    prompt = gen._build_user_prompt("rl", "test-id", plan, str(tmp_path / "ckpt"))

    assert "_policy_kwargs = None" in prompt


def test_build_user_prompt_rl_includes_warm_start_block(tmp_path, monkeypatch):
    monkeypatch.setattr("backend.config.settings.data_path", str(tmp_path))
    monkeypatch.setattr("backend.config.settings.api_port", 8200)
    monkeypatch.setattr("backend.config.settings.sandbox_host", None)

    gen = CodeGenerator(_make_provider())
    plan = _make_rl_plan()
    prompt = gen._build_user_prompt("rl", "test-id", plan, str(tmp_path / "ckpt"))

    assert "best_model.zip" in prompt
    assert "load_state_dict" in prompt
    assert "_best_ckpt" in prompt
    # architecture mismatch must be caught, not crash the script
    assert "except" in prompt


def test_build_user_prompt_ml_hardcodes_checkpoint_path(tmp_path, monkeypatch):
    monkeypatch.setattr("backend.config.settings.data_path", str(tmp_path))
    monkeypatch.setattr("backend.config.settings.api_port", 8200)
    monkeypatch.setattr("backend.config.settings.sandbox_host", None)

    gen = CodeGenerator(_make_provider())
    plan = _make_ml_plan()
    ckpt = str(tmp_path / "ckpt")
    prompt = gen._build_user_prompt("ml", "test-id", plan, ckpt)

    assert f"{ckpt}/model.joblib" in prompt
    assert "joblib.dump" in prompt


def test_generate_training_script_injects_snake_preamble(tmp_path, monkeypatch):
    monkeypatch.setattr("backend.config.settings.data_path", str(tmp_path))
    monkeypatch.setattr("backend.config.settings.api_port", 8200)
    monkeypatch.setattr("backend.config.settings.sandbox_host", None)

    # LLM returns script without registration preamble
    code = "import gymnasium as gym\nenv = gym.make('Snake-v0')\n"
    gen = CodeGenerator(_make_provider(code))
    plan = _make_rl_plan(env_id="Snake-v0")

    with patch.object(CodeGenerator, "_query_lessons", return_value=[]):
        path = asyncio.get_event_loop().run_until_complete(
            gen.generate_training_script("mission-snake", plan)
        )

    content = open(path).read()
    assert "register" in content
    assert "snake_env" in content


def test_generate_training_script_no_snake_preamble_for_non_snake(tmp_path, monkeypatch):
    monkeypatch.setattr("backend.config.settings.data_path", str(tmp_path))
    monkeypatch.setattr("backend.config.settings.api_port", 8200)
    monkeypatch.setattr("backend.config.settings.sandbox_host", None)

    code = "import gymnasium as gym\nenv = gym.make('CartPole-v1')\n"
    gen = CodeGenerator(_make_provider(code))
    plan = _make_rl_plan(env_id="CartPole-v1")

    with patch.object(CodeGenerator, "_query_lessons", return_value=[]):
        path = asyncio.get_event_loop().run_until_complete(
            gen.generate_training_script("mission-cartpole", plan)
        )

    content = open(path).read()
    assert "snake_env" not in content


def test_build_user_prompt_rl_env_kwargs_injected(tmp_path, monkeypatch):
    monkeypatch.setattr("backend.config.settings.data_path", str(tmp_path))
    monkeypatch.setattr("backend.config.settings.api_port", 8200)
    monkeypatch.setattr("backend.config.settings.sandbox_host", None)

    gen = CodeGenerator(_make_provider())
    plan = _make_rl_plan(env_kwargs={"food_reward": 20.0, "distance_weight": 0.0})
    prompt = gen._build_user_prompt("rl", "test-id", plan, str(tmp_path / "ckpt"))

    assert "food_reward=20.0" in prompt
    assert "distance_weight=0.0" in prompt


def test_build_user_prompt_rl_no_env_kwargs_clean(tmp_path, monkeypatch):
    monkeypatch.setattr("backend.config.settings.data_path", str(tmp_path))
    monkeypatch.setattr("backend.config.settings.api_port", 8200)
    monkeypatch.setattr("backend.config.settings.sandbox_host", None)

    gen = CodeGenerator(_make_provider())
    plan = _make_rl_plan()
    prompt = gen._build_user_prompt("rl", "test-id", plan, str(tmp_path / "ckpt"))

    assert 'gym.make("CartPole-v1")' in prompt


def test_fix_checkpoint_paths_replaces_relative_paths():
    ckpt = "/abs/path/to/checkpoints"
    code = (
        "model.save('./data/missions/2b395824-e128-453d-9d58-e1ba241a3522/checkpoints/best_model')\n"
        "open('data/missions/2b395824-e128-453d-9d58-e1ba241a3522/checkpoints/best_score.txt')\n"
    )
    result = CodeGenerator._fix_checkpoint_paths(code, ckpt)
    assert "./data/missions/" not in result
    assert ckpt in result
    assert result.count(ckpt) == 2


def test_fix_checkpoint_paths_leaves_absolute_paths_alone():
    ckpt = "/abs/path/to/checkpoints"
    code = f"model.save('{ckpt}/best_model')\n"
    result = CodeGenerator._fix_checkpoint_paths(code, ckpt)
    assert result == code


def test_generate_training_script_writes_train_config(tmp_path, monkeypatch):
    monkeypatch.setattr("backend.config.settings.data_path", str(tmp_path))
    monkeypatch.setattr("backend.config.settings.api_port", 8200)
    monkeypatch.setattr("backend.config.settings.sandbox_host", None)

    code = "import gymnasium as gym\nenv = gym.make('CartPole-v1')\npass"
    gen = CodeGenerator(_make_provider(code))
    plan = _make_rl_plan(algorithm="DQN", env_kwargs={"food_reward": 20.0, "distance_weight": 0.0})

    with patch.object(CodeGenerator, "_query_lessons", return_value=[]):
        asyncio.get_event_loop().run_until_complete(
            gen.generate_training_script("mission-cfg", plan)
        )

    cfg_path = tmp_path / "missions" / "mission-cfg" / "checkpoints" / "train_config.json"
    assert cfg_path.exists()
    cfg = json.loads(cfg_path.read_text())
    assert cfg["algorithm"] == "DQN"
    assert cfg["env_kwargs"]["food_reward"] == 20.0
    assert cfg["env_kwargs"]["distance_weight"] == 0.0


def test_generate_training_script_writes_train_config_no_env_kwargs(tmp_path, monkeypatch):
    monkeypatch.setattr("backend.config.settings.data_path", str(tmp_path))
    monkeypatch.setattr("backend.config.settings.api_port", 8200)
    monkeypatch.setattr("backend.config.settings.sandbox_host", None)

    code = "import gymnasium as gym\nenv = gym.make('CartPole-v1')\npass"
    gen = CodeGenerator(_make_provider(code))
    plan = _make_rl_plan()

    with patch.object(CodeGenerator, "_query_lessons", return_value=[]):
        asyncio.get_event_loop().run_until_complete(
            gen.generate_training_script("mission-cfgdefault", plan)
        )

    cfg_path = tmp_path / "missions" / "mission-cfgdefault" / "checkpoints" / "train_config.json"
    assert cfg_path.exists()
    cfg = json.loads(cfg_path.read_text())
    assert cfg["algorithm"] == "PPO"
    assert cfg["env_kwargs"] == {}


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


def test_build_user_prompt_rl_includes_iteration(tmp_path, monkeypatch):
    """Telemetry POST in the template must include the current_iteration value."""
    monkeypatch.setattr("backend.config.settings.data_path", str(tmp_path))
    monkeypatch.setattr("backend.config.settings.api_port", 8200)
    monkeypatch.setattr("backend.config.settings.sandbox_host", None)

    gen = CodeGenerator(_make_provider())
    plan = _make_rl_plan()
    prompt = gen._build_user_prompt("rl", "test-id", plan, str(tmp_path / "ckpt"), current_iteration=3)

    assert '"iteration": 3' in prompt


def test_build_user_prompt_rl_iteration_defaults_to_zero(tmp_path, monkeypatch):
    """When current_iteration is omitted the template renders iteration 0."""
    monkeypatch.setattr("backend.config.settings.data_path", str(tmp_path))
    monkeypatch.setattr("backend.config.settings.api_port", 8200)
    monkeypatch.setattr("backend.config.settings.sandbox_host", None)

    gen = CodeGenerator(_make_provider())
    plan = _make_rl_plan()
    prompt = gen._build_user_prompt("rl", "test-id", plan, str(tmp_path / "ckpt"))

    assert '"iteration": 0' in prompt


def test_build_user_prompt_rl_callback_init_loads_best_score(tmp_path, monkeypatch):
    """__init__ must load _best_reward from best_score.txt, not initialize to -inf."""
    monkeypatch.setattr("backend.config.settings.data_path", str(tmp_path))
    monkeypatch.setattr("backend.config.settings.api_port", 8200)
    monkeypatch.setattr("backend.config.settings.sandbox_host", None)

    gen = CodeGenerator(_make_provider())
    plan = _make_rl_plan()
    prompt = gen._build_user_prompt("rl", "test-id", plan, str(tmp_path / "ckpt"))

    # __init__ must read from best_score.txt so warm-start score is preserved
    assert "best_score.txt" in prompt
    # hasattr lazy-init pattern must NOT be present (it's defeated by any __init__ that sets _best_reward)
    assert "not hasattr" not in prompt
    # callback only posts mean_reward — goal metric is measured by post-iteration eval
    assert '"name": "mean_reward"' in prompt


def test_build_user_prompt_rl_callback_only_posts_mean_reward(tmp_path, monkeypatch):
    """Callback must NOT post goal-metric telemetry — that's handled by post-iteration eval."""
    monkeypatch.setattr("backend.config.settings.data_path", str(tmp_path))
    monkeypatch.setattr("backend.config.settings.api_port", 8200)
    monkeypatch.setattr("backend.config.settings.sandbox_host", None)

    gen = CodeGenerator(_make_provider())
    plan = _make_rl_plan(env_id="Tetris-v0")
    plan["target_metric"] = {"lines_cleared": 20}
    prompt = gen._build_user_prompt("rl", "test-id", plan, str(tmp_path / "ckpt"))

    # Callback posts mean_reward only — no per-episode goal metric buffer
    assert '"name": "mean_reward"' in prompt
    assert "_ep_metric_buf" not in prompt


def test_build_user_prompt_injects_tetris_setup(tmp_path, monkeypatch):
    """Tetris-v0 env must have registration preamble in the prompt."""
    monkeypatch.setattr("backend.config.settings.data_path", str(tmp_path))
    monkeypatch.setattr("backend.config.settings.api_port", 8200)
    monkeypatch.setattr("backend.config.settings.sandbox_host", None)

    gen = CodeGenerator(_make_provider())
    plan = _make_rl_plan(env_id="Tetris-v0")
    prompt = gen._build_user_prompt("rl", "test-id", plan, str(tmp_path / "ckpt"))

    assert "tetris_env" in prompt
    assert "_register_tetris" in prompt


def test_build_user_prompt_tetris_no_snake_setup(tmp_path, monkeypatch):
    """Tetris-v0 prompt must NOT include the Snake setup preamble."""
    monkeypatch.setattr("backend.config.settings.data_path", str(tmp_path))
    monkeypatch.setattr("backend.config.settings.api_port", 8200)
    monkeypatch.setattr("backend.config.settings.sandbox_host", None)

    gen = CodeGenerator(_make_provider())
    plan = _make_rl_plan(env_id="Tetris-v0")
    prompt = gen._build_user_prompt("rl", "test-id", plan, str(tmp_path / "ckpt"))

    assert "snake_env" not in prompt


def test_generate_training_script_injects_tetris_preamble(tmp_path, monkeypatch):
    """Post-generation fallback must inject Tetris registration if missing from LLM output."""
    monkeypatch.setattr("backend.config.settings.data_path", str(tmp_path))
    monkeypatch.setattr("backend.config.settings.api_port", 8200)
    monkeypatch.setattr("backend.config.settings.sandbox_host", None)

    # LLM returns script without registration preamble
    code = "import gymnasium as gym\nenv = gym.make('Tetris-v0')\n"
    gen = CodeGenerator(_make_provider(code))
    plan = _make_rl_plan(env_id="Tetris-v0")

    with patch.object(CodeGenerator, "_query_lessons", return_value=[]):
        path = asyncio.get_event_loop().run_until_complete(
            gen.generate_training_script("mission-tetris", plan)
        )

    content = open(path).read()
    assert "tetris_env" in content
    assert "_register_tetris" in content


# ── target_reward early-stop threshold ───────────────────────────────────────

def test_target_reward_uses_value_when_target_is_mean_reward(tmp_path, monkeypatch):
    """When target metric IS mean_reward, its value is used as the early-stop threshold."""
    monkeypatch.setattr("backend.config.settings.data_path", str(tmp_path))
    monkeypatch.setattr("backend.config.settings.api_port", 8200)
    monkeypatch.setattr("backend.config.settings.sandbox_host", None)

    gen = CodeGenerator(_make_provider())
    plan = _make_rl_plan(target_metric={"mean_reward": 475})
    prompt = gen._build_user_prompt("rl", "test-id", plan, str(tmp_path / "ckpt"))
    assert "475" in prompt


def test_target_reward_uses_9999_when_target_is_custom_metric(tmp_path, monkeypatch):
    """When target metric is NOT mean_reward (e.g. food_eaten=20), early-stop uses 9999
    so training never bails out early on mean_reward — the full timestep budget runs."""
    monkeypatch.setattr("backend.config.settings.data_path", str(tmp_path))
    monkeypatch.setattr("backend.config.settings.api_port", 8200)
    monkeypatch.setattr("backend.config.settings.sandbox_host", None)

    gen = CodeGenerator(_make_provider())
    plan = _make_rl_plan(target_metric={"food_eaten": 20})
    prompt = gen._build_user_prompt("rl", "test-id", plan, str(tmp_path / "ckpt"))
    assert "mean_reward >= 9999" in prompt
    # The food_eaten target value (20) must NOT be used as the threshold
    assert "mean_reward >= 20:" not in prompt


# ── Actor-Critic prompt path ──────────────────────────────────────────────────

def _make_actor_critic_plan(**overrides) -> dict:
    base = {
        "task_type": "rl",
        "trainer_type": "actor_critic",
        "algorithm": "ActorCritic",
        "env_id": "Tetris-v0",
        "target_metric": {"lines_cleared": 20},
        "hyperparameters": {"learning_rate": 0.0001, "episodes": 5000},
    }
    base.update(overrides)
    return base


def test_build_user_prompt_actor_critic_route(tmp_path, monkeypatch):
    """trainer_type=actor_critic routes to _ACTOR_CRITIC_CONTRACT, not SB3 template."""
    monkeypatch.setattr("backend.config.settings.data_path", str(tmp_path))
    monkeypatch.setattr("backend.config.settings.api_port", 8200)
    monkeypatch.setattr("backend.config.settings.sandbox_host", None)

    gen = CodeGenerator(_make_provider())
    plan = _make_actor_critic_plan()
    prompt = gen._build_user_prompt("rl", "test-id", plan, str(tmp_path / "ckpt"))

    assert "get_next_states" in prompt
    assert "actor_critic" in prompt
    assert "ActorCriticNet" in prompt


def test_build_user_prompt_actor_critic_contains_env_id(tmp_path, monkeypatch):
    monkeypatch.setattr("backend.config.settings.data_path", str(tmp_path))
    monkeypatch.setattr("backend.config.settings.api_port", 8200)
    monkeypatch.setattr("backend.config.settings.sandbox_host", None)

    gen = CodeGenerator(_make_provider())
    plan = _make_actor_critic_plan()
    prompt = gen._build_user_prompt("rl", "test-id", plan, str(tmp_path / "ckpt"))

    assert "Tetris-v0" in prompt


def test_build_user_prompt_actor_critic_contains_telemetry(tmp_path, monkeypatch):
    monkeypatch.setattr("backend.config.settings.data_path", str(tmp_path))
    monkeypatch.setattr("backend.config.settings.api_port", 8200)
    monkeypatch.setattr("backend.config.settings.sandbox_host", None)

    gen = CodeGenerator(_make_provider())
    plan = _make_actor_critic_plan()
    prompt = gen._build_user_prompt("rl", "test-id", plan, str(tmp_path / "ckpt"))

    assert "POST" in prompt
    assert "best_model" in prompt


def test_build_user_prompt_actor_critic_contains_hyperparameters(tmp_path, monkeypatch):
    monkeypatch.setattr("backend.config.settings.data_path", str(tmp_path))
    monkeypatch.setattr("backend.config.settings.api_port", 8200)
    monkeypatch.setattr("backend.config.settings.sandbox_host", None)

    gen = CodeGenerator(_make_provider())
    plan = _make_actor_critic_plan(hyperparameters={"learning_rate": 0.00025, "episodes": 8000})
    prompt = gen._build_user_prompt("rl", "test-id", plan, str(tmp_path / "ckpt"))

    assert "0.00025" in prompt
    assert "8000" in prompt


def test_build_user_prompt_actor_critic_contains_iteration(tmp_path, monkeypatch):
    monkeypatch.setattr("backend.config.settings.data_path", str(tmp_path))
    monkeypatch.setattr("backend.config.settings.api_port", 8200)
    monkeypatch.setattr("backend.config.settings.sandbox_host", None)

    gen = CodeGenerator(_make_provider())
    plan = _make_actor_critic_plan()
    prompt = gen._build_user_prompt("rl", "test-id", plan, str(tmp_path / "ckpt"), current_iteration=4)

    assert "4" in prompt


def test_build_user_prompt_no_trainer_type_uses_sb3(tmp_path, monkeypatch):
    """Plans without trainer_type still route to SB3 template."""
    monkeypatch.setattr("backend.config.settings.data_path", str(tmp_path))
    monkeypatch.setattr("backend.config.settings.api_port", 8200)
    monkeypatch.setattr("backend.config.settings.sandbox_host", None)

    gen = CodeGenerator(_make_provider())
    plan = _make_rl_plan(env_id="CartPole-v1")
    prompt = gen._build_user_prompt("rl", "test-id", plan, str(tmp_path / "ckpt"))

    assert "PPO" in prompt
    assert "get_next_states" not in prompt
