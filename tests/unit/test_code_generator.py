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


def test_build_user_prompt_rl_includes_lr_schedule_helper(tmp_path, monkeypatch):
    """The _linear_schedule helper and its opt-in guard are always emitted,
    even when the recipe/plan does not set lr_schedule."""
    monkeypatch.setattr("backend.config.settings.data_path", str(tmp_path))
    monkeypatch.setattr("backend.config.settings.api_port", 8200)
    monkeypatch.setattr("backend.config.settings.sandbox_host", None)

    gen = CodeGenerator(_make_provider())
    plan = _make_rl_plan()
    prompt = gen._build_user_prompt("rl", "test-id", plan, str(tmp_path / "ckpt"))

    assert "_linear_schedule" in prompt
    assert '_hp.get("lr_schedule") == "linear"' in prompt


def test_build_user_prompt_rl_hyperparameters_without_lr_schedule_stay_constant(tmp_path, monkeypatch):
    """When the plan/recipe omits lr_schedule, _hp has no such key and the
    generated script's guard leaves learning_rate as a constant scalar."""
    monkeypatch.setattr("backend.config.settings.data_path", str(tmp_path))
    monkeypatch.setattr("backend.config.settings.api_port", 8200)
    monkeypatch.setattr("backend.config.settings.sandbox_host", None)

    gen = CodeGenerator(_make_provider())
    plan = _make_rl_plan(hyperparameters={"learning_rate": 0.0003, "n_steps": 2048})
    prompt = gen._build_user_prompt("rl", "test-id", plan, str(tmp_path / "ckpt"))

    hp_block = prompt.split("_hp = ")[1].split("_filtered")[0]
    assert "lr_schedule" not in hp_block


def test_build_user_prompt_rl_hyperparameters_with_lr_schedule_linear(tmp_path, monkeypatch):
    """When the plan/recipe sets lr_schedule: linear, it is hardcoded into _hp
    so the opt-in guard activates at runtime."""
    monkeypatch.setattr("backend.config.settings.data_path", str(tmp_path))
    monkeypatch.setattr("backend.config.settings.api_port", 8200)
    monkeypatch.setattr("backend.config.settings.sandbox_host", None)

    gen = CodeGenerator(_make_provider())
    plan = _make_rl_plan(hyperparameters={"learning_rate": 0.0003, "lr_schedule": "linear"})
    prompt = gen._build_user_prompt("rl", "test-id", plan, str(tmp_path / "ckpt"))

    hp_block = prompt.split("_hp = ")[1].split("_filtered")[0]
    assert "lr_schedule" in hp_block
    assert "linear" in hp_block


def test_resolve_hyperparams_snake_recipe_sets_lr_schedule_linear():
    """snake_ppo_v1.yaml opts into the linear LR schedule by default."""
    from backend.agent.code_generator import _resolve_hyperparams
    result = _resolve_hyperparams("Snake-v0", {"learning_rate": 0.0003})
    assert result.get("lr_schedule") == "linear"


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


# ── hardcode → recipe wiring ──────────────────────────────────────────────────

def test_build_user_prompt_rl_telemetry_interval_from_hyperparameters(tmp_path, monkeypatch):
    """telemetry_interval in hyperparameters overrides the default 2048."""
    monkeypatch.setattr("backend.config.settings.data_path", str(tmp_path))
    monkeypatch.setattr("backend.config.settings.api_port", 8200)
    monkeypatch.setattr("backend.config.settings.sandbox_host", None)

    gen = CodeGenerator(_make_provider())
    plan = _make_rl_plan(hyperparameters={"learning_rate": 0.001, "telemetry_interval": 4096})
    prompt = gen._build_user_prompt("rl", "test-id", plan, str(tmp_path / "ckpt"))

    assert "n_calls % 4096 == 0" in prompt


def test_build_user_prompt_rl_total_timesteps_from_hyperparameters(tmp_path, monkeypatch):
    """total_timesteps in hyperparameters is substituted into model.learn() call."""
    monkeypatch.setattr("backend.config.settings.data_path", str(tmp_path))
    monkeypatch.setattr("backend.config.settings.api_port", 8200)
    monkeypatch.setattr("backend.config.settings.sandbox_host", None)

    gen = CodeGenerator(_make_provider())
    plan = _make_rl_plan(hyperparameters={"learning_rate": 0.001, "total_timesteps": 500000})
    prompt = gen._build_user_prompt("rl", "test-id", plan, str(tmp_path / "ckpt"))

    assert "total_timesteps=500000" in prompt


def test_build_user_prompt_actor_critic_ac_telemetry_interval_from_hyperparameters(tmp_path, monkeypatch):
    """ac_telemetry_interval in hyperparameters replaces the hardcoded 50-episode window."""
    monkeypatch.setattr("backend.config.settings.data_path", str(tmp_path))
    monkeypatch.setattr("backend.config.settings.api_port", 8200)
    monkeypatch.setattr("backend.config.settings.sandbox_host", None)

    gen = CodeGenerator(_make_provider())
    plan = _make_actor_critic_plan(hyperparameters={"learning_rate": 0.0001, "episodes": 5000,
                                                     "ac_telemetry_interval": 100})
    prompt = gen._build_user_prompt("rl", "test-id", plan, str(tmp_path / "ckpt"))

    assert "% 100 == 0" in prompt
    assert "ep_rewards[-100:]" in prompt


def test_build_user_prompt_actor_critic_uses_timestep_loop(tmp_path, monkeypatch):
    """AC template loops on total_steps < total_timesteps, not on episode count."""
    monkeypatch.setattr("backend.config.settings.data_path", str(tmp_path))
    monkeypatch.setattr("backend.config.settings.api_port", 8200)
    monkeypatch.setattr("backend.config.settings.sandbox_host", None)

    gen = CodeGenerator(_make_provider())
    plan = _make_actor_critic_plan(hyperparameters={"learning_rate": 0.0001, "total_timesteps": 500000})
    prompt = gen._build_user_prompt("rl", "test-id", plan, str(tmp_path / "ckpt"))

    assert "total_steps < 500000" in prompt
    assert "total_steps +=" in prompt
    assert "for episode in range" not in prompt


def test_build_user_prompt_actor_critic_ac_telemetry_interval_default(tmp_path, monkeypatch):
    """ac_telemetry_interval defaults to 50 when not in hyperparameters."""
    monkeypatch.setattr("backend.config.settings.data_path", str(tmp_path))
    monkeypatch.setattr("backend.config.settings.api_port", 8200)
    monkeypatch.setattr("backend.config.settings.sandbox_host", None)

    gen = CodeGenerator(_make_provider())
    plan = _make_actor_critic_plan()
    prompt = gen._build_user_prompt("rl", "test-id", plan, str(tmp_path / "ckpt"))

    assert "% 50 == 0" in prompt
    assert "ep_rewards[-50:]" in prompt


def test_build_user_prompt_actor_critic_includes_gym_make(tmp_path, monkeypatch):
    """AC template skeleton must include env = gym.make(...) before the training loop."""
    monkeypatch.setattr("backend.config.settings.data_path", str(tmp_path))
    monkeypatch.setattr("backend.config.settings.api_port", 8200)
    monkeypatch.setattr("backend.config.settings.sandbox_host", None)

    gen = CodeGenerator(_make_provider())
    plan = _make_actor_critic_plan()
    prompt = gen._build_user_prompt("rl", "test-id", plan, str(tmp_path / "ckpt"))

    assert 'gym.make("Tetris-v0")' in prompt
    assert "from envs.actor_critic_net import ActorCriticNet" in prompt


def test_build_user_prompt_tetris_uses_ac_template_without_trainer_type(tmp_path, monkeypatch):
    """Tetris-v0 routes to AC template even when plan omits trainer_type (recipe fallback)."""
    monkeypatch.setattr("backend.config.settings.data_path", str(tmp_path))
    monkeypatch.setattr("backend.config.settings.api_port", 8200)
    monkeypatch.setattr("backend.config.settings.sandbox_host", None)

    gen = CodeGenerator(_make_provider())
    plan = {
        "task_type": "rl",
        # trainer_type intentionally omitted — simulates LLM planner output
        "algorithm": "PPO",
        "env_id": "Tetris-v0",
        "target_metric": {"lines_cleared": 100},
        "hyperparameters": {"learning_rate": 0.0003},
    }
    prompt = gen._build_user_prompt("rl", "test-id", plan, str(tmp_path / "ckpt"))

    assert "get_next_states" in prompt
    assert "ActorCriticNet" in prompt
    assert "total_steps" in prompt


# ── Lookahead-DQN/PPO/A2C prompt path ───────────────────────────────────────────
# Real incident: vanilla SB3 DQN/PPO/A2C structurally cannot compete on
# Tetris-v0 (confirmed live: 130+ DQN pivots, dozens of PPO/A2C pivots, all
# plateaued at lines_cleared≈0-1, vs. Actor-Critic hitting 394 in 3
# iterations). LoopStateMachine now routes an explicit DQN/PPO/A2C request on
# Tetris-v0 to the matching lookahead_dqn/lookahead_ppo/lookahead_a2c
# trainer_type instead of vanilla SB3 — these tests cover code_generator's
# routing to the corresponding contract template.

def _make_lookahead_plan(trainer_type: str, **overrides) -> dict:
    base = {
        "task_type": "rl",
        "trainer_type": trainer_type,
        "algorithm": trainer_type.split("_")[1].upper(),
        "env_id": "Tetris-v0",
        "target_metric": {"lines_cleared": 200},
        "hyperparameters": {"learning_rate": 0.0005},
    }
    base.update(overrides)
    return base


def _prompt_for(tmp_path, monkeypatch, plan) -> str:
    monkeypatch.setattr("backend.config.settings.data_path", str(tmp_path))
    monkeypatch.setattr("backend.config.settings.api_port", 8200)
    monkeypatch.setattr("backend.config.settings.sandbox_host", None)
    gen = CodeGenerator(_make_provider())
    return gen._build_user_prompt("rl", "test-id", plan, str(tmp_path / "ckpt"))


def test_lookahead_dqn_routes_to_dqn_contract(tmp_path, monkeypatch):
    prompt = _prompt_for(tmp_path, monkeypatch, _make_lookahead_plan("lookahead_dqn"))
    assert "get_next_states" in prompt
    assert "ActorCriticNet" in prompt
    assert "target_model" in prompt          # DQN-defining: target network
    assert '"lookahead_dqn"' in prompt


def test_lookahead_dqn_target_network_synced_periodically(tmp_path, monkeypatch):
    prompt = _prompt_for(tmp_path, monkeypatch, _make_lookahead_plan("lookahead_dqn"))
    assert "target_model.load_state_dict(model.state_dict())" in prompt


def test_lookahead_ppo_routes_to_ppo_contract(tmp_path, monkeypatch):
    prompt = _prompt_for(tmp_path, monkeypatch, _make_lookahead_plan("lookahead_ppo"))
    assert "get_next_states" in prompt
    assert "ActorCriticNet" in prompt
    assert "GAE" in prompt or "gae" in prompt.lower()   # PPO-defining: advantage estimation
    assert '"lookahead_ppo"' in prompt


def test_lookahead_ppo_has_multi_epoch_clipped_updates(tmp_path, monkeypatch):
    prompt = _prompt_for(tmp_path, monkeypatch, _make_lookahead_plan("lookahead_ppo"))
    assert "for _epoch in range(" in prompt
    assert "clip" in prompt.lower()
    assert "torch.randperm" in prompt   # minibatch shuffling — multi-epoch reuse of one batch


def test_lookahead_a2c_routes_to_a2c_contract(tmp_path, monkeypatch):
    prompt = _prompt_for(tmp_path, monkeypatch, _make_lookahead_plan("lookahead_a2c"))
    assert "get_next_states" in prompt
    assert "ActorCriticNet" in prompt
    assert '"lookahead_a2c"' in prompt


def test_lookahead_a2c_has_no_replay_buffer_or_multi_epoch(tmp_path, monkeypatch):
    """A2C-defining: single synchronous update per short rollout, no replay
    buffer (unlike DQN) and no multi-epoch reuse (unlike PPO)."""
    prompt = _prompt_for(tmp_path, monkeypatch, _make_lookahead_plan("lookahead_a2c"))
    assert "collections.deque" not in prompt
    assert "n_epochs" not in prompt
    assert "target_model" not in prompt


def test_lookahead_dqn_no_gae_or_multi_epoch(tmp_path, monkeypatch):
    """DQN-defining: off-policy replay + target network, no on-policy GAE/epochs."""
    prompt = _prompt_for(tmp_path, monkeypatch, _make_lookahead_plan("lookahead_dqn"))
    assert "collections.deque" in prompt
    assert "n_epochs" not in prompt
    assert "advantages" not in prompt


def test_lookahead_none_of_the_three_have_stochastic_policy_head():
    """All three deliberately share ActorCriticNet's pure value-function shape
    — no separate policy network/log-prob machinery in any contract."""
    from backend.agent.code_generator import (
        _LOOKAHEAD_DQN_CONTRACT, _LOOKAHEAD_PPO_CONTRACT, _LOOKAHEAD_A2C_CONTRACT,
    )
    for contract in (_LOOKAHEAD_DQN_CONTRACT, _LOOKAHEAD_PPO_CONTRACT, _LOOKAHEAD_A2C_CONTRACT):
        assert "log_prob" not in contract
        assert "Categorical" not in contract
        assert "ActorCriticNet()" in contract


# ── obs/next_obs sliced to ActorCriticNet's 4-dim input ─────────────────────────
# Real incident: live-verified. ActorCriticNet has a hardcoded 4-dim input, but
# after the Tetris-v0 observation-space fix (piece identity added for DQN/PPO/A2C)
# env.step()/reset() now return 18-dim observations. Every trainer that stores
# raw obs/next_obs in its replay buffer/rollout and later feeds them into
# ActorCriticNet for a TD target crashed live with RuntimeError: mat1 and mat2
# shapes cannot be multiplied (64x18 and 4x64) — confirmed on mission 06e4af4e's
# first Lookahead-DQN run. This affects ALL FOUR lookahead-family contracts
# (the original actor_critic one too), since all four share ActorCriticNet.
# Fixed by slicing to [:4] at the buffer/rollout append site, since the
# observation-space fix deliberately kept the original 4 board-quality features
# as a prefix of the new 18-dim vector.

def test_actor_critic_slices_obs_to_4_dims(tmp_path, monkeypatch):
    prompt = _prompt_for(tmp_path, monkeypatch, _make_actor_critic_plan())
    assert "obs[:4]" in prompt
    assert "next_obs[:4]" in prompt


def test_lookahead_dqn_slices_obs_to_4_dims(tmp_path, monkeypatch):
    prompt = _prompt_for(tmp_path, monkeypatch, _make_lookahead_plan("lookahead_dqn"))
    assert "obs[:4]" in prompt
    assert "next_obs[:4]" in prompt


def test_lookahead_ppo_slices_obs_to_4_dims(tmp_path, monkeypatch):
    prompt = _prompt_for(tmp_path, monkeypatch, _make_lookahead_plan("lookahead_ppo"))
    assert "obs[:4]" in prompt
    assert "next_obs[:4]" in prompt


def test_lookahead_a2c_slices_obs_to_4_dims(tmp_path, monkeypatch):
    prompt = _prompt_for(tmp_path, monkeypatch, _make_lookahead_plan("lookahead_a2c"))
    assert "obs[:4]" in prompt
    assert "next_obs[:4]" in prompt


def test_lookahead_dqn_runtime_shapes_never_mismatch():
    """End-to-end runtime simulation against the real TetrisEnv (post
    observation-space fix, 18-dim) and real ActorCriticNet — the exact
    reproduction of the live crash, run 200 real steps to confirm no shape
    mismatch reaches model()/target_model()."""
    import random
    import collections
    import torch
    import torch.nn as nn
    from envs.tetris_env import register
    import gymnasium as gym
    from envs.actor_critic_net import ActorCriticNet

    register()
    env = gym.make("Tetris-v0")
    model = ActorCriticNet()
    target_model = ActorCriticNet()
    target_model.load_state_dict(model.state_dict())
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)

    BUFFER = collections.deque(maxlen=1000)
    obs, _ = env.reset()
    assert obs.shape == (18,)  # confirms this test exercises the post-fix observation shape

    for _ in range(200):
        next_states = env.unwrapped.get_next_states()
        if not next_states:
            action = 0
        elif random.random() < 0.3:
            action = random.choice(list(next_states.keys()))
        else:
            with torch.no_grad():
                action = max(
                    next_states,
                    key=lambda a: model(torch.tensor(next_states[a], dtype=torch.float32).unsqueeze(0)).item(),
                )
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        BUFFER.append((obs[:4], action, reward, next_obs[:4], float(done)))
        obs = next_obs
        if done:
            obs, _ = env.reset()

        if len(BUFFER) >= 32:
            batch = random.sample(BUFFER, 32)
            s, _, r, ns, d = zip(*batch)
            s = torch.tensor(s, dtype=torch.float32)
            ns = torch.tensor(ns, dtype=torch.float32)
            r = torch.tensor(r, dtype=torch.float32).unsqueeze(1)
            d = torch.tensor(d, dtype=torch.float32).unsqueeze(1)
            with torch.no_grad():
                td_target = r + 0.99 * target_model(ns) * (1 - d)
            loss = nn.MSELoss()(model(s), td_target)
            optimizer.zero_grad(); loss.backward(); optimizer.step()

    env.close()


# ── duplicate telemetry POST guard (PPO/A2C only) ───────────────────────────────
# Real incident: live-observed duplicate mean_reward telemetry entries (same
# step, same value, back to back) for a PPO mission. Root cause: PPO/A2C's
# outer training loop runs once per ROLLOUT, not once per episode — an
# episode can span multiple rollouts once the agent survives a while, so
# `episode` may not change between outer-loop passes. The telemetry condition
# only checked `episode % ac_telemetry_interval == 0`, which stays true
# across every subsequent pass until a NEW episode finally completes,
# re-posting the identical payload each time. DQN/actor_critic can't hit
# this — their outer loop runs once per COMPLETED episode by construction,
# so episode always increments exactly once per pass there.

def test_lookahead_ppo_has_duplicate_post_guard(tmp_path, monkeypatch):
    prompt = _prompt_for(tmp_path, monkeypatch, _make_lookahead_plan("lookahead_ppo"))
    assert "_last_posted_episode" in prompt
    assert "episode != _last_posted_episode" in prompt


def test_lookahead_a2c_has_duplicate_post_guard(tmp_path, monkeypatch):
    prompt = _prompt_for(tmp_path, monkeypatch, _make_lookahead_plan("lookahead_a2c"))
    assert "_last_posted_episode" in prompt
    assert "episode != _last_posted_episode" in prompt


def test_lookahead_dqn_has_no_duplicate_post_guard(tmp_path, monkeypatch):
    """DQN's outer loop runs once per completed episode by construction —
    episode always increments each pass, so this guard isn't needed there."""
    prompt = _prompt_for(tmp_path, monkeypatch, _make_lookahead_plan("lookahead_dqn"))
    assert "_last_posted_episode" not in prompt


def test_actor_critic_has_no_duplicate_post_guard(tmp_path, monkeypatch):
    prompt = _prompt_for(tmp_path, monkeypatch, _make_actor_critic_plan())
    assert "_last_posted_episode" not in prompt


def test_duplicate_post_guard_prevents_repeat_posts_when_episode_frozen():
    """Direct reproduction of the live incident: episode count stays frozen
    across multiple outer-loop passes (no episode boundary crossed), and the
    guard must still post exactly once per unique episode count."""
    ac_telemetry_interval = 50
    episode = 50
    ep_rewards = [1.0] * 60
    _last_posted_episode = -1
    posts = []

    for outer_pass in range(5):
        if outer_pass == 2:
            episode = 100  # only advances once, on pass 2 — mimics an episode finally completing
        if (
            len(ep_rewards) >= ac_telemetry_interval
            and episode % ac_telemetry_interval == 0
            and episode > 0
            and episode != _last_posted_episode
        ):
            _last_posted_episode = episode
            posts.append(episode)

    assert posts == [50, 100]  # exactly one post per unique episode count, no duplicates


# ── epsilon persistence across iterations ───────────────────────────────────────
# Real incident: every fresh iteration's script reset epsilon to 1.0, and
# epsilon_decay (0.9995/episode) is far too slow relative to episodes-per-
# iteration to meaningfully decay within one run — e.g. by episode 300,
# epsilon is still ~0.86 (86% of actions still uniformly random). Since the
# model IS warm-started across iterations but epsilon was NOT, training-time
# mean_reward stayed flat and unremarkable (dominated by near-random
# exploration noise) across iterations even as the model's true learned
# quality — visible only in the fully-greedy, zero-exploration eval rollout
# — improved dramatically (mission 1ed39807: mean_reward ~31 the whole way
# through, but eval lines_cleared jumped from 3 to 313). Fixed by persisting
# epsilon to a checkpoint-dir file, loaded on startup and written back on
# every decay step, so it actually continues decaying across iterations
# instead of resetting.

def test_actor_critic_persists_epsilon(tmp_path, monkeypatch):
    prompt = _prompt_for(tmp_path, monkeypatch, _make_actor_critic_plan())
    assert "_eps_path" in prompt
    assert 'os.path.exists(_eps_path) else 1.0' in prompt
    assert 'open(_eps_path, "w").write(str(epsilon))' in prompt


def test_lookahead_dqn_persists_epsilon(tmp_path, monkeypatch):
    prompt = _prompt_for(tmp_path, monkeypatch, _make_lookahead_plan("lookahead_dqn"))
    assert "_eps_path" in prompt
    assert 'os.path.exists(_eps_path) else 1.0' in prompt
    assert 'open(_eps_path, "w").write(str(epsilon))' in prompt


def test_lookahead_ppo_persists_epsilon(tmp_path, monkeypatch):
    prompt = _prompt_for(tmp_path, monkeypatch, _make_lookahead_plan("lookahead_ppo"))
    assert "_eps_path" in prompt
    assert 'os.path.exists(_eps_path) else 1.0' in prompt
    assert 'open(_eps_path, "w").write(str(epsilon))' in prompt


def test_lookahead_a2c_persists_epsilon(tmp_path, monkeypatch):
    prompt = _prompt_for(tmp_path, monkeypatch, _make_lookahead_plan("lookahead_a2c"))
    assert "_eps_path" in prompt
    assert 'os.path.exists(_eps_path) else 1.0' in prompt
    assert 'open(_eps_path, "w").write(str(epsilon))' in prompt


def test_epsilon_persistence_carries_forward_across_simulated_iterations(tmp_path):
    """Direct reproduction of the fix's core behavior: a fresh 'script'
    resuming from a saved epsilon.txt continues decay instead of resetting."""
    import os as _os
    eps_path = str(tmp_path / "epsilon.txt")

    epsilon = float(open(eps_path).read().strip()) if _os.path.exists(eps_path) else 1.0
    assert epsilon == 1.0  # iteration 0: no file yet, starts fresh
    for _ in range(300):
        epsilon = max(0.01, epsilon * 0.9995)
    open(eps_path, "w").write(str(epsilon))
    iteration_0_end = epsilon

    # Simulate a brand new script execution (iteration 1) reading the same path
    epsilon = float(open(eps_path).read().strip()) if _os.path.exists(eps_path) else 1.0
    assert epsilon == iteration_0_end  # continues, does not reset to 1.0
    assert epsilon < 1.0


# ── recipe-driven env_kwargs and hyperparams tests ────────────────────────────

def test_resolve_env_kwargs_snake_reads_from_recipe():
    """Snake-v0 gets obs_type and max_steps from snake_ppo_v1.yaml when plan omits them."""
    from backend.agent.code_generator import _resolve_env_kwargs
    result = _resolve_env_kwargs("Snake-v0", None)
    assert result["obs_type"] == "features"
    assert result["max_steps"] == 2000


def test_resolve_env_kwargs_plan_overrides_recipe():
    """Explicit plan env_kwargs override recipe defaults."""
    from backend.agent.code_generator import _resolve_env_kwargs
    result = _resolve_env_kwargs("Snake-v0", {"obs_type": "grid", "max_steps": 500})
    assert result["obs_type"] == "grid"
    assert result["max_steps"] == 500


def test_resolve_env_kwargs_non_snake_unchanged():
    """Envs with no recipe are returned as-is."""
    from backend.agent.code_generator import _resolve_env_kwargs
    result = _resolve_env_kwargs("CartPole-v1", {"foo": "bar"})
    assert result == {"foo": "bar"}
    assert "obs_type" not in result


def test_resolve_hyperparams_snake_uses_recipe_total_timesteps():
    """Snake-v0 gets total_timesteps=3_000_000 from snake_ppo_v1.yaml when plan omits it."""
    from backend.agent.code_generator import _resolve_hyperparams
    result = _resolve_hyperparams("Snake-v0", {"learning_rate": 0.0003})
    assert result["total_timesteps"] == 3_000_000


def test_resolve_hyperparams_plan_overrides_recipe():
    """Explicit plan total_timesteps overrides recipe value."""
    from backend.agent.code_generator import _resolve_hyperparams
    result = _resolve_hyperparams("Snake-v0", {"total_timesteps": 500_000})
    assert result["total_timesteps"] == 500_000


def test_build_user_prompt_snake_uses_recipe_env_kwargs(tmp_path, monkeypatch):
    """Snake-v0 prompt includes obs_type='features' and max_steps=2000 from recipe."""
    monkeypatch.setattr("backend.config.settings.data_path", str(tmp_path))
    monkeypatch.setattr("backend.config.settings.api_port", 8200)
    monkeypatch.setattr("backend.config.settings.sandbox_host", None)

    gen = CodeGenerator(_make_provider())
    plan = {
        "task_type": "rl",
        "algorithm": "PPO",
        "env_id": "Snake-v0",
        "hyperparameters": {"learning_rate": 0.0003},
        "target_metric": {"food_eaten": 20},
    }
    prompt = gen._build_user_prompt("rl", "test-id", plan, str(tmp_path / "ckpt"))
    assert "obs_type='features'" in prompt
    assert "max_steps=2000" in prompt
    assert "total_timesteps=3000000" in prompt


# ── mlx_lora template ─────────────────────────────────────────────────────────

def test_build_user_prompt_mlx_lora_contains_base_model(tmp_path, monkeypatch):
    monkeypatch.setattr("backend.config.settings.data_path", str(tmp_path))
    monkeypatch.setattr("backend.config.settings.api_port", 8200)
    monkeypatch.setattr("backend.config.settings.sandbox_host", None)

    gen = CodeGenerator(_make_provider())
    plan = {
        "task_type": "mlx_lora",
        "algorithm": "lora",
        "dataset": {
            "train": "data/datasets/train.jsonl",
            "valid": "data/datasets/valid.jsonl",
        },
        "hyperparameters": {"base_model": "mlx-community/gemma-3-12b-it-4bit"},
        "target_metric": {"eval_loss": 1.0},
    }
    prompt = gen._build_user_prompt("mlx_lora", "test-id", plan, str(tmp_path / "ckpt"))
    assert "mlx-community/gemma-3-12b-it-4bit" in prompt
    assert "mlx_lm" in prompt


def test_build_user_prompt_mlx_lora_contains_dataset_paths(tmp_path, monkeypatch):
    monkeypatch.setattr("backend.config.settings.data_path", str(tmp_path))
    monkeypatch.setattr("backend.config.settings.api_port", 8200)
    monkeypatch.setattr("backend.config.settings.sandbox_host", None)

    gen = CodeGenerator(_make_provider())
    plan = {
        "task_type": "mlx_lora",
        "algorithm": "lora",
        "dataset": {
            "train": "/my/data/train.jsonl",
            "valid": "/my/data/valid.jsonl",
        },
        "hyperparameters": {"base_model": "mlx-community/gemma-3-12b-it-4bit"},
        "target_metric": {"eval_loss": 1.0},
    }
    prompt = gen._build_user_prompt("mlx_lora", "test-id", plan, str(tmp_path / "ckpt"))
    assert "/my/data/train.jsonl" in prompt
    assert "/my/data/valid.jsonl" in prompt


def test_build_user_prompt_mlx_lora_uses_recipe_defaults(tmp_path, monkeypatch):
    """mlx_lora prompt fills missing HP from mlx_lora_v1.yaml recipe."""
    monkeypatch.setattr("backend.config.settings.data_path", str(tmp_path))
    monkeypatch.setattr("backend.config.settings.api_port", 8200)
    monkeypatch.setattr("backend.config.settings.sandbox_host", None)

    gen = CodeGenerator(_make_provider())
    plan = {
        "task_type": "mlx_lora",
        "algorithm": "lora",
        "dataset": {"train": "train.jsonl", "valid": "valid.jsonl"},
        "hyperparameters": {},  # let recipe fill everything
        "target_metric": {"eval_loss": 1.0},
    }
    prompt = gen._build_user_prompt("mlx_lora", "test-id", plan, str(tmp_path / "ckpt"))
    assert "gemma-3-12b-it-4bit" in prompt
    assert "600" in prompt  # iters from recipe


def test_resolve_hyperparams_mlx_lora_uses_recipe_iters():
    """mlx_lora task gets iters=600 from mlx_lora_v1.yaml when plan omits it."""
    from backend.agent.code_generator import _resolve_hyperparams
    result = _resolve_hyperparams("mlx_lora", {})
    assert result["iters"] == 600


# ── dpo template ──────────────────────────────────────────────────────────────

def test_build_user_prompt_dpo_wraps_existing_script_not_reimplemented(tmp_path, monkeypatch):
    """The DPO template must instruct wrapping dpo_train.py, not reimplementing DPO math."""
    monkeypatch.setattr("backend.config.settings.data_path", str(tmp_path))
    monkeypatch.setattr("backend.config.settings.api_port", 8200)
    monkeypatch.setattr("backend.config.settings.sandbox_host", None)

    gen = CodeGenerator(_make_provider())
    plan = {
        "task_type": "dpo",
        "hyperparameters": {},
        "target_metric": {"pass_rate": 0.9},
    }
    prompt = gen._build_user_prompt("dpo", "test-id", plan, str(tmp_path / "ckpt"))
    assert "dpo_train.py" in prompt
    assert "do NOT reimplement" in prompt


def test_build_user_prompt_dpo_uses_recipe_defaults(tmp_path, monkeypatch):
    """dpo prompt fills missing HP from ensemble_dpo_v1.yaml recipe."""
    monkeypatch.setattr("backend.config.settings.data_path", str(tmp_path))
    monkeypatch.setattr("backend.config.settings.api_port", 8200)
    monkeypatch.setattr("backend.config.settings.sandbox_host", None)

    gen = CodeGenerator(_make_provider())
    plan = {
        "task_type": "dpo",
        "hyperparameters": {},   # let recipe fill everything
        "target_metric": {"pass_rate": 0.9},
    }
    prompt = gen._build_user_prompt("dpo", "test-id", plan, str(tmp_path / "ckpt"))
    assert "gemma-3-12b-it-4bit" in prompt
    assert "/Users/kewang/finetune" in prompt
    assert "finetune-env/bin/python" in prompt


def test_build_user_prompt_dpo_routing_only_flag_default_true(tmp_path, monkeypatch):
    monkeypatch.setattr("backend.config.settings.data_path", str(tmp_path))
    monkeypatch.setattr("backend.config.settings.api_port", 8200)
    monkeypatch.setattr("backend.config.settings.sandbox_host", None)

    gen = CodeGenerator(_make_provider())
    plan = {"task_type": "dpo", "hyperparameters": {}, "target_metric": {"pass_rate": 0.9}}
    prompt = gen._build_user_prompt("dpo", "test-id", plan, str(tmp_path / "ckpt"))
    assert "--routing-only" in prompt


def test_build_user_prompt_dpo_no_telemetry_network_calls(tmp_path, monkeypatch):
    """Astra tails the remote log itself — the wrapper script must not import
    requests or POST telemetry (no network deps needed on the Mac Mini)."""
    monkeypatch.setattr("backend.config.settings.data_path", str(tmp_path))
    monkeypatch.setattr("backend.config.settings.api_port", 8200)
    monkeypatch.setattr("backend.config.settings.sandbox_host", None)

    gen = CodeGenerator(_make_provider())
    plan = {"task_type": "dpo", "hyperparameters": {}, "target_metric": {"pass_rate": 0.9}}
    prompt = gen._build_user_prompt("dpo", "test-id", plan, str(tmp_path / "ckpt"))
    assert "Do NOT import subprocess or requests" in prompt
    assert "Telemetry URL" not in prompt


def test_build_user_prompt_dpo_uses_execv_not_subprocess(tmp_path, monkeypatch):
    """Must os.execv (replaces process image, same pid) instead of subprocess.run
    (forks a child with a different pid astra can't track) — a subprocess.run
    wrapper orphans the real training process if the wrapper ever dies, since
    astra only tracks the wrapper's pid."""
    monkeypatch.setattr("backend.config.settings.data_path", str(tmp_path))
    monkeypatch.setattr("backend.config.settings.api_port", 8200)
    monkeypatch.setattr("backend.config.settings.sandbox_host", None)

    gen = CodeGenerator(_make_provider())
    plan = {"task_type": "dpo", "hyperparameters": {}, "target_metric": {"pass_rate": 0.9}}
    prompt = gen._build_user_prompt("dpo", "test-id", plan, str(tmp_path / "ckpt"))
    assert "os.execv(" in prompt
    assert "subprocess.run(" not in prompt


def test_build_user_prompt_dpo_sets_cwd_to_finetune_dir(tmp_path, monkeypatch):
    """--prompt-template and dpo_train.py's own hardcoded eval-cases path are
    both relative to the process cwd, not the script's own location — must
    os.chdir(finetune_dir) before exec or those loads silently fail."""
    monkeypatch.setattr("backend.config.settings.data_path", str(tmp_path))
    monkeypatch.setattr("backend.config.settings.api_port", 8200)
    monkeypatch.setattr("backend.config.settings.sandbox_host", None)

    gen = CodeGenerator(_make_provider())
    plan = {"task_type": "dpo", "hyperparameters": {}, "target_metric": {"pass_rate": 0.9}}
    prompt = gen._build_user_prompt("dpo", "test-id", plan, str(tmp_path / "ckpt"))
    assert 'os.chdir("/Users/kewang/finetune")' in prompt


def test_resolve_hyperparams_dpo_uses_recipe_beta():
    """dpo task gets beta=0.1 from ensemble_dpo_v1.yaml when plan omits it."""
    from backend.agent.code_generator import _resolve_hyperparams
    result = _resolve_hyperparams("dpo", {})
    assert result["beta"] == 0.1


def test_resolve_hyperparams_dpo_ignores_plan_override():
    """The recipe must be authoritative for dpo — unlike every other task
    type, a plan-provided hyperparameter must NOT override it. Confirmed via
    a real incident: a pivot proposed learning_rate=0.001 (a plausible RL-style
    value) which silently overrode the recipe's 5e-7, collapsing a DPO run's
    pass_rate from a 62% baseline to 0% within 50 steps. PIVOT_SYSTEM doesn't
    document dpo/grpo hyperparameter ranges at all, so this must be a hard
    code-level guarantee, not just a prompt-level convention."""
    from backend.agent.code_generator import _resolve_hyperparams
    result = _resolve_hyperparams("dpo", {"learning_rate": 0.001, "num_layers": 5})
    assert result["learning_rate"] == 5e-7
    assert result["num_layers"] == 8


def test_resolve_hyperparams_grpo_ignores_plan_override():
    from backend.agent.code_generator import _resolve_hyperparams
    result = _resolve_hyperparams("grpo", {"learning_rate": 0.001})
    assert result["learning_rate"] != 0.001


# ── grpo template ─────────────────────────────────────────────────────────────

def test_build_user_prompt_grpo_wraps_existing_script_not_reimplemented(tmp_path, monkeypatch):
    """The GRPO template must instruct wrapping grpo_train.py, not reimplementing GRPO math."""
    monkeypatch.setattr("backend.config.settings.data_path", str(tmp_path))
    monkeypatch.setattr("backend.config.settings.api_port", 8200)
    monkeypatch.setattr("backend.config.settings.sandbox_host", None)

    gen = CodeGenerator(_make_provider())
    plan = {
        "task_type": "grpo",
        "hyperparameters": {},
        "target_metric": {"pass_rate": 0.9},
    }
    prompt = gen._build_user_prompt("grpo", "test-id", plan, str(tmp_path / "ckpt"))
    assert "grpo_train.py" in prompt
    assert "do NOT reimplement" in prompt


def test_build_user_prompt_grpo_uses_recipe_defaults(tmp_path, monkeypatch):
    """grpo prompt fills missing HP from ensemble_grpo_v1.yaml recipe."""
    monkeypatch.setattr("backend.config.settings.data_path", str(tmp_path))
    monkeypatch.setattr("backend.config.settings.api_port", 8200)
    monkeypatch.setattr("backend.config.settings.sandbox_host", None)

    gen = CodeGenerator(_make_provider())
    plan = {
        "task_type": "grpo",
        "hyperparameters": {},
        "target_metric": {"pass_rate": 0.9},
    }
    prompt = gen._build_user_prompt("grpo", "test-id", plan, str(tmp_path / "ckpt"))
    assert "gemma-3-12b-it-4bit" in prompt
    assert "iters=100" in prompt   # iters from recipe (documented best practice, not the script's raw 300 default)


def test_resolve_hyperparams_grpo_uses_recipe_clip_epsilon():
    """grpo task gets clip_epsilon=0.1 from ensemble_grpo_v1.yaml when plan omits it."""
    from backend.agent.code_generator import _resolve_hyperparams
    result = _resolve_hyperparams("grpo", {})
    assert result["clip_epsilon"] == 0.1


def test_resolve_hyperparams_grpo_num_layers_matches_warm_start_adapter():
    """num_layers MUST be 8 (not grpo_train.py's raw --num-layers default of 16) —
    all current warm-start adapters (grpo_v9_min/best, retrain_best) are 8-layer;
    a mismatch is a LoRA weight shape error at load time, not a soft warning."""
    from backend.agent.code_generator import _resolve_hyperparams
    result = _resolve_hyperparams("grpo", {})
    assert result["num_layers"] == 8


def test_resolve_hyperparams_grpo_uses_current_best_warm_start_adapter():
    """Warm-start must be the documented current-best grpo_v9_min/best, not
    grpo_train.py's generic (and mutable) DEFAULT_ADAPTER constant."""
    from backend.agent.code_generator import _resolve_hyperparams
    result = _resolve_hyperparams("grpo", {})
    assert result["adapter"] == "adapters/grpo_v9_min/best"


def test_resolve_hyperparams_grpo_uses_documented_iters_not_raw_default():
    """docs/FINETUNE.md: 'use 100 — gains are front-loaded', not the raw
    --iters default of 300."""
    from backend.agent.code_generator import _resolve_hyperparams
    result = _resolve_hyperparams("grpo", {})
    assert result["iters"] == 100


def test_resolve_hyperparams_grpo_uses_documented_max_tokens_not_raw_default():
    """docs/FINETUNE.md: 'use 96 — skill name appears in first ~80 tokens,
    halves step time', not the raw --max-tokens default of 256."""
    from backend.agent.code_generator import _resolve_hyperparams
    result = _resolve_hyperparams("grpo", {})
    assert result["max_tokens"] == 96


def test_resolve_hyperparams_grpo_email_weight_excludes_email_cases():
    """All runs since grpo_v8_min use --email-weight 0 ('GRPO-impenetrable'),
    not the raw --email-weight default of 3."""
    from backend.agent.code_generator import _resolve_hyperparams
    result = _resolve_hyperparams("grpo", {})
    assert result["email_weight"] == 0


def test_build_user_prompt_dpo_includes_save_pairs(tmp_path, monkeypatch):
    """--save-pairs must always be passed so a future pivot can reuse collected
    pairs via --load-pairs instead of re-running the slowest phase (~30-60 min)."""
    monkeypatch.setattr("backend.config.settings.data_path", str(tmp_path))
    monkeypatch.setattr("backend.config.settings.api_port", 8200)
    monkeypatch.setattr("backend.config.settings.sandbox_host", None)

    gen = CodeGenerator(_make_provider())
    plan = {"task_type": "dpo", "hyperparameters": {}, "target_metric": {"pass_rate": 0.9}}
    prompt = gen._build_user_prompt("dpo", "test-id12345", plan, str(tmp_path / "ckpt"))
    assert "--save-pairs" in prompt
    assert "/Users/kewang/finetune/logs/astra_test-id1_pairs.jsonl" in prompt


def test_build_user_prompt_grpo_no_telemetry_network_calls(tmp_path, monkeypatch):
    """Same no-network-calls requirement as dpo — astra tails the log itself."""
    monkeypatch.setattr("backend.config.settings.data_path", str(tmp_path))
    monkeypatch.setattr("backend.config.settings.api_port", 8200)
    monkeypatch.setattr("backend.config.settings.sandbox_host", None)

    gen = CodeGenerator(_make_provider())
    plan = {"task_type": "grpo", "hyperparameters": {}, "target_metric": {"pass_rate": 0.9}}
    prompt = gen._build_user_prompt("grpo", "test-id", plan, str(tmp_path / "ckpt"))
    assert "Do NOT import subprocess or requests" in prompt
    assert "Telemetry URL" not in prompt


def test_build_user_prompt_grpo_uses_execv_not_subprocess(tmp_path, monkeypatch):
    """Same os.execv requirement as dpo — no orphan-prone subprocess.run fork."""
    monkeypatch.setattr("backend.config.settings.data_path", str(tmp_path))
    monkeypatch.setattr("backend.config.settings.api_port", 8200)
    monkeypatch.setattr("backend.config.settings.sandbox_host", None)

    gen = CodeGenerator(_make_provider())
    plan = {"task_type": "grpo", "hyperparameters": {}, "target_metric": {"pass_rate": 0.9}}
    prompt = gen._build_user_prompt("grpo", "test-id", plan, str(tmp_path / "ckpt"))
    assert "os.execv(" in prompt
    assert "subprocess.run(" not in prompt


def test_build_user_prompt_grpo_sets_cwd_to_finetune_dir(tmp_path, monkeypatch):
    """Same cwd requirement as dpo — grpo_train.py also resolves --prompt-template
    and its hardcoded eval-cases path relative to the process cwd."""
    monkeypatch.setattr("backend.config.settings.data_path", str(tmp_path))
    monkeypatch.setattr("backend.config.settings.api_port", 8200)
    monkeypatch.setattr("backend.config.settings.sandbox_host", None)

    gen = CodeGenerator(_make_provider())
    plan = {"task_type": "grpo", "hyperparameters": {}, "target_metric": {"pass_rate": 0.9}}
    prompt = gen._build_user_prompt("grpo", "test-id", plan, str(tmp_path / "ckpt"))
    assert 'os.chdir("/Users/kewang/finetune")' in prompt


# ── manifest checkpoint patterns ──────────────────────────────────────────────

def test_manifest_checkpoint_pattern_dpo_grpo():
    from backend.services.manifest_generator import _CHECKPOINT_PATTERNS
    assert _CHECKPOINT_PATTERNS["dpo"] == "checkpoints/best/"
    assert _CHECKPOINT_PATTERNS["grpo"] == "checkpoints/best/"


# ── sandbox_host scoping regression (must not leak to non-finetune task types) ─

def test_generate_training_script_rl_uses_local_checkpoint_dir_even_with_sandbox_host(tmp_path, monkeypatch):
    """A configured sandbox_host (for dpo/grpo pinning) must NOT redirect RL
    missions' checkpoint_dir to the remote path — only dpo/grpo do that."""
    monkeypatch.setattr("backend.config.settings.data_path", str(tmp_path))
    monkeypatch.setattr("backend.config.settings.api_port", 8200)
    monkeypatch.setattr("backend.config.settings.sandbox_host", "mac-mini.local")
    monkeypatch.setattr("backend.config.settings.sandbox_data_path", "/tmp/astra")

    code = "import gymnasium as gym\nenv = gym.make('CartPole-v1')\n"
    gen = CodeGenerator(_make_provider(code))

    with patch.object(CodeGenerator, "_query_lessons", return_value=[]):
        path = asyncio.get_event_loop().run_until_complete(
            gen.generate_training_script("mission-rl-with-host", _make_rl_plan())
        )

    # Local data_path, not settings.sandbox_data_path
    assert str(tmp_path) in path
    assert "/tmp/astra" not in path


def test_generate_training_script_dpo_uses_finetune_adapters_dir_with_sandbox_host(tmp_path, monkeypatch):
    """dpo (and grpo) save under finetune_dir/adapters/, matching where
    ensemble/finetune's own manual workflow keeps adapters (grpo_v<N>_min/,
    retrain_best/, ...) — not the generic sandbox_data_path checkpoints path."""
    monkeypatch.setattr("backend.config.settings.data_path", str(tmp_path))
    monkeypatch.setattr("backend.config.settings.api_port", 8200)
    monkeypatch.setattr("backend.config.settings.sandbox_host", "mac-mini.local")
    monkeypatch.setattr("backend.config.settings.sandbox_data_path", "/tmp/astra")

    provider = _make_provider("print('dpo wrapper')\n")
    gen = CodeGenerator(provider)
    plan = {"task_type": "dpo", "hyperparameters": {}, "target_metric": {"pass_rate": 0.9}}

    with patch.object(CodeGenerator, "_query_lessons", return_value=[]):
        path = asyncio.get_event_loop().run_until_complete(
            gen.generate_training_script("mission-dpo-with-host", plan)
        )

    assert str(tmp_path) in path   # train.py itself is still written locally
    # The prompt sent to the LLM must carry the finetune_dir/adapters/ checkpoint_dir
    sent_messages = provider.generate.call_args[0][0]
    prompt_text = "\n".join(m.content for m in sent_messages)
    assert "/Users/kewang/finetune/adapters/astra_mission-" in prompt_text
    assert "/tmp/astra" not in prompt_text


def test_finetune_checkpoint_dir_uses_recipe_finetune_dir():
    from backend.agent.code_generator import finetune_checkpoint_dir
    result = finetune_checkpoint_dir("dpo", {"hyperparameters": {}}, "abc12345-full-uuid")
    assert result == "/Users/kewang/finetune/adapters/astra_abc12345"


def test_detect_backend_ignores_sandbox_host():
    """_detect_backend() must not treat a configured sandbox_host as the
    general default — only _FINETUNE_REMOTE_TASK_TYPES force ssh, in launch()."""
    from backend.sandbox.manager import _detect_backend
    with patch("backend.sandbox.manager.settings.sandbox_host", "mac-mini.local"):
        assert _detect_backend() != "ssh"


def test_resolve_hyperparams_mlx_lora_plan_overrides_iters():
    """Explicit plan iters overrides recipe."""
    from backend.agent.code_generator import _resolve_hyperparams
    result = _resolve_hyperparams("mlx_lora", {"iters": 100})
    assert result["iters"] == 100


# ── _inject_curriculum ────────────────────────────────────────────────────────

_SAMPLE_PHASES = [
    {"grid_h": 8,  "grid_w": 8,  "food_target": 15, "timesteps": 300000},
    {"grid_h": 12, "grid_w": 12, "food_target": 40, "timesteps": 700000},
    {"grid_h": 16, "grid_w": 16, "food_target": 100, "timesteps": 2000000},
]

_SAMPLE_ENV_KWARGS = {
    "obs_type": "features",
    "food_reward": 20.0,
    "death_penalty": -10.0,
    "max_steps": 2000,
}

def _make_injectable_script() -> str:
    """Minimal script with the two hooks _inject_curriculum looks for."""
    return (
        "import os\n"
        "import gymnasium as gym\n"
        "import numpy as np\n"
        "import logging\n"
        "env = gym.make('Snake-v0')\n"
        "class CustomCallback(BaseCallback):\n"
        "    def __init__(self, verbose=0):\n"
        "        super().__init__(verbose=verbose)\n"
        "        try:\n"
        '            self._best_reward = float(open("ckpt/best_score.txt").read().strip())\n'
        "        except Exception:\n"
        '            self._best_reward = float("-inf")\n'
        "    def _on_step(self) -> bool:\n"
        "        if self.n_calls % 2048 == 0:\n"
        "            pass\n"
        "        return True\n"
        "callback = CustomCallback()\n"
        "model.learn(total_timesteps=3000000, callback=callback)\n"
        "model.save('ckpt/last_model')\n"
    )


def test_inject_curriculum_replaces_learn_call():
    code = _make_injectable_script()
    result = CodeGenerator._inject_curriculum(code, _SAMPLE_PHASES, "Snake-v0", _SAMPLE_ENV_KWARGS)
    assert "model.learn(total_timesteps=3000000, callback=callback)" not in result
    assert "_CURRICULUM_PHASES" in result
    assert "for _ph_idx, _ph in enumerate(_CURRICULUM_PHASES):" in result


def test_inject_curriculum_phases_list_present():
    code = _make_injectable_script()
    result = CodeGenerator._inject_curriculum(code, _SAMPLE_PHASES, "Snake-v0", _SAMPLE_ENV_KWARGS)
    assert '"grid_h": 8' in result
    assert '"food_target": 15' in result
    assert '"timesteps": 2000000' in result


def test_inject_curriculum_set_env_per_phase():
    code = _make_injectable_script()
    result = CodeGenerator._inject_curriculum(code, _SAMPLE_PHASES, "Snake-v0", _SAMPLE_ENV_KWARGS)
    assert "model.set_env(_ph_env)" in result


def test_inject_curriculum_grid_dims_come_from_phase():
    """grid_h/grid_w must be passed from the phase dict, not hardcoded from env_kwargs."""
    code = _make_injectable_script()
    result = CodeGenerator._inject_curriculum(code, _SAMPLE_PHASES, "Snake-v0", _SAMPLE_ENV_KWARGS)
    assert '_ph["grid_h"]' in result
    assert '_ph["grid_w"]' in result


def test_inject_curriculum_excludes_grid_dims_from_base_kwargs():
    """grid_h/grid_w must not appear in the base gym.make call (they come from phase)."""
    env_kw_with_grid = {**_SAMPLE_ENV_KWARGS, "grid_h": 16, "grid_w": 16}
    code = _make_injectable_script()
    result = CodeGenerator._inject_curriculum(code, _SAMPLE_PHASES, "Snake-v0", env_kw_with_grid)
    # The _ph_env = gym.make(...) line should not have literal grid_h=16
    for line in result.splitlines():
        if "_ph_env = gym.make" in line:
            assert "grid_h=16" not in line
            assert "grid_w=16" not in line


def test_inject_curriculum_adds_phase_best_metric_to_init():
    code = _make_injectable_script()
    result = CodeGenerator._inject_curriculum(code, _SAMPLE_PHASES, "Snake-v0", _SAMPLE_ENV_KWARGS)
    assert "self._phase_best_metric = 0" in result


def test_inject_curriculum_adds_food_tracking_to_on_step():
    code = _make_injectable_script()
    result = CodeGenerator._inject_curriculum(code, _SAMPLE_PHASES, "Snake-v0", _SAMPLE_ENV_KWARGS)
    assert 'self.locals.get("infos"' in result
    assert 'self.locals.get("dones"' in result
    assert "self._phase_best_metric" in result
    assert '"food_eaten" in _info' in result  # default metric_name


def test_inject_curriculum_reset_num_timesteps_only_on_first_phase():
    """reset_num_timesteps=True only for phase 0 so step counter is continuous."""
    code = _make_injectable_script()
    result = CodeGenerator._inject_curriculum(code, _SAMPLE_PHASES, "Snake-v0", _SAMPLE_ENV_KWARGS)
    assert "reset_num_timesteps=(_ph_idx == 0)" in result


# ── _inject_curriculum: Tetris-shaped phases (no grid dims) ────────────────────
# Real incident: this function was hardcoded entirely for Snake's phase shape.
# A recipe merge gave Tetris-v0 a curriculum for the first time
# (max_lines_cleared/max_iterations, no grid_h/grid_w — the board is fixed
# 20x10) and every phase transition crashed with KeyError: 'grid_h' because
# the old code unconditionally referenced _ph["grid_h"]/_ph["grid_w"].

_TETRIS_PHASES = [
    {"max_lines_cleared": 50, "max_iterations": 10000},
    {"max_lines_cleared": 100, "max_iterations": 20000},
    {"max_lines_cleared": 200, "max_iterations": 30000},
]


def test_inject_curriculum_tetris_no_grid_dims_in_gym_make():
    """The exact real-incident reproduction: no grid_h/grid_w key access at all."""
    code = _make_injectable_script()
    result = CodeGenerator._inject_curriculum(code, _TETRIS_PHASES, "Tetris-v0", {"max_steps": 1000})
    for line in result.splitlines():
        if "_ph_env = gym.make" in line:
            assert "grid_h" not in line
            assert "grid_w" not in line


def test_inject_curriculum_tetris_uses_max_iterations_as_duration():
    code = _make_injectable_script()
    result = CodeGenerator._inject_curriculum(code, _TETRIS_PHASES, "Tetris-v0", {"max_steps": 1000})
    assert 'model.learn(total_timesteps=_ph["max_iterations"]' in result


def test_inject_curriculum_tetris_uses_target_metric_name():
    code = _make_injectable_script()
    result = CodeGenerator._inject_curriculum(
        code, _TETRIS_PHASES, "Tetris-v0", {"max_steps": 1000}, metric_name="lines_cleared"
    )
    assert '"lines_cleared" in _info' in result
    assert '_info["lines_cleared"]' in result
    assert '"food_eaten"' not in result


def test_inject_curriculum_no_duration_key_skips_injection():
    """Neither 'timesteps' nor 'max_iterations' present — don't crash, just skip."""
    code = _make_injectable_script()
    bad_phases = [{"max_lines_cleared": 50}]
    result = CodeGenerator._inject_curriculum(code, bad_phases, "Tetris-v0", {})
    assert result == code  # unchanged — original single model.learn() call kept
    assert "model.learn(total_timesteps=3000000, callback=callback)" in result


def test_inject_curriculum_empty_phases_returns_unchanged():
    code = _make_injectable_script()
    result = CodeGenerator._inject_curriculum(code, [], "Tetris-v0", {})
    assert result == code


def test_generate_training_script_snake_injects_curriculum(tmp_path, monkeypatch):
    """End-to-end: Snake-v0 script gets curriculum injected after LLM generation."""
    monkeypatch.setattr("backend.config.settings.data_path", str(tmp_path))
    monkeypatch.setattr("backend.config.settings.api_port", 8200)
    monkeypatch.setattr("backend.config.settings.sandbox_host", None)

    llm_output = _make_injectable_script()
    gen = CodeGenerator(_make_provider(llm_output))
    plan = {
        "task_type": "rl",
        "algorithm": "PPO",
        "env_id": "Snake-v0",
        "hyperparameters": {"learning_rate": 0.0003},
        "target_metric": {"food_eaten": 100},
    }

    with patch.object(CodeGenerator, "_query_lessons", return_value=[]):
        path = asyncio.get_event_loop().run_until_complete(
            gen.generate_training_script("mission-curriculum", plan)
        )

    content = open(path).read()
    assert "_CURRICULUM_PHASES" in content
    assert "model.set_env(_ph_env)" in content
    assert "_phase_best_metric" in content


def test_generate_training_script_non_snake_no_curriculum(tmp_path, monkeypatch):
    """CartPole has no curriculum in its recipe — no injection."""
    monkeypatch.setattr("backend.config.settings.data_path", str(tmp_path))
    monkeypatch.setattr("backend.config.settings.api_port", 8200)
    monkeypatch.setattr("backend.config.settings.sandbox_host", None)

    llm_output = _make_injectable_script()
    gen = CodeGenerator(_make_provider(llm_output))
    plan = _make_rl_plan(env_id="CartPole-v1")

    with patch.object(CodeGenerator, "_query_lessons", return_value=[]):
        path = asyncio.get_event_loop().run_until_complete(
            gen.generate_training_script("mission-cartpole2", plan)
        )

    content = open(path).read()
    assert "_CURRICULUM_PHASES" not in content


# ── valid_algo_keys tests ──────────────────────────────────────────────────────

def test_valid_algo_keys_ppo():
    keys = CodeGenerator.valid_algo_keys("PPO")
    assert "ent_coef" in keys
    assert "vf_coef" in keys
    assert "buffer_size" not in keys


def test_valid_algo_keys_dqn():
    keys = CodeGenerator.valid_algo_keys("DQN")
    assert "buffer_size" in keys
    assert "exploration_fraction" in keys
    assert "ent_coef" not in keys
    assert "vf_coef" not in keys


def test_valid_algo_keys_dqn_excludes_optimize_memory_usage():
    """Real incident: optimize_memory_usage=True crashes SB3's ReplayBuffer
    (ValueError: incompatible with the default handle_timeout_termination=True,
    which isn't exposed here either) — reachable from a live pivot proposal,
    not just a recipe default, so it must never appear in the valid-key
    allowlist at all."""
    assert "optimize_memory_usage" not in CodeGenerator.valid_algo_keys("DQN")


def test_valid_algo_keys_case_insensitive():
    assert CodeGenerator.valid_algo_keys("dqn") == CodeGenerator.valid_algo_keys("DQN")


def test_valid_algo_keys_unknown_returns_empty():
    assert CodeGenerator.valid_algo_keys("UNKNOWN_ALGO") == set()
