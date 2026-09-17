"""
Unit tests for MultiTurnAgentGym-v0 (Agent Gym multi-turn interactive tool RL environment).
"""
from __future__ import annotations

import os
import pytest
import numpy as np
import gymnasium as gym
from unittest.mock import MagicMock, patch

from envs.agent_gym import (
    register,
    AgentToolGym,
    ACTION_FINISH,
    ACTION_QUERY_DB,
    ACTION_SEND_EMAIL,
    ACTION_SEARCH_KB,
    ACTION_CALCULATE,
    ACTION_NAMES,
    SCENARIOS,
)
from backend.agent.code_generator import CodeGenerator, _load_recipe_for_env
from backend.config import settings


@pytest.fixture(autouse=True)
def ensure_registered():
    register()


def test_agent_gym_registration():
    """Verify MultiTurnAgentGym-v0 and AgentGym-v0 are registered in Gymnasium."""
    env = gym.make("MultiTurnAgentGym-v0")
    assert isinstance(env.unwrapped, AgentToolGym)
    assert env.action_space.n == 8
    assert env.observation_space.shape == (32,)

    env_alias = gym.make("AgentGym-v0")
    assert isinstance(env_alias.unwrapped, AgentToolGym)


def test_agent_gym_reset_and_scenarios():
    """Verify reset returns 32D observation and scenario metadata."""
    env = gym.make("MultiTurnAgentGym-v0", scenario_idx=0)
    obs, info = env.reset()

    assert isinstance(obs, np.ndarray)
    assert obs.shape == (32,)
    assert obs.dtype == np.float32
    assert np.all(obs >= 0.0) and np.all(obs <= 1.0)

    assert info["scenario_id"] == "order_delay_notify"
    assert info["required_steps"] == 3
    assert "Check status of order" in info["query"]


def test_agent_gym_successful_sequence():
    """Verify following the required tool sequence achieves full completion reward."""
    env = gym.make("MultiTurnAgentGym-v0", scenario_idx=0)
    obs, info = env.reset()

    # Step 1: QUERY_DATABASE
    obs, reward1, term, trunc, info = env.step(ACTION_QUERY_DB)
    assert not term and not trunc
    assert reward1 > 0.0
    assert info["sequence_progress"] == 1

    # Step 2: SEND_EMAIL
    obs, reward2, term, trunc, info = env.step(ACTION_SEND_EMAIL)
    assert not term and not trunc
    assert reward2 > 0.0
    assert info["sequence_progress"] == 2

    # Step 3: FINISH_TASK
    obs, reward3, term, trunc, info = env.step(ACTION_FINISH)
    assert term
    assert not trunc
    assert reward3 >= 2.0  # completion bonus
    assert info["task_success"] == 1.0
    assert info["sequence_progress"] == 3


def test_agent_gym_premature_finish():
    """Verify calling finish before completing required steps incurs heavy penalty and terminates."""
    env = gym.make("MultiTurnAgentGym-v0", scenario_idx=0)
    obs, info = env.reset()

    # Finish on step 1 without performing query or email
    obs, reward, term, trunc, info = env.step(ACTION_FINISH)
    assert term
    assert reward < 0.0
    assert info["task_success"] == 0.0
    assert info["sequence_progress"] == 0


def test_agent_gym_out_of_order_and_repetition():
    """Verify out-of-order and repeated identical tool calls are penalized."""
    env = gym.make("MultiTurnAgentGym-v0", scenario_idx=0)
    obs, info = env.reset()

    # Incorrect tool: CALCULATE on step 1
    obs, reward1, term, trunc, info = env.step(ACTION_CALCULATE)
    assert not term
    assert reward1 < 0.0
    assert obs[17] == 0.0  # last_action_success = 0.0

    # Repeat CALCULATE consecutively -> repetition penalty
    obs, reward2, term, trunc, info = env.step(ACTION_CALCULATE)
    assert not term
    assert reward2 < reward1  # additional repetition penalty
    assert obs[27] == 1.0     # repetition flag active


def test_agent_gym_text_and_json_actions():
    """Verify string action names and JSON payloads are parsed accurately."""
    env = gym.make("MultiTurnAgentGym-v0", scenario_idx=1)
    obs, info = env.reset()
    # Scenario 1 requires: SEARCH_KB, CALCULATE, FINISH

    # String name
    obs, r1, term, trunc, info = env.step("search_kb")
    assert r1 > 0.0
    assert info["sequence_progress"] == 1

    # JSON dict string
    obs, r2, term, trunc, info = env.step('{"tool": "calculate", "args": {"val": 100}}')
    assert r2 > 0.0
    assert info["sequence_progress"] == 2

    # Dict object
    obs, r3, term, trunc, info = env.step({"action": "finish_task"})
    assert term
    assert info["task_success"] == 1.0


def test_agent_gym_max_steps_truncation():
    """Verify episode truncates when exceeding max_steps."""
    env = gym.make("MultiTurnAgentGym-v0", max_steps=3, scenario_idx=0)
    env.reset()

    env.step(ACTION_SEARCH_KB)
    env.step(ACTION_SEARCH_KB)
    obs, reward, term, trunc, info = env.step(ACTION_SEARCH_KB)

    assert trunc
    assert not term
    assert info["task_success"] == 0.0


def test_agent_gym_recipe_and_codegen():
    """Verify recipe loading and setup preamble injection for MultiTurnAgentGym-v0."""
    recipe = _load_recipe_for_env("MultiTurnAgentGym-v0", "PPO")
    assert recipe["name"] == "agent_gym_ppo_v1"
    assert recipe["task_type"] == "RL"
    assert recipe["target_metric"] == {"task_success": 0.85}

    generator = CodeGenerator(provider=MagicMock())
    plan = {
        "task_type": "rl",
        "algorithm": "PPO",
        "env_id": "MultiTurnAgentGym-v0",
        "hyperparameters": {
            "learning_rate": 0.0003,
            "total_timesteps": 10000,
        },
        "target_metric": {"task_success": 0.85},
    }

    user_prompt = generator._build_user_prompt(
        task_type="rl",
        mission_id="test-agent-gym-123",
        plan=plan,
        checkpoint_dir="checkpoints",
    )
    assert "_register_agent_gym" in user_prompt
    assert "envs.agent_gym" in user_prompt
