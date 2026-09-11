"""Unit tests for Game 2048 Lookahead DQN and Policy Explainability."""
import os
import tempfile
import torch
import numpy as np
import gymnasium as gym
from unittest.mock import MagicMock

from envs.actor_critic_net import ActorCriticNet, Game2048ValueNet
from envs.game2048_env import Game2048Env, register as register_2048
from backend.agent.code_generator import CodeGenerator
from backend.routers.play import _run_episode_actor_critic, _run_episode


def test_actor_critic_net_flexibility():
    """ActorCriticNet supports default 4-dim and arbitrary input_dim."""
    net_default = ActorCriticNet()
    assert net_default.input_dim == 4
    x4 = torch.randn(2, 4)
    out4 = net_default(x4)
    assert out4.shape == (2, 1)

    net16 = ActorCriticNet(input_dim=16)
    assert net16.input_dim == 16
    x16 = torch.randn(3, 16)
    out16 = net16(x16)
    assert out16.shape == (3, 1)


def test_game2048_value_net():
    """Game2048ValueNet evaluates 16-element observation vectors into scalar values."""
    net = Game2048ValueNet(input_dim=16)
    x = torch.randn(4, 16)
    out = net(x)
    assert out.shape == (4, 1)


import pytest
from unittest.mock import AsyncMock


@pytest.mark.asyncio
async def test_code_generator_2048_lookahead():
    """CodeGenerator produces specialized Lookahead DQN script for Game2048-v0."""
    provider = AsyncMock()
    provider.generate = AsyncMock(return_value="print('generated')")
    gen = CodeGenerator(provider=provider)
    plan = {
        "mission_id": "test_2048_lookahead",
        "task_type": "rl",
        "env_id": "Game2048-v0",
        "trainer_type": "lookahead_dqn",
        "algorithm": "LOOKAHEAD_DQN",
        "hyperparameters": {
            "learning_rate": 0.0003,
            "total_timesteps": 10000,
            "replay_buffer_size": 1000,
            "batch_size": 32,
            "gamma": 0.99,
        },
        "target_metric": {"score": 4096},
    }
    await gen.generate_training_script("test_2048_lookahead", plan)
    prompt = provider.generate.call_args[0][0]
    user_content = prompt[-1].content if isinstance(prompt, list) else str(prompt)
    assert "Game2048ValueNet" in user_content
    assert "get_next_states()" in user_content
    assert "lookahead_dqn" in user_content
    assert "best_model.pth" in user_content
    assert "Game2048-v0" in user_content


def test_run_episode_actor_critic_2048():
    """_run_episode_actor_critic streams 2048 frames with Q-values and action probabilities."""
    register_2048()
    env = gym.make("Game2048-v0", max_steps=10)
    model = Game2048ValueNet(input_dim=16)
    model.eval()

    frames, reward = _run_episode_actor_critic(model, env)
    assert len(frames) > 0
    f0 = frames[0]
    assert f0["type"] == "frame"
    assert len(f0["grid"]) == 16
    assert "score" in f0
    assert "max_tile" in f0
    assert "q_values" in f0
    assert "action_probs" in f0
    assert "selected_action" in f0
    assert f0["selected_action"] in ("UP", "DOWN", "LEFT", "RIGHT")


def test_run_episode_explainability_sb3():
    """_run_episode computes Q-values, action probabilities, and entropy for SB3 DQN models."""
    register_2048()
    env = gym.make("Game2048-v0", max_steps=5)

    mock_model = MagicMock()
    mock_model.predict.return_value = (2, None)
    mock_q_net = MagicMock()
    # 4 actions for 2048: UP, DOWN, LEFT, RIGHT
    mock_q_net.return_value = torch.tensor([[10.0, 5.0, 25.0, 2.0]])
    mock_model.q_net = mock_q_net
    mock_model.device = "cpu"

    frames, reward = _run_episode(mock_model, env)
    assert len(frames) > 0
    f0 = frames[0]
    assert f0["selected_action"] == "LEFT"
    assert "q_values" in f0
    assert f0["q_values"]["LEFT"] == 25.0
    assert "action_probs" in f0
    assert f0["action_probs"]["LEFT"] > f0["action_probs"]["DOWN"]
    assert "entropy" in f0
    assert isinstance(f0["entropy"], float)
