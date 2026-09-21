"""Unit tests for MinAtar Asterix (envs/minatar_asterix_env.py)."""
import numpy as np
import gymnasium as gym
import pytest

from envs.minatar_asterix_env import MinAtarAsterixEnv, register


@pytest.fixture(autouse=True)
def ensure_registered():
    register()


def test_asterix_spaces():
    env = MinAtarAsterixEnv()
    assert env.action_space.n == 6
    assert env.observation_space.shape == (400,)
    assert env.observation_space.dtype == np.float32


def test_asterix_reset():
    env = MinAtarAsterixEnv()
    obs, info = env.reset(seed=42)
    assert obs.shape == (400,)
    assert info["score"] == 0.0
    assert info["gold_collected"] == 0
    assert env._player_x == 5
    assert env._player_y == 5


def test_asterix_movement_bounds():
    env = MinAtarAsterixEnv()
    env.reset(seed=0)
    env._entities = [None] * 8
    env.step(2)  # UP
    assert env._player_y == 4
    env._player_y = 1
    env.step(2)
    assert env._player_y == 1
    env._player_y = 8
    env.step(4)  # DOWN
    assert env._player_y == 8
    env._player_x = 0
    env.step(1)  # LEFT
    assert env._player_x == 0
    env._player_x = 9
    env.step(3)  # RIGHT
    assert env._player_x == 9


def test_asterix_gold_pickup():
    env = MinAtarAsterixEnv()
    env.reset(seed=0)
    env._entities = [[5, 5, True, True]] + [None] * 7
    env._spawn_timer = 99
    env._move_timer = 99
    obs, reward, terminated, truncated, info = env.step(0)
    assert reward == 1.0
    assert not terminated
    assert info["gold_collected"] == 1
    assert env._entities[0] is None


def test_asterix_enemy_kills():
    env = MinAtarAsterixEnv(death_penalty=-2.0)
    env.reset(seed=0)
    env._entities = [[5, 5, True, False]] + [None] * 7
    env._spawn_timer = 99
    env._move_timer = 99
    obs, reward, terminated, truncated, info = env.step(0)
    assert terminated
    assert reward == -2.0


def test_asterix_gym_make():
    env = gym.make("MinAtar-Asterix-v0")
    obs, info = env.reset(seed=1)
    assert obs.shape == (400,)
    env.close()


def test_asterix_recipe_hps():
    from backend.agent.code_generator import _resolve_hyperparams
    result = _resolve_hyperparams("MinAtar-Asterix-v0", {}, algorithm="DQN")
    assert result["learning_rate"] == 0.00025
    assert result["buffer_size"] == 100000


def test_register_asterix_alias():
    from envs.register import register_for_env_id
    register_for_env_id("asterix")
    env = gym.make("MinAtar-Asterix-v0")
    env.close()
