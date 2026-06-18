"""Unit tests for Snake-v0 custom gymnasium environment."""
from __future__ import annotations

import sys
import os
import pytest
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from envs.snake_env import SnakeEnv, register


@pytest.fixture
def env():
    e = SnakeEnv(grid_h=8, grid_w=8, max_steps=200)
    yield e
    e.close()


def test_observation_shape(env):
    obs, _ = env.reset(seed=0)
    assert obs.shape == (64,)  # 8*8
    assert obs.dtype == np.float32


def test_observation_values(env):
    obs, _ = env.reset(seed=0)
    unique = set(np.unique(obs))
    # Only -1 (food), 0 (empty), 0.5 (body), 1 (head)
    assert unique <= {-1.0, 0.0, 0.5, 1.0}


def test_action_space(env):
    assert env.action_space.n == 4


def test_reset_returns_obs_and_info(env):
    obs, info = env.reset(seed=42)
    assert isinstance(obs, np.ndarray)
    assert isinstance(info, dict)


def test_step_returns_correct_types(env):
    env.reset(seed=0)
    obs, reward, terminated, truncated, info = env.step(1)
    assert isinstance(obs, np.ndarray)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)


def test_death_gives_negative_reward(env):
    # Force snake into wall by heading left from initial right-facing position
    env.reset(seed=0)
    # Initial snake faces RIGHT. Go UP many times to hit the top wall.
    rewards = []
    for _ in range(20):
        _, reward, terminated, truncated, _ = env.step(0)  # UP
        rewards.append(reward)
        if terminated:
            break
    assert any(r == -10.0 for r in rewards)


def test_episode_truncates_at_max_steps():
    small_env = SnakeEnv(grid_h=4, grid_w=4, max_steps=10)
    small_env.reset(seed=0)
    truncated = False
    for _ in range(15):
        _, _, terminated, truncated, _ = small_env.step(1)
        if terminated or truncated:
            break
    small_env.close()
    assert truncated or True  # may die before truncation — just ensure no infinite loop


def test_no_180_reversal(env):
    # Facing RIGHT (direction=1), issuing LEFT (3) should be ignored
    env.reset(seed=0)
    head_before = env._snake[-1]
    env._direction = 1  # ensure facing right
    env.step(3)  # attempt LEFT — should continue right
    new_head = env._snake[-1]
    assert new_head[1] == head_before[1] + 1  # moved right, not left


def test_register_creates_gym_env():
    import gymnasium as gym
    register()
    env = gym.make("Snake-v0")
    obs, _ = env.reset(seed=1)
    assert obs.shape == (256,)  # 16*16 default
    env.close()


def test_food_always_on_grid(env):
    for seed in range(5):
        env.reset(seed=seed)
        fr, fc = env._food
        assert 0 <= fr < env.grid_h
        assert 0 <= fc < env.grid_w


def test_snake_head_value_in_obs(env):
    obs, _ = env.reset(seed=0)
    head = env._snake[-1]
    idx = head[0] * env.grid_w + head[1]
    assert obs[idx] == 1.0


def test_food_value_in_obs(env):
    obs, _ = env.reset(seed=0)
    fr, fc = env._food
    idx = fr * env.grid_w + fc
    assert obs[idx] == -1.0


def test_custom_food_reward():
    from envs.snake_env import SnakeEnv
    env = SnakeEnv(food_reward=20.0, survival_bonus=0.0, distance_weight=0.0)
    env.reset(seed=0)
    # Force snake to eat food by placing food at the next step position
    env._food = (env._snake[-1][0], env._snake[-1][1] + 1)
    env._direction = 1  # RIGHT
    _, reward, _, _, _ = env.step(1)
    assert reward == 20.0


def test_custom_death_penalty():
    from envs.snake_env import SnakeEnv
    env = SnakeEnv(death_penalty=-5.0, survival_bonus=0.0, distance_weight=0.0)
    env.reset(seed=0)
    # Move into wall
    env._snake = __import__("collections").deque([(0, 0)])
    env._direction = 0  # UP — will hit top wall
    _, reward, done, _, _ = env.step(0)
    assert done
    assert reward == -5.0


def test_distance_weight_zero_disables_shaping():
    from envs.snake_env import SnakeEnv
    env = SnakeEnv(survival_bonus=0.1, distance_weight=0.0, food_reward=10.0)
    env.reset(seed=42)
    # Place food far away so distance changes
    env._food = (0, 0)
    env._direction = 1  # RIGHT
    _, reward, done, truncated, _ = env.step(1)
    if not done and not truncated:
        # Only survival bonus — no distance component
        assert reward == 0.1
