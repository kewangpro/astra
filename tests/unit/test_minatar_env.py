"""Unit tests for MinAtar-Breakout-v0 custom gymnasium environment."""
from __future__ import annotations

import sys
import os
import pytest
import numpy as np
import gymnasium as gym

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from envs.minatar_env import MinAtarBreakoutEnv, register


@pytest.fixture
def env():
    e = MinAtarBreakoutEnv(max_steps=500)
    yield e
    e.close()


# ── Observation and Action Space ─────────────────────────────────────────────

def test_observation_shape(env):
    obs, info = env.reset(seed=42)
    assert obs.shape == (400,)
    assert obs.dtype == np.float32
    assert obs.min() >= 0.0
    assert obs.max() <= 1.0


def test_action_space(env):
    assert env.action_space.n == 3  # NOOP, LEFT, RIGHT


def test_reset_initializes_state(env):
    obs, info = env.reset(seed=42)
    assert env._steps == 0
    assert env._score == 0.0
    assert env._bricks_cleared == 0
    # 3 rows of 10 bricks = 30 bricks
    assert np.sum(env._bricks) == 30
    # Paddle width is 2, col must be between 0 and 8
    assert 0 <= env._paddle_col <= 8
    assert env._ball_r == 6


def test_reset_returns_obs_and_info(env):
    obs, info = env.reset(seed=1)
    assert isinstance(obs, np.ndarray)
    assert isinstance(info, dict)


# ── Paddle Mechanics ─────────────────────────────────────────────────────────

def test_paddle_movement(env):
    env.reset(seed=42)
    env._paddle_col = 4

    # NOOP (action 0)
    env.step(0)
    assert env._paddle_col == 4

    # LEFT (action 1)
    env.step(1)
    assert env._paddle_col == 3

    # Move left multiple times to check boundary clamp
    for _ in range(10):
        env.step(1)
    assert env._paddle_col == 0

    # RIGHT (action 2) multiple times to check right clamp
    for _ in range(15):
        env.step(2)
    assert env._paddle_col == env.COLS - env.PADDLE_WIDTH


# ── Step Types and Dynamics ──────────────────────────────────────────────────

def test_step_returns_correct_types(env):
    env.reset(seed=0)
    obs, reward, terminated, truncated, info = env.step(0)
    assert isinstance(obs, np.ndarray)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert isinstance(info, dict)
    assert "score" in info
    assert "bricks_cleared" in info
    assert "steps" in info


def test_brick_collision(env):
    env.reset(seed=0)
    # Set ball right below a brick heading straight up
    env._ball_r = 4
    env._ball_c = 5
    env._ball_dr = -1
    env._ball_dc = 0
    env._bricks[3, 5] = 1

    obs, reward, terminated, truncated, info = env.step(0)
    # Brick at (3,5) should now be 0
    assert env._bricks[3, 5] == 0
    assert reward >= 1.0
    assert env._bricks_cleared == 1
    # Ball should have bounced downwards
    assert env._ball_dr == 1


def test_paddle_bounce_and_score(env):
    env.reset(seed=0)
    # Set paddle at col 4
    env._paddle_col = 4
    # Set ball directly above left half of paddle falling down
    env._ball_r = 8
    env._ball_c = 4
    env._ball_dr = 1
    env._ball_dc = 0

    obs, reward, terminated, truncated, info = env.step(0)
    assert not terminated
    # Paddle hit reward awarded
    assert reward == pytest.approx(env.paddle_hit_reward)
    # Bounced up and angled left
    assert env._ball_dr == -1
    assert env._ball_dc == -1


def test_death_penalty_on_ball_loss(env):
    env.reset(seed=0)
    # Put paddle far on the right
    env._paddle_col = 8
    # Ball falls on the left
    env._ball_r = 8
    env._ball_c = 0
    env._ball_dr = 1
    env._ball_dc = 0

    obs, reward, terminated, truncated, info = env.step(0)
    assert terminated
    assert reward == pytest.approx(env.death_penalty)


def test_episode_truncation():
    e = MinAtarBreakoutEnv(max_steps=5)
    e.reset(seed=0)
    truncated = False
    for _ in range(10):
        _, _, terminated, truncated, _ = e.step(0)
        if terminated or truncated:
            break
    assert truncated or e._steps <= 5
    e.close()


def test_all_bricks_cleared_bonus(env):
    env.reset(seed=0)
    # Clear all bricks except one
    env._bricks[:] = 0
    env._bricks[3, 5] = 1

    env._ball_r = 4
    env._ball_c = 5
    env._ball_dr = -1
    env._ball_dc = 0

    _, reward, _, _, _ = env.step(0)
    # Should get brick_reward (1.0) + clear bonus (10.0) = 11.0
    assert reward == pytest.approx(11.0)
    # Bricks should be re-initialized
    assert np.sum(env._bricks) == 30


# ── Viewer Grid and Registration ─────────────────────────────────────────────

def test_viewer_grid_format(env):
    env.reset(seed=0)
    grid = env.get_viewer_grid()
    assert isinstance(grid, list)
    assert len(grid) == 100
    # Values can be 0 (empty), 1 (paddle), 2 (ball), 3 (brick)
    assert set(grid).issubset({0, 1, 2, 3})
    assert 1 in grid  # paddle exists
    assert 2 in grid  # ball exists
    assert 3 in grid  # bricks exist


def test_register_creates_gym_env():
    register()
    e = gym.make("MinAtar-Breakout-v0")
    obs, info = e.reset(seed=42)
    assert obs.shape == (400,)
    obs2, reward, terminated, truncated, info = e.step(0)
    assert isinstance(reward, float)
    e.close()


def test_register_minatar_v0_alias():
    register()
    e = gym.make("MinAtar-v0")
    obs, info = e.reset(seed=42)
    assert obs.shape == (400,)
    e.close()
