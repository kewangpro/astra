"""Unit tests for MinAtar Asteroids environment (envs/minatar_asteroids_env.py)."""
import numpy as np
import gymnasium as gym
import pytest

from envs.minatar_asteroids_env import MinAtarAsteroidsEnv, register


@pytest.fixture(autouse=True)
def ensure_registered():
    register()


def test_asteroids_spaces():
    env = MinAtarAsteroidsEnv()
    assert env.action_space.n == 5  # NOOP, TURN_L, TURN_R, THRUST, FIRE
    assert env.observation_space.shape == (400,)
    assert env.observation_space.dtype == np.float32


def test_asteroids_reset():
    env = MinAtarAsteroidsEnv()
    obs, info = env.reset(seed=42)
    assert obs.shape == (400,)
    assert info["score"] == 0.0
    assert info["asteroids_hit"] == 0
    assert env._ship_r == 5
    assert env._ship_c == 5
    assert len(env._asteroids) == 4


def test_asteroids_turn_and_thrust():
    env = MinAtarAsteroidsEnv()
    env.reset(seed=42)
    # Heading starts at 0 (UP)
    assert env._ship_dir == 0

    # THRUST UP: r decreases from 5 to 4
    env.step(3)
    assert env._ship_r == 4
    assert env._ship_c == 5

    # TURN_RIGHT: heading becomes 1 (RIGHT)
    env.step(2)
    assert env._ship_dir == 1

    # THRUST RIGHT: c increases from 5 to 6
    env.step(3)
    assert env._ship_r == 4
    assert env._ship_c == 6


def test_asteroids_toroidal_wrap():
    env = MinAtarAsteroidsEnv()
    env.reset(seed=42)
    env._ship_r = 0
    env._ship_dir = 0  # UP

    # Thrust UP past row 0 wraps to row 9
    env.step(3)
    assert env._ship_r == 9


def test_asteroids_shooting_and_split():
    env = MinAtarAsteroidsEnv()
    env.reset(seed=42)
    # Place a large asteroid directly above ship at row 4, col 5
    env._asteroids = [{"r": 4, "c": 5, "dr": 0, "dc": 0, "size": 2}]
    env._ship_r = 5
    env._ship_c = 5
    env._ship_dir = 0  # UP

    # Fire bullet
    env.step(4)
    # Bullet hits asteroid at row 4, col 5
    # Should destroy large asteroid and spawn 2 small ones
    assert env._asteroids_hit == 1
    assert env._score >= 2.0  # size 2 reward
    assert len(env._asteroids) == 2
    for ast in env._asteroids:
        assert ast["size"] == 1


def test_asteroids_wave_clear():
    env = MinAtarAsteroidsEnv()
    env.reset(seed=42)
    # Only 1 small asteroid left
    env._asteroids = [{"r": 4, "c": 5, "dr": 0, "dc": 0, "size": 1}]
    env._ship_r = 5
    env._ship_c = 5
    env._ship_dir = 0  # UP

    obs, reward, terminated, truncated, info = env.step(4)  # FIRE
    assert reward >= 6.0  # 1 hit + 5 wave clear bonus
    # New wave spawned
    assert len(env._asteroids) == 4


def test_asteroids_collision_death():
    env = MinAtarAsteroidsEnv()
    env.reset(seed=42)
    # Put asteroid directly on ship
    env._asteroids = [{"r": 5, "c": 5, "dr": 0, "dc": 0, "size": 1}]
    env._ship_r = 5
    env._ship_c = 5

    obs, reward, terminated, truncated, info = env.step(0)  # NOOP
    assert terminated
    assert reward <= -1.0


def test_asteroids_viewer_grid():
    env = MinAtarAsteroidsEnv()
    env.reset(seed=42)
    grid = env.get_viewer_grid()
    assert len(grid) == 100
    # Ship is at (5, 5) -> index 55 has value 1
    assert grid[5 * 10 + 5] == 1
    # Check asteroids value 2 present
    assert 2 in grid


def test_asteroids_gym_make():
    env = gym.make("MinAtar-Asteroids-v0")
    obs, info = env.reset()
    assert obs.shape == (400,)
    next_obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
    assert next_obs.shape == (400,)
    env.close()
