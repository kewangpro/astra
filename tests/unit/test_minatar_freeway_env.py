"""Unit tests for MinAtar Freeway environment (envs/minatar_freeway_env.py)."""
import numpy as np
import gymnasium as gym
import pytest

from envs.minatar_freeway_env import MinAtarFreewayEnv, register


@pytest.fixture(autouse=True)
def ensure_registered():
    register()


def test_freeway_spaces():
    env = MinAtarFreewayEnv()
    assert env.action_space.n == 3  # NOOP, UP, DOWN
    assert env.observation_space.shape == (400,)
    assert env.observation_space.dtype == np.float32


def test_freeway_reset():
    env = MinAtarFreewayEnv()
    obs, info = env.reset(seed=42)
    assert obs.shape == (400,)
    assert info["score"] == 0.0
    assert info["crossings"] == 0
    assert info["collisions"] == 0
    assert env._player_r == 9
    assert env._player_c == 4


def test_freeway_vertical_movement():
    env = MinAtarFreewayEnv()
    env.reset(seed=42)
    # Clear cars to test pure movement
    for r in range(10):
        env._cars[r].clear()

    # Move UP: row 9 -> row 8
    env.step(1)
    assert env._player_r == 8
    assert env._player_c == 4

    # Move UP again: row 8 -> row 7
    env.step(1)
    assert env._player_r == 7

    # Move DOWN: row 7 -> row 8
    env.step(2)
    assert env._player_r == 8

    # Move DOWN to boundary: row 9
    env.step(2)
    assert env._player_r == 9
    # Can't move below row 9
    env.step(2)
    assert env._player_r == 9


def test_freeway_crossing_goal():
    env = MinAtarFreewayEnv()
    env.reset(seed=42)
    # Clear cars to test goal
    for r in range(10):
        env._cars[r].clear()

    # Move all the way UP to goal (row 0)
    for _ in range(9):
        env.step(1)

    assert env._crossings == 1
    assert env._score == 1.0
    # Resets to start sidewalk (row 9)
    assert env._player_r == 9


def test_freeway_car_collision():
    env = MinAtarFreewayEnv(death_penalty=-2.0)
    env.reset(seed=42)
    # Place a car right above player at (8, 4)
    env._cars[8] = [4.0]
    env._lane_freqs[8] = 999  # keep car stationary

    obs, reward, terminated, truncated, info = env.step(1)  # UP into car
    assert info["collisions"] == 1
    assert reward <= -2.0
    # Knocked back to row 9
    assert env._player_r == 9
    assert not terminated  # standard Freeway continues


def test_freeway_viewer_grid():
    env = MinAtarFreewayEnv()
    env.reset(seed=42)
    grid = env.get_viewer_grid()
    assert len(grid) == 100
    # Chicken at row 9, col 4 -> 94 has value 1
    assert grid[9 * 10 + 4] == 1
    # Sidewalks have value 4
    assert grid[0 * 10 + 0] == 4
    assert grid[9 * 10 + 0] == 4
    # Cars present in traffic lanes
    assert (2 in grid) or (3 in grid)


def test_freeway_gym_make():
    env = gym.make("MinAtar-Freeway-v0")
    obs, info = env.reset()
    assert obs.shape == (400,)
    next_obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
    assert next_obs.shape == (400,)
    env.close()
