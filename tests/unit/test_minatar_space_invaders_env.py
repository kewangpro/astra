"""Unit tests for MinAtar Space Invaders environment (envs/minatar_space_invaders_env.py)."""
import numpy as np
import gymnasium as gym
import pytest

from envs.minatar_space_invaders_env import MinAtarSpaceInvadersEnv, register


@pytest.fixture(autouse=True)
def ensure_registered():
    register()


def test_space_invaders_spaces():
    env = MinAtarSpaceInvadersEnv()
    assert env.action_space.n == 4  # NOOP, LEFT, RIGHT, FIRE
    assert env.observation_space.shape == (400,)
    assert env.observation_space.dtype == np.float32


def test_space_invaders_reset():
    env = MinAtarSpaceInvadersEnv()
    obs, info = env.reset(seed=42)
    assert obs.shape == (400,)
    assert info["score"] == 0.0
    assert info["aliens_killed"] == 0
    assert env._cannon_x == 4
    assert np.any(env._aliens)


def test_space_invaders_cannon_movement():
    env = MinAtarSpaceInvadersEnv()
    env.reset(seed=42)

    # Move LEFT
    env.step(1)
    assert env._cannon_x == 3
    # Move further LEFT to boundary
    for _ in range(5):
        env.step(1)
    assert env._cannon_x == 0  # clamped at 0

    # Move RIGHT
    env.step(2)
    assert env._cannon_x == 1
    # Move further RIGHT to boundary
    for _ in range(15):
        env.step(2)
    assert env._cannon_x == 9  # clamped at 9


def test_space_invaders_laser_kill():
    env = MinAtarSpaceInvadersEnv()
    env.reset(seed=42)
    # Place an alien directly above cannon at row 2, col 4
    env._aliens.fill(False)
    env._aliens[2, 4] = True
    env._cannon_x = 4
    env._alien_move_freq = 999  # keep alien stationary

    # Fire laser
    env.step(3)
    assert len(env._player_lasers) == 1

    # Advance steps until laser hits alien
    killed = False
    for _ in range(10):
        obs, reward, terminated, truncated, info = env.step(0)  # NOOP
        if info["aliens_killed"] > 0:
            killed = True
            break
    assert killed
    assert env._aliens_killed == 1
    assert env._score >= 1.0


def test_space_invaders_wave_clear_bonus():
    env = MinAtarSpaceInvadersEnv()
    env.reset(seed=42)
    # Wipe aliens to trigger wave clear on next laser kill
    env._aliens.fill(False)
    env._aliens[8, 4] = True
    env._cannon_x = 4
    obs, reward, terminated, truncated, info = env.step(3)  # FIRE immediately hits [8, 4]
    assert reward >= 5.0  # Wave clear bonus
    # New wave spawned
    assert np.any(env._aliens)


def test_space_invaders_viewer_grid():
    env = MinAtarSpaceInvadersEnv()
    env.reset(seed=42)
    grid = env.get_viewer_grid()
    assert len(grid) == 100
    # Cannon is at row 9, col 4 -> index 94 has value 1
    assert grid[9 * 10 + 4] == 1
    # Check aliens have value 2
    assert 2 in grid
    # Check shields have value 5
    assert 5 in grid


def test_space_invaders_gym_make():
    env = gym.make("MinAtar-SpaceInvaders-v0")
    obs, info = env.reset()
    assert obs.shape == (400,)
    next_obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
    assert next_obs.shape == (400,)
    env.close()
