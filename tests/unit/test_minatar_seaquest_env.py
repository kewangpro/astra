"""Unit tests for MinAtar Seaquest environment (envs/minatar_seaquest_env.py)."""
import numpy as np
import gymnasium as gym
import pytest

from envs.minatar_seaquest_env import MinAtarSeaquestEnv, register


@pytest.fixture(autouse=True)
def ensure_registered():
    register()


def test_seaquest_spaces():
    env = MinAtarSeaquestEnv()
    assert env.action_space.n == 6  # NOOP, LEFT, RIGHT, UP, DOWN, FIRE
    assert env.observation_space.shape == (400,)
    assert env.observation_space.dtype == np.float32


def test_seaquest_reset():
    env = MinAtarSeaquestEnv()
    obs, info = env.reset(seed=42)
    assert obs.shape == (400,)
    assert info["score"] == 0.0
    assert info["enemies_killed"] == 0
    assert info["divers_saved"] == 0
    assert info["oxygen"] == env.oxygen_max
    assert env._sub_r == 5
    assert env._sub_c == 2
    assert env._facing == 1


def test_seaquest_submarine_movement():
    env = MinAtarSeaquestEnv()
    env.reset(seed=42)
    env._enemies.clear()
    env._divers.clear()

    # Move RIGHT (action 2)
    env.step(2)
    assert env._sub_c == 3
    assert env._facing == 1

    # Move LEFT (action 1)
    env.step(1)
    assert env._sub_c == 2
    assert env._facing == -1

    # Move UP (action 3)
    env.step(3)
    assert env._sub_r == 4

    # Move DOWN (action 4)
    env.step(4)
    assert env._sub_r == 5


def test_seaquest_torpedo_kill():
    env = MinAtarSeaquestEnv()
    env.reset(seed=42)
    env._enemies.clear()
    env._divers.clear()

    # Place an enemy in front of the submarine
    env._sub_r = 5
    env._sub_c = 2
    env._facing = 1  # facing right
    env._enemies = [{
        "r": 5,
        "c": 5.0,
        "dir": -1,
        "type": "sub",
        "timer": 0,
        "freq": 999,  # keep stationary
    }]

    # Fire torpedo (action 5)
    env.step(5)
    assert len(env._player_torpedoes) == 1

    # Advance steps until torpedo strikes enemy
    hit = False
    for _ in range(5):
        obs, reward, terminated, truncated, info = env.step(0)  # NOOP
        if info["enemies_killed"] > 0:
            hit = True
            break
    assert hit
    assert env._enemies_killed == 1
    assert env._score >= 1.0


def test_seaquest_diver_pickup_and_surfacing():
    env = MinAtarSeaquestEnv()
    env.reset(seed=42)
    env._enemies.clear()
    env._divers.clear()
    env._oxygen = 50  # depleted oxygen

    # Place diver at (5, 3)
    env._sub_r = 5
    env._sub_c = 2
    env._divers = [{
        "r": 5,
        "c": 3.0,
        "dir": 1,
        "timer": 0,
        "freq": 999,
    }]

    # Step RIGHT onto diver
    obs, reward, terminated, truncated, info = env.step(2)
    assert env._divers_held == 1
    assert reward >= 0.5  # pickup reward

    # Move to surface (row 0)
    for _ in range(5):
        env.step(3)  # UP
    assert env._sub_r == 0

    # Diver deposited, oxygen refilled
    assert env._divers_saved == 1
    assert env._divers_held == 0
    assert env._oxygen == env.oxygen_max
    assert env._score >= 2.5  # pickup (0.5) + rescue (2.0)


def test_seaquest_oxygen_depletion_death():
    env = MinAtarSeaquestEnv(death_penalty=-5.0)
    env.reset(seed=42)
    env._enemies.clear()
    env._divers.clear()
    env._oxygen = 1

    obs, reward, terminated, truncated, info = env.step(0)  # NOOP
    assert terminated
    assert reward <= -5.0
    assert info["oxygen"] == 0


def test_seaquest_enemy_collision_death():
    env = MinAtarSeaquestEnv(death_penalty=-3.0)
    env.reset(seed=42)
    # Place enemy directly on submarine position
    env._enemies = [{
        "r": env._sub_r,
        "c": float(env._sub_c),
        "dir": 1,
        "type": "shark",
        "timer": 0,
        "freq": 999,
    }]

    obs, reward, terminated, truncated, info = env.step(0)  # NOOP
    assert terminated
    assert reward <= -3.0


def test_seaquest_viewer_grid():
    env = MinAtarSeaquestEnv()
    env.reset(seed=42)
    grid = env.get_viewer_grid()
    assert len(grid) == 100
    # Submarine at row 5 col 2 -> 52 has value 1
    assert grid[5 * 10 + 2] == 1
    # Surface water at row 0 has value 6
    assert grid[0 * 10 + 0] == 6


def test_seaquest_gym_make():
    env = gym.make("MinAtar-Seaquest-v0")
    obs, info = env.reset()
    assert obs.shape == (400,)
    next_obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
    assert next_obs.shape == (400,)
    env.close()
