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
    assert env._sub_r == 0
    assert env._sub_c == 5
    assert env._facing == 1
    assert env._at_surface is True


def test_seaquest_submarine_movement():
    env = MinAtarSeaquestEnv()
    env.reset(seed=42)
    env._enemies.clear()
    env._divers.clear()

    env.step(2)  # RIGHT
    assert env._sub_c == 6
    assert env._facing == 1

    env.step(1)  # LEFT
    assert env._sub_c == 5
    assert env._facing == -1

    env.step(3)  # UP from surface stays at row 0
    assert env._sub_r == 0

    env.step(4)  # DOWN
    assert env._sub_r == 1


def test_seaquest_max_depth_is_row_8():
    env = MinAtarSeaquestEnv()
    env.reset(seed=42)
    env._enemies.clear()
    env._divers.clear()
    env._sub_r = 8
    env._at_surface = False
    env.step(4)
    assert env._sub_r == 8


def test_seaquest_torpedo_kill():
    env = MinAtarSeaquestEnv()
    env.reset(seed=42)
    env._enemies.clear()
    env._divers.clear()

    env._sub_r = 5
    env._sub_c = 2
    env._at_surface = False
    env._facing = 1
    env._enemies = [{
        "r": 5,
        "c": 5.0,
        "dir": -1,
        "type": "sub",
        "timer": 0,
        "freq": 999,
    }]

    env.step(5)
    assert len(env._player_torpedoes) == 1

    hit = False
    for _ in range(5):
        obs, reward, terminated, truncated, info = env.step(0)
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
    env._oxygen = 50
    env._at_surface = False

    env._sub_r = 5
    env._sub_c = 2
    env._divers = [{
        "r": 5,
        "c": 3.0,
        "dir": 1,
        "timer": 0,
        "freq": 999,
    }]

    obs, reward, terminated, truncated, info = env.step(2)
    assert env._divers_held == 1
    assert reward >= 0.5

    for _ in range(5):
        env.step(3)
    assert env._sub_r == 0

    assert env._divers_saved == 1
    assert env._divers_held == 0
    assert env._oxygen == env.oxygen_max
    assert env._at_surface is True
    assert env._score >= 2.5  # pickup (0.5) + rescue (2.0)


def test_seaquest_surface_deposits_one_diver_unless_full():
    env = MinAtarSeaquestEnv()
    env.reset(seed=42)
    env._enemies.clear()
    env._divers.clear()
    env._sub_r = 1
    env._at_surface = False
    env._divers_held = 3
    env.step(3)
    assert env._sub_r == 0
    assert env._divers_held == 2
    assert env._divers_saved == 1
    assert env._oxygen == env.oxygen_max


def test_seaquest_oxygen_does_not_drain_at_surface():
    env = MinAtarSeaquestEnv()
    env.reset(seed=42)
    env._enemies.clear()
    env._divers.clear()
    env._oxygen = 40
    for _ in range(10):
        obs, reward, terminated, truncated, info = env.step(0)
        assert not terminated
        assert info["oxygen"] == 40
    assert env._sub_r == 0


def test_seaquest_empty_surface_death():
    env = MinAtarSeaquestEnv(death_penalty=-4.0)
    env.reset(seed=42)
    env._enemies.clear()
    env._divers.clear()
    env._divers_held = 0
    env.step(4)  # dive
    assert env._sub_r == 1
    assert env._at_surface is False
    obs, reward, terminated, truncated, info = env.step(3)  # resurface empty
    assert terminated
    assert reward <= -4.0
    assert env._sub_r == 0


def test_seaquest_oxygen_depletion_death():
    env = MinAtarSeaquestEnv(death_penalty=-5.0)
    env.reset(seed=42)
    env._enemies.clear()
    env._divers.clear()
    env._sub_r = 3
    env._at_surface = False
    env._oxygen = 1

    obs, reward, terminated, truncated, info = env.step(0)
    assert terminated
    assert reward <= -5.0
    assert info["oxygen"] == 0


def test_seaquest_enemy_collision_death():
    env = MinAtarSeaquestEnv(death_penalty=-3.0)
    env.reset(seed=42)
    env._enemies = [{
        "r": env._sub_r,
        "c": float(env._sub_c),
        "dir": 1,
        "type": "shark",
        "timer": 0,
        "freq": 999,
    }]

    obs, reward, terminated, truncated, info = env.step(0)
    assert terminated
    assert reward <= -3.0


def test_seaquest_viewer_grid():
    env = MinAtarSeaquestEnv()
    env.reset(seed=42)
    grid = env.get_viewer_grid()
    assert len(grid) == 100
    assert grid[0 * 10 + 5] == 1
    assert grid[0 * 10 + 0] == 6


def test_seaquest_gym_make():
    env = gym.make("MinAtar-Seaquest-v0")
    obs, info = env.reset()
    assert obs.shape == (400,)
    next_obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
    assert next_obs.shape == (400,)
    env.close()
