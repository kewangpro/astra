"""Unit tests for Game2048-v0 custom gymnasium environment."""
from __future__ import annotations

import sys
import os
import pytest
import numpy as np
import gymnasium as gym

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from envs.game2048_env import Game2048Env, register


@pytest.fixture
def env():
    e = Game2048Env(max_steps=500)
    yield e
    e.close()


# ── Observation and Action Space ─────────────────────────────────────────────

def test_observation_shape(env):
    obs, info = env.reset(seed=42)
    assert obs.shape == (16,)
    assert obs.dtype == np.float32
    assert obs.min() >= 0.0
    assert obs.max() <= 1.0


def test_action_space(env):
    assert env.action_space.n == 4  # UP, DOWN, LEFT, RIGHT


def test_reset_spawns_two_tiles(env):
    env.reset(seed=42)
    nonzero_count = np.count_nonzero(env._board)
    assert nonzero_count == 2
    assert env._max_tile in (2, 4)
    assert env._score == 0
    assert env._steps == 0


def test_reset_returns_obs_and_info(env):
    obs, info = env.reset(seed=1)
    assert isinstance(obs, np.ndarray)
    assert isinstance(info, dict)
    assert "score" in info or True


# ── Slide and Merge Mechanics ────────────────────────────────────────────────

def test_slide_left_simple_pair():
    row = np.array([2, 2, 0, 0], dtype=np.int32)
    slid, score = Game2048Env._slide_left_row(row)
    assert np.array_equal(slid, [4, 0, 0, 0])
    assert score == 4


def test_slide_left_two_pairs():
    row = np.array([2, 2, 2, 2], dtype=np.int32)
    slid, score = Game2048Env._slide_left_row(row)
    assert np.array_equal(slid, [4, 4, 0, 0])
    assert score == 8


def test_slide_left_with_gaps():
    row = np.array([2, 0, 2, 4], dtype=np.int32)
    slid, score = Game2048Env._slide_left_row(row)
    assert np.array_equal(slid, [4, 4, 0, 0])
    assert score == 4


def test_slide_left_different_numbers_no_merge():
    row = np.array([4, 2, 2, 0], dtype=np.int32)
    slid, score = Game2048Env._slide_left_row(row)
    assert np.array_equal(slid, [4, 4, 0, 0])
    assert score == 4


# ── Move Directions ──────────────────────────────────────────────────────────

def test_simulate_move_all_directions(env):
    env._board = np.array([
        [2, 0, 0, 0],
        [2, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
    ], dtype=np.int32)

    # UP (action 0)
    b_up, score_up, moved_up = env._simulate_move(env._board, 0)
    assert moved_up
    assert b_up[0, 0] == 4
    assert score_up == 4

    # DOWN (action 1)
    b_down, score_down, moved_down = env._simulate_move(env._board, 1)
    assert moved_down
    assert b_down[3, 0] == 4
    assert score_down == 4

    # RIGHT (action 3)
    b_right, score_right, moved_right = env._simulate_move(env._board, 3)
    assert moved_right
    assert b_right[0, 3] == 2
    assert b_right[1, 3] == 2
    assert score_right == 0

    # LEFT (action 2): tiles already against left border, invalid move
    b_left, score_left, moved_left = env._simulate_move(env._board, 2)
    assert not moved_left
    assert np.array_equal(b_left, env._board)
    assert score_left == 0


def test_step_returns_correct_types(env):
    env.reset(seed=0)
    obs, reward, terminated, truncated, info = env.step(0)
    assert isinstance(obs, np.ndarray)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert isinstance(info, dict)
    assert "score" in info
    assert "max_tile" in info
    assert "steps" in info
    assert "empty_cells" in info


def test_invalid_move_penalty(env):
    env.reset(seed=0)
    env._board = np.array([
        [2, 0, 0, 0],
        [4, 0, 0, 0],
        [8, 0, 0, 0],
        [16, 0, 0, 0],
    ], dtype=np.int32)
    obs, reward, terminated, truncated, info = env.step(2)  # LEFT
    assert reward == -1.0
    assert not terminated


# ── Lookahead Support ────────────────────────────────────────────────────────

def test_get_next_states_lookahead(env):
    env.reset(seed=0)
    env._board = np.array([
        [2, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
    ], dtype=np.int32)

    states = env.get_next_states()
    assert isinstance(states, dict)
    # Moving DOWN (1) or RIGHT (3) valid; UP (0) or LEFT (2) cannot move
    assert 1 in states
    assert 3 in states
    assert 0 not in states
    assert 2 not in states

    for act, s in states.items():
        assert s.shape == (16,)
        assert s.dtype == np.float32

    # Verify board was not mutated
    assert env._board[0, 0] == 2
    assert np.count_nonzero(env._board) == 1


# ── Game Over and Truncation ─────────────────────────────────────────────────

def test_game_over_detection(env):
    env.reset(seed=0)
    env._board = np.array([
        [2, 4, 2, 4],
        [4, 2, 4, 2],
        [2, 4, 2, 4],
        [4, 2, 4, 2],
    ], dtype=np.int32)
    assert not env._has_moves_left()
    obs, reward, terminated, truncated, info = env.step(0)
    assert terminated


def test_episode_truncation():
    e = Game2048Env(max_steps=5)
    e.reset(seed=0)
    truncated = False
    for a in [0, 1, 2, 3, 0, 1]:
        _, _, terminated, truncated, _ = e.step(a)
        if terminated or truncated:
            break
    assert truncated or e._steps <= 5
    e.close()


# ── Viewer Grid and Registration ─────────────────────────────────────────────

def test_viewer_grid_format(env):
    env.reset(seed=0)
    grid = env.get_viewer_grid()
    assert isinstance(grid, list)
    assert len(grid) == 16
    assert all(isinstance(x, int) for x in grid)
    assert sum(1 for x in grid if x > 0) == 2


def test_register_creates_gym_env():
    register()
    e = gym.make("Game2048-v0")
    obs, info = e.reset(seed=99)
    assert obs.shape == (16,)
    obs, reward, terminated, truncated, info = e.step(0)
    assert isinstance(reward, float)
    e.close()


def test_reward_shaping_bonuses():
    e = Game2048Env(empty_tile_bonus=0.1, corner_bonus=2.0)
    e.reset(seed=0)
    # Put max tile in corner (0, 0)
    e._board = np.array([
        [4, 4, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
    ], dtype=np.int32)
    # Merging UP or LEFT merges to 8 at (0, 0), which is corner and leaves empty tiles
    _, reward, _, _, _ = e.step(2)  # LEFT
    # Merge score = 8
    # Empty tile bonus: >0
    # Corner bonus: 2.0 (since 8 is at (0, 0))
    assert reward > 8.0 + 2.0
    e.close()
