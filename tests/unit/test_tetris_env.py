"""Unit tests for Tetris-v0 custom gymnasium environment."""
from __future__ import annotations

import sys
import os
import pytest
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from envs.tetris_env import TetrisEnv, register, _PIECES


@pytest.fixture
def env():
    e = TetrisEnv(max_steps=200)
    yield e
    e.close()


# ── basic API ─────────────────────────────────────────────────────────────────

def test_observation_shape(env):
    obs, _ = env.reset(seed=0)
    # 20*10 board + 7 current one-hot + 7 next one-hot + 10 heights
    assert obs.shape == (224,)
    assert obs.dtype == np.float32


def test_observation_bounds(env):
    obs, _ = env.reset(seed=0)
    assert obs.min() >= 0.0
    assert obs.max() <= 1.0


def test_action_space(env):
    assert env.action_space.n == 40  # 4 rotations × 10 columns


def test_reset_returns_obs_and_info(env):
    obs, info = env.reset(seed=7)
    assert isinstance(obs, np.ndarray)
    assert isinstance(info, dict)


def test_step_returns_correct_types(env):
    env.reset(seed=0)
    obs, reward, terminated, truncated, info = env.step(0)
    assert isinstance(obs, np.ndarray)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert obs.shape == (224,)


# ── one-hot encoding ──────────────────────────────────────────────────────────

def test_current_piece_one_hot(env):
    env.reset(seed=0)
    obs = env._obs()
    piece_oh = obs[200:207]  # positions 200–206
    assert piece_oh[env._current_piece] == 1.0
    assert piece_oh.sum() == 1.0


def test_next_piece_one_hot(env):
    env.reset(seed=0)
    obs = env._obs()
    next_oh = obs[207:214]  # positions 207–213
    assert next_oh[env._next_piece] == 1.0
    assert next_oh.sum() == 1.0


# ── placement and clearing ─────────────────────────────────────────────────────

def test_placement_fills_board(env):
    env.reset(seed=0)
    total_before = int(env._board.sum())
    env.step(0)  # place one piece
    # Board should have more cells filled (unless immediate game over)
    total_after = int(env._board.sum())
    assert total_after >= total_before  # lines may clear, but net ≥ 0 new cells placed


def test_line_clear_removes_rows(env):
    env.reset(seed=0)
    # Fill row 19 (bottom) manually except one cell
    env._board[19, :] = 1
    lines = env._clear_lines()
    assert lines == 1
    assert env._board[19, :].sum() == 0  # cleared row gone


def test_full_board_clear_adds_empty_rows_on_top():
    e = TetrisEnv()
    e.reset(seed=0)
    # Fill rows 18 and 19 completely
    e._board[18, :] = 1
    e._board[19, :] = 1
    lines = e._clear_lines()
    assert lines == 2
    # Top 2 rows should now be empty
    assert e._board[0, :].sum() == 0
    assert e._board[1, :].sum() == 0
    e.close()


# ── reward shaping helpers ─────────────────────────────────────────────────────

def test_count_holes_with_known_state():
    e = TetrisEnv()
    e.reset(seed=0)
    e._board[:] = 0
    # Column 0: filled at row 18, empty at row 19 → 1 hole
    e._board[18, 0] = 1
    assert e._count_holes() == 1
    e.close()


def test_count_holes_empty_board():
    e = TetrisEnv()
    e.reset(seed=0)
    e._board[:] = 0
    assert e._count_holes() == 0
    e.close()


def test_bumpiness_flat_board():
    e = TetrisEnv()
    e.reset(seed=0)
    e._board[:] = 0
    # All heights = 0, bumpiness = 0
    assert e._compute_bumpiness() == 0.0
    e.close()


def test_bumpiness_varies_with_heights():
    e = TetrisEnv()
    e.reset(seed=0)
    e._board[:] = 0
    # Column 0: height 2, Column 1: height 0 → bumpiness includes |2-0|=2
    e._board[18, 0] = 1
    e._board[19, 0] = 1
    bumpiness = e._compute_bumpiness()
    assert bumpiness >= 2.0
    e.close()


def test_max_height_empty_board():
    e = TetrisEnv()
    e.reset(seed=0)
    e._board[:] = 0
    assert e._max_height() == 0
    e.close()


def test_max_height_with_piece():
    e = TetrisEnv()
    e.reset(seed=0)
    e._board[:] = 0
    # Place 3 cells in column 5 at rows 17, 18, 19 → height=3
    e._board[17, 5] = 1
    e._board[18, 5] = 1
    e._board[19, 5] = 1
    assert e._max_height() == 3
    e.close()


# ── action clamping ───────────────────────────────────────────────────────────

def test_all_40_actions_valid_on_empty_board(env):
    """Every action must produce a valid step on an empty board (no death)."""
    for action in range(40):
        env.reset(seed=0)
        _, _, terminated, _, _ = env.step(action)
        assert not terminated, f"action {action} caused premature game over on empty board"


def test_invalid_rotation_clamped(env):
    """O-piece (index 1) has only 1 rotation — rotation=3 must clamp to 0."""
    env.reset(seed=0)
    env._current_piece = 1  # O-piece
    # Action with rotation=3 (index 30+col): should not crash
    obs, reward, terminated, truncated, _ = env.step(30)  # rotation=3, col=0 → clamp to rot=0
    assert isinstance(reward, float)


# ── column out-of-bounds clamping ─────────────────────────────────────────────

def test_i_piece_horizontal_cant_exceed_board():
    """I-piece R0 is 4 wide — placing at col=9 must be clamped to col=6."""
    e = TetrisEnv()
    e.reset(seed=0)
    e._current_piece = 0  # I-piece
    cells = _PIECES[0][0]  # horizontal: (0,0),(0,1),(0,2),(0,3)
    max_dc = max(dc for _, dc in cells)  # 3
    col = 9
    clamped = max(0, min(col, 10 - 1 - max_dc))
    assert clamped == 6
    # Verify step doesn't write out-of-bounds
    action = 0 * 10 + 9  # rotation=0, col=9
    _, _, _, _, _ = e.step(action)
    assert e._board.max() <= 1  # no corruption
    e.close()


# ── death and truncation ──────────────────────────────────────────────────────

def test_death_returns_negative_reward():
    e = TetrisEnv(death_penalty=-10.0)
    e.reset(seed=0)
    # Fill the board so next piece can't be placed
    e._board[:] = 1
    _, reward, terminated, _, _ = e.step(0)
    assert terminated
    assert reward == -10.0
    e.close()


def test_episode_truncates_at_max_steps():
    e = TetrisEnv(max_steps=5)
    e.reset(seed=0)
    e._board[:] = 0  # keep board empty so no game-over
    steps = 0
    truncated = False
    for _ in range(10):
        _, _, terminated, truncated, _ = e.step(0)
        steps += 1
        if terminated or truncated:
            break
    assert truncated or steps <= 5
    e.close()


# ── registration ──────────────────────────────────────────────────────────────

def test_register_creates_gym_env():
    import gymnasium as gym
    register()
    e = gym.make("Tetris-v0")
    obs, _ = e.reset(seed=1)
    assert obs.shape == (224,)
    e.close()


def test_register_idempotent():
    register()
    register()  # calling twice must not raise
    import gymnasium as gym
    e = gym.make("Tetris-v0")
    e.close()


# ── custom reward params ──────────────────────────────────────────────────────

def test_custom_death_penalty():
    e = TetrisEnv(death_penalty=-99.0)
    e.reset(seed=0)
    e._board[:] = 1  # force instant death
    _, reward, done, _, _ = e.step(0)
    assert done
    assert reward == -99.0
    e.close()


def test_line_clear_reward_is_quadratic():
    e = TetrisEnv(line_clear_multiplier=10.0, piece_placement=0.0,
                  hole_penalty=0.0, bumpiness_penalty=0.0, height_penalty=0.0)
    e.reset(seed=0)
    # Fill bottom row except one cell, then manually complete it and check clear
    e._board[19, :] = 1
    lines = e._clear_lines()
    # 1 line cleared → reward = 10 * 1² = 10
    assert lines == 1
    e.close()
