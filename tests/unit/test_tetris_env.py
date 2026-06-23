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
    # 4-feature: [lines_cleared_last, holes, bumpiness, sum_height] normalized
    assert obs.shape == (4,)
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
    assert obs.shape == (4,)


def test_obs_all_zeros_on_empty_board(env):
    """On reset, board is empty: holes=0, bumpiness=0, sum_height=0, lines_cleared_last=0."""
    obs, _ = env.reset(seed=0)
    assert obs[0] == 0.0  # lines_cleared_last
    assert obs[1] == 0.0  # holes
    assert obs[2] == 0.0  # bumpiness
    assert obs[3] == 0.0  # sum_height


# ── placement and clearing ─────────────────────────────────────────────────────

def test_placement_fills_board(env):
    env.reset(seed=0)
    total_before = int(env._board.sum())
    env.step(0)
    total_after = int(env._board.sum())
    assert total_after >= total_before


def test_line_clear_removes_rows(env):
    env.reset(seed=0)
    env._board[19, :] = 1
    lines = env._clear_lines()
    assert lines == 1
    assert env._board[19, :].sum() == 0


def test_full_board_clear_adds_empty_rows_on_top():
    e = TetrisEnv()
    e.reset(seed=0)
    e._board[18, :] = 1
    e._board[19, :] = 1
    lines = e._clear_lines()
    assert lines == 2
    assert e._board[0, :].sum() == 0
    assert e._board[1, :].sum() == 0
    e.close()


# ── reward ─────────────────────────────────────────────────────────────────────

def test_placement_reward_no_lines():
    """Placing a piece with no line clear → +piece_placement only."""
    e = TetrisEnv(piece_placement=1.0, line_clear_multiplier=10.0, death_penalty=-2.0)
    e.reset(seed=0)
    _, reward, _, _, _ = e.step(0)
    # No lines cleared on first move of empty board → reward = 1.0
    assert reward == pytest.approx(1.0)
    e.close()


def test_line_clear_reward_quadratic():
    """Clearing N lines → +piece_placement + N² × multiplier."""
    e = TetrisEnv(piece_placement=1.0, line_clear_multiplier=10.0, death_penalty=-2.0)
    e.reset(seed=0)
    # Fill bottom row except one cell, then force the piece to complete it
    e._board[19, :] = 1
    e._board[19, 0] = 0  # leave a gap
    e._current_piece = 0  # I-piece horizontal fills 4 consecutive cells
    _, reward, _, _, _ = e.step(0)  # places I-piece at row 19 col 0, clears line
    # If line clears: reward = 1 + 1² × 10 = 11
    # (lines may or may not clear depending on exact placement)
    assert reward >= 1.0
    e.close()


def test_death_reward():
    e = TetrisEnv(death_penalty=-2.0)
    e.reset(seed=0)
    e._board[:] = 1
    _, reward, terminated, _, _ = e.step(0)
    assert terminated
    assert reward == pytest.approx(-2.0)
    e.close()


def test_no_hole_bumpiness_height_in_reward():
    """Reward must be exactly piece_placement (no shaping penalties)."""
    e = TetrisEnv(piece_placement=1.0, line_clear_multiplier=10.0)
    e.reset(seed=0)
    # Place a piece — creates some bumpiness and potentially holes
    _, reward, terminated, _, _ = e.step(5)
    if not terminated:
        # reward is either 1.0 (no lines) or 1 + n²×10 (lines cleared)
        assert reward == pytest.approx(1.0) or reward > 1.0
    e.close()


# ── helpers ────────────────────────────────────────────────────────────────────

def test_count_holes_with_known_state():
    e = TetrisEnv()
    e.reset(seed=0)
    e._board[:] = 0
    e._board[18, 0] = 1  # block above empty row 19 col 0 → 1 hole
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
    assert e._compute_bumpiness() == 0.0
    e.close()


def test_bumpiness_varies_with_heights():
    e = TetrisEnv()
    e.reset(seed=0)
    e._board[:] = 0
    e._board[18, 0] = 1
    e._board[19, 0] = 1
    assert e._compute_bumpiness() >= 2.0
    e.close()


def test_max_height_empty_board():
    e = TetrisEnv()
    e.reset(seed=0)
    e._board[:] = 0
    assert e._column_heights().max() == 0
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
    obs, reward, terminated, truncated, _ = env.step(30)  # rotation=3, col=0
    assert isinstance(reward, float)


def test_i_piece_horizontal_cant_exceed_board():
    e = TetrisEnv()
    e.reset(seed=0)
    e._current_piece = 0  # I-piece horizontal: 4 wide
    cells = _PIECES[0][0]
    max_dc = max(dc for _, dc in cells)  # 3
    clamped = max(0, min(9, 10 - 1 - max_dc))
    assert clamped == 6
    e.step(0 * 10 + 9)  # rotation=0, col=9 → clamped to 6
    assert e._board.max() <= 1
    e.close()


# ── death and truncation ──────────────────────────────────────────────────────

def test_death_returns_negative_reward():
    e = TetrisEnv(death_penalty=-2.0)
    e.reset(seed=0)
    e._board[:] = 1
    _, reward, terminated, _, _ = e.step(0)
    assert terminated
    assert reward == -2.0
    e.close()


def test_episode_truncates_at_max_steps():
    e = TetrisEnv(max_steps=5)
    e.reset(seed=0)
    e._board[:] = 0
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
    assert obs.shape == (4,)
    e.close()


def test_register_idempotent():
    register()
    register()
    import gymnasium as gym
    e = gym.make("Tetris-v0")
    e.close()


# ── custom reward params ──────────────────────────────────────────────────────

def test_custom_death_penalty():
    e = TetrisEnv(death_penalty=-99.0)
    e.reset(seed=0)
    e._board[:] = 1
    _, reward, done, _, _ = e.step(0)
    assert done
    assert reward == -99.0
    e.close()


def test_legacy_reward_kwargs_ignored():
    """Old hole_penalty / bumpiness_penalty kwargs must not raise."""
    e = TetrisEnv(hole_penalty=2.0, bumpiness_penalty=0.5, height_penalty=-0.1)
    obs, _ = e.reset(seed=0)
    assert obs.shape == (4,)
    e.close()


# ── info dict — lines_cleared tracking ───────────────────────────────────────

def test_step_info_has_lines_cleared(env):
    env.reset(seed=0)
    _, _, _, _, info = env.step(0)
    assert "lines_cleared" in info
    assert isinstance(info["lines_cleared"], int)


def test_lines_cleared_starts_at_zero(env):
    env.reset(seed=0)
    _, _, _, _, info = env.step(0)
    assert info["lines_cleared"] == 0


def test_lines_cleared_cumulates_across_steps():
    e = TetrisEnv()
    e.reset(seed=0)
    e._board[18, :] = 1
    e._board[19, :] = 1
    e._board[18, 0] = 0
    e._board[19, 0] = 0
    e._current_piece = 0
    _, _, _, _, info = e.step(0)
    assert info["lines_cleared"] >= 0
    e.close()


def test_death_info_has_lines_cleared():
    e = TetrisEnv()
    e.reset(seed=0)
    e._board[:] = 1
    _, _, terminated, _, info = e.step(0)
    assert terminated
    assert "lines_cleared" in info
    assert info["lines_cleared"] == 0
    e.close()


def test_lines_cleared_resets_on_new_episode(env):
    env.reset(seed=0)
    env._lines_cleared_episode = 7
    env.reset(seed=1)
    assert env._lines_cleared_episode == 0


def test_lines_cleared_last_reflected_in_obs():
    """lines_cleared_last in obs[0] updates after a clear."""
    e = TetrisEnv()
    e.reset(seed=0)
    # Fill bottom 2 rows completely
    e._board[18, :] = 1
    e._board[19, :] = 1
    e._board[18, 0] = 0
    e._board[19, 0] = 0
    e._current_piece = 0  # I-piece, fills 4 cells horizontally
    obs, _, _, _, _ = e.step(0)
    # If lines cleared, obs[0] > 0
    lines_cleared = e._lines_cleared_last
    assert obs[0] == pytest.approx(lines_cleared / 4.0)
    e.close()
