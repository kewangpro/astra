"""
Tetris-v0: A gymnasium-compatible Tetris environment for ASTRA training missions.

Observation: flat float32 array of length 224
  - Board: 200 floats (20×10, 0=empty, 1=filled)
  - Current piece: 7 floats (one-hot over pieces I, O, T, S, Z, J, L)
  - Next piece: 7 floats (one-hot)
  - Column heights: 10 floats (height of each column, normalized to [0, 1])

Action: Discrete(40) — rotation(0–3) × 10 + column(0–9)
  Invalid rotations are clamped to the piece's maximum rotation count.
  Invalid columns (piece extends past board edge) are clamped to the nearest valid column.
  Every action maps to a valid placement attempt; -1 row from _drop means game over.

Reward (all configurable via constructor kwargs):
  +piece_placement                    for every successfully placed piece
  +line_clear_multiplier × lines²     quadratic — clearing 4 >> clearing 1×4
  −hole_penalty × holes               holes below the topmost filled cell per column
  −bumpiness_penalty × bumpiness      sum of |height differences| between adjacent columns
  +height_penalty × max_height        penalty (height_penalty should be ≤ 0) for tall stacks
  +death_penalty                      on game over (death_penalty should be < 0)

Episode terminates when no valid placement exists for the current piece.
Episode truncates after max_steps placements.
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import gymnasium as gym
from gymnasium import spaces


# ── Tetromino definitions ─────────────────────────────────────────────────────
# Each entry: list of rotations. Each rotation: list of (row, col) offsets
# from the top-left anchor; row increases downward.

_PIECES: List[List[List[Tuple[int, int]]]] = [
    # 0 — I
    [
        [(0, 0), (0, 1), (0, 2), (0, 3)],  # ████
        [(0, 0), (1, 0), (2, 0), (3, 0)],  # vertical
    ],
    # 1 — O  (single rotation)
    [
        [(0, 0), (0, 1), (1, 0), (1, 1)],
    ],
    # 2 — T
    [
        [(0, 0), (0, 1), (0, 2), (1, 1)],  # ███ / _█_
        [(0, 0), (1, 0), (1, 1), (2, 0)],  # █_ / ██ / █_
        [(0, 1), (1, 0), (1, 1), (1, 2)],  # _█_ / ███
        [(0, 1), (1, 0), (1, 1), (2, 1)],  # _█ / ██ / _█
    ],
    # 3 — S
    [
        [(0, 1), (0, 2), (1, 0), (1, 1)],  # _██ / ██_
        [(0, 0), (1, 0), (1, 1), (2, 1)],  # █_ / ██ / _█
    ],
    # 4 — Z
    [
        [(0, 0), (0, 1), (1, 1), (1, 2)],  # ██_ / _██
        [(0, 1), (1, 0), (1, 1), (2, 0)],  # _█ / ██ / █_
    ],
    # 5 — J
    [
        [(0, 0), (1, 0), (1, 1), (1, 2)],  # █__ / ███
        [(0, 0), (0, 1), (1, 0), (2, 0)],  # ██ / █_ / █_
        [(0, 0), (0, 1), (0, 2), (1, 2)],  # ███ / __█
        [(0, 1), (1, 1), (2, 0), (2, 1)],  # _█ / _█ / ██
    ],
    # 6 — L
    [
        [(0, 2), (1, 0), (1, 1), (1, 2)],  # __█ / ███
        [(0, 0), (1, 0), (2, 0), (2, 1)],  # █_ / █_ / ██
        [(0, 0), (0, 1), (0, 2), (1, 0)],  # ███ / █__
        [(0, 0), (0, 1), (1, 1), (2, 1)],  # ██ / _█ / _█
    ],
]


class TetrisEnv(gym.Env):
    metadata = {"render_modes": []}

    ROWS = 20
    COLS = 10
    N_PIECES = 7
    N_ROTATIONS = 4  # max rotations; actual per piece may be less

    def __init__(
        self,
        max_steps: int = 500,
        render_mode: Optional[str] = None,
        line_clear_multiplier: float = 10.0,
        hole_penalty: float = 2.0,
        bumpiness_penalty: float = 0.5,
        piece_placement: float = 1.0,
        height_penalty: float = -0.1,
        death_penalty: float = -10.0,
    ):
        super().__init__()
        self.max_steps = max_steps
        self.render_mode = render_mode
        self.line_clear_multiplier = line_clear_multiplier
        self.hole_penalty = hole_penalty
        self.bumpiness_penalty = bumpiness_penalty
        self.piece_placement = piece_placement
        self.height_penalty = height_penalty
        self.death_penalty = death_penalty

        self.action_space = spaces.Discrete(self.N_ROTATIONS * self.COLS)
        obs_len = self.ROWS * self.COLS + self.N_PIECES + self.N_PIECES + self.COLS
        self.observation_space = spaces.Box(
            low=0.0, high=1.0,
            shape=(obs_len,),
            dtype=np.float32,
        )

        self._board: np.ndarray = np.zeros((self.ROWS, self.COLS), dtype=np.int8)
        self._current_piece: int = 0
        self._next_piece: int = 0
        self._steps: int = 0
        self._lines_cleared_episode: int = 0

    def reset(self, *, seed: Optional[int] = None, options=None):
        super().reset(seed=seed)
        self._board = np.zeros((self.ROWS, self.COLS), dtype=np.int8)
        self._current_piece = int(self.np_random.integers(self.N_PIECES))
        self._next_piece = int(self.np_random.integers(self.N_PIECES))
        self._steps = 0
        self._lines_cleared_episode = 0
        return self._obs(), {}

    def step(self, action: int):
        rotation = int(action) // self.COLS
        col = int(action) % self.COLS

        # Clamp rotation to valid range for this piece type
        piece_rotations = _PIECES[self._current_piece]
        rotation = min(rotation, len(piece_rotations) - 1)
        cells = piece_rotations[rotation]

        # Clamp column so every cell stays within horizontal bounds
        max_dc = max(dc for _, dc in cells)
        col = max(0, min(col, self.COLS - 1 - max_dc))

        # Drop: find the lowest valid anchor row
        placement_row = self._drop(cells, col)

        if placement_row < 0:
            info = {"lines_cleared": self._lines_cleared_episode}
            return self._obs(), self.death_penalty, True, False, info

        # Place cells on board
        for dr, dc in cells:
            self._board[placement_row + dr, col + dc] = 1

        lines_cleared = self._clear_lines()
        self._lines_cleared_episode += lines_cleared

        reward = float(self.piece_placement)
        reward += self.line_clear_multiplier * (lines_cleared ** 2)
        reward -= self.hole_penalty * self._count_holes()
        reward -= self.bumpiness_penalty * self._compute_bumpiness()
        reward += self.height_penalty * self._max_height()

        self._current_piece = self._next_piece
        self._next_piece = int(self.np_random.integers(self.N_PIECES))
        self._steps += 1

        truncated = self._steps >= self.max_steps
        info = {"lines_cleared": self._lines_cleared_episode}
        return self._obs(), reward, False, truncated, info

    # ── helpers ───────────────────────────────────────────────────────────────

    def _drop(self, cells: List[Tuple[int, int]], col: int) -> int:
        """Return the anchor row where the piece lands under gravity, or -1."""
        placement = -1
        for r in range(self.ROWS):
            abs_cells = [(dr + r, dc + col) for dr, dc in cells]
            if all(
                0 <= ar < self.ROWS and 0 <= ac < self.COLS and not self._board[ar, ac]
                for ar, ac in abs_cells
            ):
                placement = r
            else:
                break
        return placement

    def _clear_lines(self) -> int:
        full = np.all(self._board, axis=1)
        n = int(full.sum())
        if n:
            remaining = self._board[~full]
            self._board = np.vstack([
                np.zeros((n, self.COLS), dtype=np.int8),
                remaining,
            ])
        return n

    def _column_heights(self) -> np.ndarray:
        heights = np.zeros(self.COLS, dtype=np.int32)
        for c in range(self.COLS):
            filled = np.where(self._board[:, c])[0]
            if filled.size:
                heights[c] = self.ROWS - filled[0]
        return heights

    def _count_holes(self) -> int:
        holes = 0
        for c in range(self.COLS):
            col = self._board[:, c]
            filled = np.where(col)[0]
            if filled.size:
                holes += int(np.sum(col[filled[0]:] == 0))
        return holes

    def _compute_bumpiness(self) -> float:
        h = self._column_heights()
        return float(np.sum(np.abs(np.diff(h))))

    def _max_height(self) -> int:
        return int(self._column_heights().max())

    def _obs(self) -> np.ndarray:
        board_flat = self._board.flatten().astype(np.float32)
        piece_oh = np.zeros(self.N_PIECES, dtype=np.float32)
        piece_oh[self._current_piece] = 1.0
        next_oh = np.zeros(self.N_PIECES, dtype=np.float32)
        next_oh[self._next_piece] = 1.0
        heights = self._column_heights().astype(np.float32) / self.ROWS
        return np.concatenate([board_flat, piece_oh, next_oh, heights])


# ── registration ──────────────────────────────────────────────────────────────

def register():
    """Call this once before gym.make('Tetris-v0')."""
    if "Tetris-v0" not in gym.envs.registry:
        gym.register(
            id="Tetris-v0",
            entry_point="envs.tetris_env:TetrisEnv",
            kwargs={"max_steps": 500},
        )
