"""
Tetris-v0: A gymnasium-compatible Tetris environment for ASTRA training missions.

Observation: float32 array of shape (4,)
  [lines_cleared_last, holes, bumpiness, sum_height]  — normalized to [0, 1]
  lines_cleared_last : lines cleared by the previous move   / 4
  holes              : empty cells with a filled cell above  / 200
  bumpiness          : sum of |height diff| between adj cols / 180
  sum_height         : sum of all column heights             / 200

  This compact 4-feature representation matches the approach used by the
  reference project (tetris_ppo_cnn) which achieved 45+ lines with a plain
  PPO MLP — versus only ~10 with a flat 224-element board observation.

Action: Discrete(40) — rotation(0–3) × 10 + column(0–9)
  Invalid rotations are clamped to the piece's maximum rotation count.
  Invalid columns (piece extends past board edge) are clamped to the nearest
  valid column.  Every action maps to a valid placement.

Reward:
  +1                        for every successfully placed piece
  +lines_cleared² × 10     quadratic — clearing 4 lines >> clearing 1×4 times
  −2                        on game over

Episode terminates when no valid placement exists for the current piece.
Episode truncates after max_steps placements (default 1000).
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
        [(0, 0), (0, 1), (0, 2), (1, 1)],
        [(0, 0), (1, 0), (1, 1), (2, 0)],
        [(0, 1), (1, 0), (1, 1), (1, 2)],
        [(0, 1), (1, 0), (1, 1), (2, 1)],
    ],
    # 3 — S
    [
        [(0, 1), (0, 2), (1, 0), (1, 1)],
        [(0, 0), (1, 0), (1, 1), (2, 1)],
    ],
    # 4 — Z
    [
        [(0, 0), (0, 1), (1, 1), (1, 2)],
        [(0, 1), (1, 0), (1, 1), (2, 0)],
    ],
    # 5 — J
    [
        [(0, 0), (1, 0), (1, 1), (1, 2)],
        [(0, 0), (0, 1), (1, 0), (2, 0)],
        [(0, 0), (0, 1), (0, 2), (1, 2)],
        [(0, 1), (1, 1), (2, 0), (2, 1)],
    ],
    # 6 — L
    [
        [(0, 2), (1, 0), (1, 1), (1, 2)],
        [(0, 0), (1, 0), (2, 0), (2, 1)],
        [(0, 0), (0, 1), (0, 2), (1, 0)],
        [(0, 0), (0, 1), (1, 1), (2, 1)],
    ],
]

_MAX_LINES_PER_STEP = 4
_MAX_HOLES = 200        # 10 cols × 20 rows
_MAX_BUMPINESS = 180    # worst case adjacent height diffs
_MAX_SUM_HEIGHT = 200   # all 10 cols at height 20


class TetrisEnv(gym.Env):
    metadata = {"render_modes": []}

    ROWS = 20
    COLS = 10
    N_PIECES = 7
    N_ROTATIONS = 4  # max rotations; actual per piece may be fewer

    def __init__(
        self,
        max_steps: int = 1000,
        render_mode: Optional[str] = None,
        # kept for backwards-compat with recipe kwargs — not used in reward
        line_clear_multiplier: float = 10.0,
        piece_placement: float = 1.0,
        death_penalty: float = -2.0,
        **kwargs,  # absorb legacy reward-shaping kwargs silently
    ):
        super().__init__()
        self.max_steps = max_steps
        self.render_mode = render_mode
        self.line_clear_multiplier = line_clear_multiplier
        self.piece_placement = piece_placement
        self.death_penalty = death_penalty

        self.action_space = spaces.Discrete(self.N_ROTATIONS * self.COLS)
        # 4-feature obs: [lines_cleared_last, holes, bumpiness, sum_height] (normalized)
        self.observation_space = spaces.Box(
            low=0.0, high=1.0,
            shape=(4,),
            dtype=np.float32,
        )

        self._board: np.ndarray = np.zeros((self.ROWS, self.COLS), dtype=np.int8)
        self._current_piece: int = 0
        self._next_piece: int = 0
        self._steps: int = 0
        self._lines_cleared_episode: int = 0
        self._lines_cleared_last: int = 0

    def reset(self, *, seed: Optional[int] = None, options=None):
        super().reset(seed=seed)
        self._board = np.zeros((self.ROWS, self.COLS), dtype=np.int8)
        self._current_piece = int(self.np_random.integers(self.N_PIECES))
        self._next_piece = int(self.np_random.integers(self.N_PIECES))
        self._steps = 0
        self._lines_cleared_episode = 0
        self._lines_cleared_last = 0
        return self._obs(), {}

    def step(self, action: int):
        rotation = int(action) // self.COLS
        col = int(action) % self.COLS

        # Clamp rotation to valid range for this piece
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
            return self._obs(), float(self.death_penalty), True, False, info

        # Place cells on board
        for dr, dc in cells:
            self._board[placement_row + dr, col + dc] = 1

        lines_cleared = self._clear_lines()
        self._lines_cleared_episode += lines_cleared
        self._lines_cleared_last = lines_cleared

        reward = float(self.piece_placement) + self.line_clear_multiplier * (lines_cleared ** 2)

        self._current_piece = self._next_piece
        self._next_piece = int(self.np_random.integers(self.N_PIECES))
        self._steps += 1

        truncated = self._steps >= self.max_steps
        info = {"lines_cleared": self._lines_cleared_episode}
        return self._obs(), reward, False, truncated, info

    # ── helpers ───────────────────────────────────────────────────────────────

    def _drop(self, cells: List[Tuple[int, int]], col: int) -> int:
        """Return the anchor row where the piece lands, or -1 if no valid placement."""
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

    def _obs(self) -> np.ndarray:
        holes = self._count_holes()
        bumpiness = self._compute_bumpiness()
        sum_height = float(self._column_heights().sum())
        return np.array([
            self._lines_cleared_last / _MAX_LINES_PER_STEP,
            holes / _MAX_HOLES,
            bumpiness / _MAX_BUMPINESS,
            sum_height / _MAX_SUM_HEIGHT,
        ], dtype=np.float32)

    def get_board_props(self) -> List[float]:
        """Return raw [holes, bumpiness, sum_height] for inspection/logging."""
        return [
            float(self._count_holes()),
            self._compute_bumpiness(),
            float(self._column_heights().sum()),
        ]

    def get_next_states(self) -> dict:
        """Return {action: 4-feature obs} for every valid placement of the current piece.

        Simulates each of the 40 possible (rotation, col) actions on a board copy
        without modifying live state. Terminal placements (no valid drop) are excluded.
        Used by Actor-Critic agents to evaluate resulting states before acting.
        """
        states = {}
        piece_rotations = _PIECES[self._current_piece]
        for action in range(self.N_ROTATIONS * self.COLS):
            rotation = min(action // self.COLS, len(piece_rotations) - 1)
            col = action % self.COLS
            cells = piece_rotations[rotation]
            max_dc = max(dc for _, dc in cells)
            col = max(0, min(col, self.COLS - 1 - max_dc))

            board = self._board.copy()
            placement_row = self._drop_on(board, cells, col)
            if placement_row < 0:
                continue  # game-over placement — skip

            for dr, dc in cells:
                board[placement_row + dr, col + dc] = 1

            lines = self._clear_lines_on(board)
            states[action] = self._obs_from(board, lines)
        return states

    # ── board-copy helpers for get_next_states ────────────────────────────────

    @staticmethod
    def _drop_on(board: np.ndarray, cells: List[Tuple[int, int]], col: int) -> int:
        rows, cols = board.shape
        placement = -1
        for r in range(rows):
            abs_cells = [(dr + r, dc + col) for dr, dc in cells]
            if all(0 <= ar < rows and 0 <= ac < cols and not board[ar, ac] for ar, ac in abs_cells):
                placement = r
            else:
                break
        return placement

    @staticmethod
    def _clear_lines_on(board: np.ndarray) -> int:
        full = np.all(board, axis=1)
        n = int(full.sum())
        if n:
            remaining = board[~full]
            board[:] = np.vstack([np.zeros((n, board.shape[1]), dtype=np.int8), remaining])
        return n

    @staticmethod
    def _obs_from(board: np.ndarray, lines_cleared_last: int) -> np.ndarray:
        rows, cols = board.shape
        heights = np.zeros(cols, dtype=np.int32)
        for c in range(cols):
            filled = np.where(board[:, c])[0]
            if filled.size:
                heights[c] = rows - filled[0]
        holes = sum(
            int(np.sum(board[np.where(board[:, c])[0][0]:, c] == 0))
            for c in range(cols) if np.any(board[:, c])
        )
        bumpiness = float(np.sum(np.abs(np.diff(heights))))
        sum_height = float(heights.sum())
        return np.array([
            lines_cleared_last / _MAX_LINES_PER_STEP,
            holes / _MAX_HOLES,
            bumpiness / _MAX_BUMPINESS,
            sum_height / _MAX_SUM_HEIGHT,
        ], dtype=np.float32)


# ── registration ──────────────────────────────────────────────────────────────

def register():
    """Call this once before gym.make('Tetris-v0')."""
    if "Tetris-v0" not in gym.envs.registry:
        gym.register(
            id="Tetris-v0",
            entry_point="envs.tetris_env:TetrisEnv",
            kwargs={"max_steps": 1000},
        )
