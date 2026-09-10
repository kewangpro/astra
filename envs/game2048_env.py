from __future__ import annotations

import copy
from typing import Optional, Dict, Tuple
import numpy as np
import gymnasium as gym
from gymnasium import spaces


class Game2048Env(gym.Env):
    """
    Gymnasium environment for the classic 2048 puzzle game.

    Board: 4x4 grid with powers of 2 (0, 2, 4, 8, 16, 32, ..., 2048+).
    Actions:
      0: UP
      1: DOWN
      2: LEFT
      3: RIGHT

    Observation Space:
      Box(low=0.0, high=1.0, shape=(16,), dtype=np.float32)
      Normalized log2 representation: cell value > 0 -> log2(val) / 16.0, else 0.0.

    Reward:
      Points scored from merging tiles on each step + optional reward shaping.
    """

    metadata = {"render_modes": []}

    SIZE = 4

    def __init__(
        self,
        max_steps: int = 2000,
        render_mode: Optional[str] = None,
        merge_multiplier: float = 1.0,
        empty_tile_bonus: float = 0.0,
        corner_bonus: float = 0.0,
        **kwargs,
    ):
        super().__init__()
        self.max_steps = max_steps
        self.render_mode = render_mode
        self.merge_multiplier = merge_multiplier
        self.empty_tile_bonus = empty_tile_bonus
        self.corner_bonus = corner_bonus

        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(self.SIZE * self.SIZE,),
            dtype=np.float32,
        )

        self._board = np.zeros((self.SIZE, self.SIZE), dtype=np.int32)
        self._score = 0
        self._steps = 0
        self._max_tile = 0

    def reset(self, *, seed: Optional[int] = None, options=None):
        super().reset(seed=seed)
        self._board = np.zeros((self.SIZE, self.SIZE), dtype=np.int32)
        self._score = 0
        self._steps = 0
        self._max_tile = 0

        # Spawn initial 2 tiles
        self._spawn_tile()
        self._spawn_tile()
        self._update_max_tile()

        return self._get_obs(), {}

    def _get_obs(self) -> np.ndarray:
        """Normalized log2 observation: 2048 -> 11/16, 4096 -> 12/16."""
        flat = self._board.flatten()
        obs = np.zeros_like(flat, dtype=np.float32)
        nonzero = flat > 0
        obs[nonzero] = np.log2(flat[nonzero]) / 16.0
        return np.clip(obs, 0.0, 1.0).astype(np.float32)

    def _update_max_tile(self) -> None:
        self._max_tile = int(np.max(self._board))

    def _spawn_tile(self) -> bool:
        empty = list(zip(*np.where(self._board == 0)))
        if not empty:
            return False
        idx = int(self.np_random.integers(len(empty)))
        r, c = empty[idx]
        self._board[r, c] = 2 if self.np_random.random() < 0.9 else 4
        return True

    @staticmethod
    def _slide_left_row(row: np.ndarray) -> Tuple[np.ndarray, int]:
        """Slide and merge a single 1D row to the left."""
        nonzeros = row[row != 0]
        merged = []
        score = 0
        skip = False
        for i in range(len(nonzeros)):
            if skip:
                skip = False
                continue
            if i + 1 < len(nonzeros) and nonzeros[i] == nonzeros[i + 1]:
                val = int(nonzeros[i] * 2)
                merged.append(val)
                score += val
                skip = True
            else:
                merged.append(int(nonzeros[i]))
        res = np.zeros(len(row), dtype=np.int32)
        res[:len(merged)] = merged
        return res, score

    def _simulate_move(self, board: np.ndarray, action: int) -> Tuple[np.ndarray, int, bool]:
        """
        Simulate move for a given board without modifying self._board.
        Returns: (new_board, score_gained, moved)
        0: UP, 1: DOWN, 2: LEFT, 3: RIGHT
        """
        b = board.copy()
        total_score = 0

        if action == 0:  # UP
            b = b.T
            for r in range(self.SIZE):
                b[r], s = self._slide_left_row(b[r])
                total_score += s
            b = b.T
        elif action == 1:  # DOWN
            b = b.T
            for r in range(self.SIZE):
                row_rev = b[r, ::-1]
                slid, s = self._slide_left_row(row_rev)
                b[r] = slid[::-1]
                total_score += s
            b = b.T
        elif action == 2:  # LEFT
            for r in range(self.SIZE):
                b[r], s = self._slide_left_row(b[r])
                total_score += s
        elif action == 3:  # RIGHT
            for r in range(self.SIZE):
                row_rev = b[r, ::-1]
                slid, s = self._slide_left_row(row_rev)
                b[r] = slid[::-1]
                total_score += s

        moved = not np.array_equal(board, b)
        return b, total_score, moved

    def _has_moves_left(self) -> bool:
        """Check if any legal move is available on self._board."""
        if np.any(self._board == 0):
            return True
        # Check horizontal neighbors
        if np.any(self._board[:, :-1] == self._board[:, 1:]):
            return True
        # Check vertical neighbors
        if np.any(self._board[:-1, :] == self._board[1:, :]):
            return True
        return False

    def get_next_states(self) -> Dict[int, np.ndarray]:
        """
        Lookahead support: returns deterministic next states for valid slide actions
        before random tile spawn, enabling Lookahead DQN / expectimax evaluation.
        """
        res = {}
        for act in range(4):
            nb, _, moved = self._simulate_move(self._board, act)
            if moved:
                flat = nb.flatten()
                obs = np.zeros_like(flat, dtype=np.float32)
                nonzero = flat > 0
                obs[nonzero] = np.log2(flat[nonzero]) / 16.0
                res[act] = np.clip(obs, 0.0, 1.0).astype(np.float32)
        return res

    def step(self, action: int):
        action = int(action)
        self._steps += 1

        new_board, move_score, moved = self._simulate_move(self._board, action)

        if moved:
            self._board = new_board
            self._score += move_score
            self._spawn_tile()
            self._update_max_tile()

            # Base reward from merges
            reward = float(move_score) * self.merge_multiplier

            # Optional reward shaping
            if self.empty_tile_bonus > 0:
                empty_count = int(np.sum(self._board == 0))
                reward += empty_count * self.empty_tile_bonus

            if self.corner_bonus > 0:
                corners = [self._board[0, 0], self._board[0, 3], self._board[3, 0], self._board[3, 3]]
                if self._max_tile in corners:
                    reward += self.corner_bonus
        else:
            # Invalid move (no tiles slid)
            reward = -1.0

        terminated = not self._has_moves_left()
        truncated = self._steps >= self.max_steps
        done = terminated or truncated

        info = {
            "score": self._score,
            "max_tile": self._max_tile,
            "steps": self._steps,
            "empty_cells": int(np.sum(self._board == 0)),
        }

        return self._get_obs(), reward, terminated, truncated, info

    def get_viewer_grid(self) -> list[int]:
        """Return flat 16-element list of tile values for UI rendering."""
        return self._board.flatten().tolist()


def register():
    """Register Game2048-v0 with Gymnasium."""
    if "Game2048-v0" not in gym.envs.registry:
        gym.register(
            id="Game2048-v0",
            entry_point="envs.game2048_env:Game2048Env",
            kwargs={"max_steps": 2000},
        )
