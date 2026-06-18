"""
Snake-v0: A gymnasium-compatible Snake environment for ASTRA training missions.

Observation: flattened (grid_h * grid_w,) float32 array
  0.0 = empty, 0.5 = snake body, 1.0 = snake head, -1.0 = food

Action space: Discrete(4) — 0=UP, 1=RIGHT, 2=DOWN, 3=LEFT

Reward (all components configurable via constructor kwargs):
  +food_reward   eating food            (default +10)
  +death_penalty dying                  (default -10, passed as negative)
  +survival_bonus every step alive      (default +0.1)
  +/-distance_weight step toward/away food (default 1.0; set 0 to disable shaping)

Episode terminates on wall/self collision or after max_steps.
Solved threshold: mean_reward >= 50 (roughly food_items = 5 per episode).
"""
from __future__ import annotations

from collections import deque
from typing import Optional, Tuple

import numpy as np
import gymnasium as gym
from gymnasium import spaces


class SnakeEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        grid_h: int = 16,
        grid_w: int = 16,
        max_steps: int = 500,
        render_mode: Optional[str] = None,
        food_reward: float = 10.0,
        death_penalty: float = -10.0,
        survival_bonus: float = 0.1,
        distance_weight: float = 1.0,
    ):
        super().__init__()
        self.grid_h = grid_h
        self.grid_w = grid_w
        self.max_steps = max_steps
        self.render_mode = render_mode
        self.food_reward = food_reward
        self.death_penalty = death_penalty
        self.survival_bonus = survival_bonus
        self.distance_weight = distance_weight

        self.action_space = spaces.Discrete(4)  # UP RIGHT DOWN LEFT
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0,
            shape=(grid_h * grid_w,),
            dtype=np.float32,
        )

        # Direction deltas: UP, RIGHT, DOWN, LEFT
        self._deltas = [(-1, 0), (0, 1), (1, 0), (0, -1)]
        self._snake: deque[Tuple[int, int]] = deque()
        self._food: Tuple[int, int] = (0, 0)
        self._direction: int = 1  # RIGHT
        self._steps: int = 0
        self._prev_dist: float = 0.0

    # ── gymnasium API ──────────────────────────────────────────────────────────

    def reset(self, *, seed: Optional[int] = None, options=None):
        super().reset(seed=seed)
        mid_r, mid_c = self.grid_h // 2, self.grid_w // 2
        self._snake = deque([(mid_r, mid_c - 1), (mid_r, mid_c), (mid_r, mid_c + 1)])
        self._direction = 1  # facing RIGHT
        self._steps = 0
        self._place_food()
        self._prev_dist = self._dist_to_food()
        return self._obs(), {}

    def step(self, action: int):
        # Ignore 180° reversal
        if (action + 2) % 4 != self._direction:
            self._direction = action

        dr, dc = self._deltas[self._direction]
        head_r, head_c = self._snake[-1]
        new_head = (head_r + dr, head_c + dc)

        # Collision check
        dead = (
            not (0 <= new_head[0] < self.grid_h and 0 <= new_head[1] < self.grid_w)
            or new_head in self._snake
        )
        if dead:
            return self._obs(), self.death_penalty, True, False, {}

        self._snake.append(new_head)
        self._steps += 1

        reward = self.survival_bonus

        if new_head == self._food:
            reward += self.food_reward
            self._place_food()
        else:
            self._snake.popleft()

        # Distance shaping (disabled when distance_weight == 0)
        if self.distance_weight != 0.0:
            dist = self._dist_to_food()
            reward += self.distance_weight if dist < self._prev_dist else -self.distance_weight
            self._prev_dist = dist

        truncated = self._steps >= self.max_steps
        return self._obs(), reward, False, truncated, {}

    # ── helpers ────────────────────────────────────────────────────────────────

    def _place_food(self):
        snake_set = set(self._snake)
        empty = [
            (r, c)
            for r in range(self.grid_h)
            for c in range(self.grid_w)
            if (r, c) not in snake_set
        ]
        idx = self.np_random.integers(len(empty))
        self._food = empty[idx]
        self._prev_dist = self._dist_to_food()

    def _dist_to_food(self) -> float:
        hr, hc = self._snake[-1]
        fr, fc = self._food
        return float(abs(hr - fr) + abs(hc - fc))

    def _obs(self) -> np.ndarray:
        grid = np.zeros((self.grid_h, self.grid_w), dtype=np.float32)
        for r, c in self._snake:
            grid[r, c] = 0.5
        hr, hc = self._snake[-1]
        grid[hr, hc] = 1.0
        fr, fc = self._food
        grid[fr, fc] = -1.0
        return grid.flatten()


# ── gymnasium registration ────────────────────────────────────────────────────

def register():
    """Call this once before gym.make('Snake-v0')."""
    if "Snake-v0" not in gym.envs.registry:
        gym.register(
            id="Snake-v0",
            entry_point="envs.snake_env:SnakeEnv",
            kwargs={"grid_h": 16, "grid_w": 16, "max_steps": 500},
        )
