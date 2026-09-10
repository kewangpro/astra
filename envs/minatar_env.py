from __future__ import annotations

from typing import Optional, Dict
import numpy as np
import gymnasium as gym
from gymnasium import spaces


class MinAtarBreakoutEnv(gym.Env):
    """
    Gymnasium environment for MinAtar Breakout (Young & Tian, 2019).

    A minimalist, fast 10x10 symbolic arcade environment.
    Grid: 10 rows x 10 cols.
    Actions:
      0: NOOP
      1: LEFT
      2: RIGHT

    Observation Space:
      Box(low=0.0, high=1.0, shape=(400,), dtype=np.float32)
      4 flattened 10x10 binary channels:
        [0..99]    Paddle plane
        [100..199] Ball plane
        [200..299] Brick plane
        [300..399] Ball velocity/direction plane
    """

    metadata = {"render_modes": []}

    ROWS = 10
    COLS = 10
    PADDLE_WIDTH = 2
    PADDLE_ROW = 9
    BRICK_ROWS = (1, 2, 3)

    def __init__(
        self,
        max_steps: int = 1500,
        render_mode: Optional[str] = None,
        brick_reward: float = 1.0,
        paddle_hit_reward: float = 0.1,
        death_penalty: float = -1.0,
        **kwargs,
    ):
        super().__init__()
        self.max_steps = max_steps
        self.render_mode = render_mode
        self.brick_reward = brick_reward
        self.paddle_hit_reward = paddle_hit_reward
        self.death_penalty = death_penalty

        self.action_space = spaces.Discrete(3)  # 0: NOOP, 1: LEFT, 2: RIGHT
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(self.ROWS * self.COLS * 4,),
            dtype=np.float32,
        )

        self._bricks = np.zeros((self.ROWS, self.COLS), dtype=np.int8)
        self._paddle_col = 4
        self._ball_r = 7
        self._ball_c = 4
        self._ball_dr = -1
        self._ball_dc = 1
        self._steps = 0
        self._score = 0.0
        self._bricks_cleared = 0

    def reset(self, *, seed: Optional[int] = None, options=None):
        super().reset(seed=seed)
        self._steps = 0
        self._score = 0.0
        self._bricks_cleared = 0

        # Initialize bricks
        self._init_bricks()

        # Initialize paddle
        self._paddle_col = int(self.np_random.integers(0, self.COLS - self.PADDLE_WIDTH + 1))

        # Initialize ball
        self._ball_r = 6
        self._ball_c = self._paddle_col
        self._ball_dr = -1
        self._ball_dc = int(self.np_random.choice([-1, 1]))

        return self._get_obs(), {}

    def _init_bricks(self) -> None:
        self._bricks = np.zeros((self.ROWS, self.COLS), dtype=np.int8)
        for r in self.BRICK_ROWS:
            self._bricks[r, :] = 1

    def _get_obs(self) -> np.ndarray:
        planes = np.zeros((4, self.ROWS, self.COLS), dtype=np.float32)

        # Plane 0: Paddle
        for c in range(self._paddle_col, self._paddle_col + self.PADDLE_WIDTH):
            if 0 <= c < self.COLS:
                planes[0, self.PADDLE_ROW, c] = 1.0

        # Plane 1: Ball
        if 0 <= self._ball_r < self.ROWS and 0 <= self._ball_c < self.COLS:
            planes[1, self._ball_r, self._ball_c] = 1.0

        # Plane 2: Bricks
        planes[2] = self._bricks.astype(np.float32)

        # Plane 3: Ball velocity direction (1.0 if moving downward, 0.5 if upward)
        if 0 <= self._ball_r < self.ROWS and 0 <= self._ball_c < self.COLS:
            planes[3, self._ball_r, self._ball_c] = 1.0 if self._ball_dr > 0 else 0.5

        return planes.flatten()

    def step(self, action: int):
        action = int(action)
        self._steps += 1
        reward = 0.0
        terminated = False

        # 1. Update paddle position
        if action == 1:  # LEFT
            self._paddle_col = max(0, self._paddle_col - 1)
        elif action == 2:  # RIGHT
            self._paddle_col = min(self.COLS - self.PADDLE_WIDTH, self._paddle_col + 1)

        # 2. Update ball position
        # Check horizontal bounce
        next_c = self._ball_c + self._ball_dc
        if next_c < 0:
            next_c = 0
            self._ball_dc = 1
        elif next_c >= self.COLS:
            next_c = self.COLS - 1
            self._ball_dc = -1

        next_r = self._ball_r + self._ball_dr

        # Check top bounce
        if next_r < 0:
            next_r = 0
            self._ball_dr = 1

        # Check brick collision
        if 0 <= next_r < self.ROWS and 0 <= next_c < self.COLS and self._bricks[next_r, next_c] == 1:
            self._bricks[next_r, next_c] = 0
            reward += self.brick_reward
            self._bricks_cleared += 1
            self._ball_dr = -self._ball_dr

            # Check if all bricks cleared
            if np.sum(self._bricks) == 0:
                reward += 10.0
                self._init_bricks()

        # Check paddle hit or bottom fall
        elif next_r >= self.PADDLE_ROW:
            if self._paddle_col <= next_c < self._paddle_col + self.PADDLE_WIDTH:
                # Hit paddle!
                next_r = self.PADDLE_ROW - 1
                self._ball_dr = -1
                # Angle deflection based on hit position
                if next_c == self._paddle_col:
                    self._ball_dc = -1
                else:
                    self._ball_dc = 1
                reward += self.paddle_hit_reward
            else:
                # Missed paddle, ball lost
                terminated = True
                reward += self.death_penalty

        self._ball_r = min(self.ROWS - 1, max(0, next_r))
        self._ball_c = min(self.COLS - 1, max(0, next_c))
        self._score += reward

        truncated = self._steps >= self.max_steps
        info = {
            "score": round(self._score, 2),
            "bricks_cleared": self._bricks_cleared,
            "steps": self._steps,
        }

        return self._get_obs(), reward, terminated, truncated, info

    def get_viewer_grid(self) -> list[int]:
        """
        Return 100-element list representing 10x10 board:
          0: empty
          1: paddle
          2: ball
          3: brick
        """
        grid = np.zeros((self.ROWS, self.COLS), dtype=np.int8)

        # Bricks
        grid[self._bricks == 1] = 3

        # Paddle
        for c in range(self._paddle_col, self._paddle_col + self.PADDLE_WIDTH):
            if 0 <= c < self.COLS:
                grid[self.PADDLE_ROW, c] = 1

        # Ball
        if 0 <= self._ball_r < self.ROWS and 0 <= self._ball_c < self.COLS:
            grid[self._ball_r, self._ball_c] = 2

        return grid.flatten().tolist()


def register():
    """Register MinAtar-Breakout-v0 with Gymnasium."""
    if "MinAtar-Breakout-v0" not in gym.envs.registry:
        gym.register(
            id="MinAtar-Breakout-v0",
            entry_point="envs.minatar_env:MinAtarBreakoutEnv",
            kwargs={"max_steps": 1500},
        )
    if "MinAtar-v0" not in gym.envs.registry:
        gym.register(
            id="MinAtar-v0",
            entry_point="envs.minatar_env:MinAtarBreakoutEnv",
            kwargs={"max_steps": 1500},
        )
