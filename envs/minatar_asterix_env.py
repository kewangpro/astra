from __future__ import annotations

from typing import List, Optional, Tuple
import numpy as np
import gymnasium as gym
from gymnasium import spaces


# kenjyoung/MinAtar asterix.py
_RAMP_INTERVAL = 100
_INIT_SPAWN_SPEED = 10
_INIT_MOVE_INTERVAL = 5


class MinAtarAsterixEnv(gym.Env):
    """
    Gymnasium environment for MinAtar Asterix (Young & Tian, 2019).
    Pure Python/NumPy implementation of kenjyoung/MinAtar asterix.py.

    Grid: 10×10. Player moves on rows 1–8. Enemies and gold spawn from the
    left (col 0) or right (col 9) into eight row slots. Trail marks heading.
    +gold_reward per treasure; enemy contact is terminal.

    Actions (original MinAtar order n,l,u,r,d,f):
      0: NOOP
      1: LEFT
      2: UP
      3: RIGHT
      4: DOWN
      5: FIRE (unused / noop)

    Observations:
      Box(0.0, 1.0, shape=(400,), dtype=np.float32)
      4 flattened 10×10 planes: player, enemy, trail, gold.
    """

    metadata = {"render_modes": []}

    ROWS = 10
    COLS = 10
    N_SLOTS = 8

    def __init__(
        self,
        max_steps: int = 1000,
        render_mode: Optional[str] = None,
        gold_reward: float = 1.0,
        death_penalty: float = -1.0,
        ramping: bool = True,
        **kwargs,
    ):
        super().__init__()
        self.max_steps = max_steps
        self.render_mode = render_mode
        self.gold_reward = gold_reward
        self.death_penalty = death_penalty
        self.ramping = ramping

        self.action_space = spaces.Discrete(6)
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(self.ROWS * self.COLS * 4,),
            dtype=np.float32,
        )

        self._player_x = 5
        self._player_y = 5
        self._entities: List[Optional[list]] = [None] * self.N_SLOTS
        self._spawn_speed = _INIT_SPAWN_SPEED
        self._spawn_timer = _INIT_SPAWN_SPEED
        self._move_speed = _INIT_MOVE_INTERVAL
        self._move_timer = _INIT_MOVE_INTERVAL
        self._ramp_timer = _RAMP_INTERVAL
        self._ramp_index = 0
        self._terminal = False
        self._score = 0.0
        self._gold_collected = 0
        self._step_count = 0
        self._rng: np.random.Generator = np.random.default_rng()

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        self._rng = np.random.default_rng(seed)
        self._player_x = 5
        self._player_y = 5
        self._entities = [None] * self.N_SLOTS
        self._spawn_speed = _INIT_SPAWN_SPEED
        self._spawn_timer = self._spawn_speed
        self._move_speed = _INIT_MOVE_INTERVAL
        self._move_timer = self._move_speed
        self._ramp_timer = _RAMP_INTERVAL
        self._ramp_index = 0
        self._terminal = False
        self._score = 0.0
        self._gold_collected = 0
        self._step_count = 0
        return self._get_obs(), self._info()

    def _spawn_entity(self) -> None:
        lr = bool(self._rng.random() < 0.5)
        is_gold = bool(self._rng.random() < 1.0 / 3.0)
        x = 0 if lr else 9
        slot_options = [i for i, e in enumerate(self._entities) if e is None]
        if not slot_options:
            return
        slot = int(slot_options[int(self._rng.integers(len(slot_options)))])
        self._entities[slot] = [x, slot + 1, lr, is_gold]

    def _touch(self, entity: list) -> float:
        if entity[0:2] != [self._player_x, self._player_y]:
            return 0.0
        if entity[3]:
            return self.gold_reward
        self._terminal = True
        return 0.0

    def step(self, action: int):
        reward = 0.0
        if self._terminal:
            return self._get_obs(), 0.0, True, False, self._info()

        a = int(action)
        if self._spawn_timer == 0:
            self._spawn_entity()
            self._spawn_timer = self._spawn_speed

        if a == 1:
            self._player_x = max(0, self._player_x - 1)
        elif a == 3:
            self._player_x = min(9, self._player_x + 1)
        elif a == 2:
            self._player_y = max(1, self._player_y - 1)
        elif a == 4:
            self._player_y = min(8, self._player_y + 1)

        for i, ent in enumerate(self._entities):
            if ent is None:
                continue
            r = self._touch(ent)
            if r > 0:
                self._entities[i] = None
                reward += r
                self._gold_collected += 1
            elif self._terminal:
                reward += self.death_penalty
                break

        if self._move_timer == 0 and not self._terminal:
            self._move_timer = self._move_speed
            for i, ent in enumerate(self._entities):
                if ent is None:
                    continue
                ent[0] += 1 if ent[2] else -1
                if ent[0] < 0 or ent[0] > 9:
                    self._entities[i] = None
                    continue
                r = self._touch(ent)
                if r > 0:
                    self._entities[i] = None
                    reward += r
                    self._gold_collected += 1
                elif self._terminal:
                    reward += self.death_penalty
                    break

        self._spawn_timer -= 1
        self._move_timer -= 1

        if self.ramping and (self._spawn_speed > 1 or self._move_speed > 1):
            if self._ramp_timer >= 0:
                self._ramp_timer -= 1
            else:
                if self._move_speed > 1 and self._ramp_index % 2:
                    self._move_speed -= 1
                if self._spawn_speed > 1:
                    self._spawn_speed -= 1
                self._ramp_index += 1
                self._ramp_timer = _RAMP_INTERVAL

        self._step_count += 1
        self._score += reward
        truncated = self._step_count >= self.max_steps
        terminated = self._terminal
        return self._get_obs(), float(reward), terminated, truncated, self._info()

    def _get_obs(self) -> np.ndarray:
        planes = np.zeros((4, self.ROWS, self.COLS), dtype=np.float32)
        planes[0, self._player_y, self._player_x] = 1.0
        for ent in self._entities:
            if ent is None:
                continue
            x, y, lr, is_gold = ent
            planes[3 if is_gold else 1, y, x] = 1.0
            back_x = x - 1 if lr else x + 1
            if 0 <= back_x <= 9:
                planes[2, y, back_x] = 1.0
        return planes.reshape(-1)

    def get_viewer_grid(self) -> list:
        grid = np.zeros((self.ROWS, self.COLS), dtype=np.int32)
        for ent in self._entities:
            if ent is None:
                continue
            x, y, lr, is_gold = ent
            back_x = x - 1 if lr else x + 1
            if 0 <= back_x <= 9:
                grid[y, back_x] = 4
            grid[y, x] = 3 if is_gold else 2
        grid[self._player_y, self._player_x] = 1
        return grid.flatten().tolist()

    def _info(self) -> dict:
        return {
            "score": float(self._score),
            "gold_collected": int(self._gold_collected),
            "ramp_index": int(self._ramp_index),
        }


def register():
    if "MinAtar-Asterix-v0" not in gym.envs.registration.registry:
        gym.register(
            id="MinAtar-Asterix-v0",
            entry_point="envs.minatar_asterix_env:MinAtarAsterixEnv",
        )


register()
