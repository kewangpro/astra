from __future__ import annotations

from typing import Optional, Dict, List, Tuple
import numpy as np
import gymnasium as gym
from gymnasium import spaces


class MinAtarSpaceInvadersEnv(gym.Env):
    """
    Gymnasium environment for MinAtar Space Invaders (Young & Tian, 2019).
    Pure Python/NumPy implementation running at >50,000 steps/sec.

    Grid: 10 rows x 10 cols.
    Actions:
      0: NOOP
      1: LEFT
      2: RIGHT
      3: FIRE

    Observations:
      Box(0.0, 1.0, shape=(400,), dtype=np.float32)
      4 flattened 10x10 binary planes:
        [0..99]    Cannon plane (row 9)
        [100..199] Alien invaders plane
        [200..299] Alien bombs dropping down
        [300..399] Player laser firing up & defensive shields
    """

    metadata = {"render_modes": []}

    ROWS = 10
    COLS = 10
    CANNON_ROW = 9
    SHIELD_ROW = 8

    def __init__(
        self,
        max_steps: int = 1500,
        render_mode: Optional[str] = None,
        alien_kill_reward: float = 1.0,
        wave_clear_bonus: float = 5.0,
        death_penalty: float = -1.0,
        **kwargs,
    ):
        super().__init__()
        self.max_steps = max_steps
        self.render_mode = render_mode
        self.alien_kill_reward = alien_kill_reward
        self.wave_clear_bonus = wave_clear_bonus
        self.death_penalty = death_penalty

        self.action_space = spaces.Discrete(4)  # 0: NOOP, 1: LEFT, 2: RIGHT, 3: FIRE
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(self.ROWS * self.COLS * 4,),
            dtype=np.float32,
        )

        self._cannon_x: int = 4
        self._aliens: np.ndarray = np.zeros((self.ROWS, self.COLS), dtype=bool)
        self._alien_dir: int = 1  # 1: right, -1: left
        self._alien_move_timer: int = 0
        self._alien_move_freq: int = 4  # move aliens every N steps
        self._alien_bombs: List[List[int]] = []  # [[r, c], ...]
        self._player_lasers: List[List[int]] = []  # [[r, c], ...]
        self._shields: np.ndarray = np.zeros(self.COLS, dtype=int)  # hitpoints at SHIELD_ROW

        self._score: float = 0.0
        self._aliens_killed: int = 0
        self._step_count: int = 0

    def _init_wave(self):
        """Spawn initial alien fleet."""
        self._aliens.fill(False)
        # 3 rows of aliens across cols 1..8
        for r in range(1, 4):
            for c in range(1, 9, 2):
                self._aliens[r, c] = True
        self._alien_dir = 1
        self._alien_move_timer = 0

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        self._cannon_x = 4
        self._init_wave()
        self._alien_bombs.clear()
        self._player_lasers.clear()
        # Shields at cols 2, 4, 7 with 2 hitpoints each
        self._shields.fill(0)
        self._shields[2] = 2
        self._shields[5] = 2
        self._shields[7] = 2

        self._score = 0.0
        self._aliens_killed = 0
        self._step_count = 0

        return self._get_obs(), {"score": self._score, "aliens_killed": self._aliens_killed}

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        self._step_count += 1
        reward = 0.0
        terminated = False

        # 1. Player Cannon Action
        if action == 1:  # LEFT
            self._cannon_x = max(0, self._cannon_x - 1)
        elif action == 2:  # RIGHT
            self._cannon_x = min(self.COLS - 1, self._cannon_x + 1)
        elif action == 3:  # FIRE
            if len(self._player_lasers) < 3:
                self._player_lasers.append([self.CANNON_ROW, self._cannon_x])

        # 2. Update Player Lasers (move up)
        new_lasers = []
        for r, c in self._player_lasers:
            nr = r - 1
            if nr < 0:
                continue
            # Check hit shield
            if nr == self.SHIELD_ROW and self._shields[c] > 0:
                self._shields[c] -= 1
                continue
            # Check hit alien
            if self._aliens[nr, c]:
                self._aliens[nr, c] = False
                self._aliens_killed += 1
                reward += self.alien_kill_reward
                self._score += self.alien_kill_reward
                continue
            new_lasers.append([nr, c])
        self._player_lasers = new_lasers

        # 3. Wave cleared bonus & respawn
        if not np.any(self._aliens):
            reward += self.wave_clear_bonus
            self._score += self.wave_clear_bonus
            self._init_wave()

        # 4. Alien Movement
        self._alien_move_timer += 1
        if self._alien_move_timer >= self._alien_move_freq:
            self._alien_move_timer = 0
            # Check if moving in alien_dir hits border
            alien_coords = np.argwhere(self._aliens)
            if len(alien_coords) > 0:
                cols = alien_coords[:, 1]
                hit_right = self._alien_dir == 1 and np.max(cols) >= self.COLS - 1
                hit_left = self._alien_dir == -1 and np.min(cols) <= 0
                if hit_right or hit_left:
                    # Drop aliens down 1 row and reverse
                    new_aliens = np.zeros_like(self._aliens)
                    for r, c in alien_coords:
                        if r + 1 < self.ROWS:
                            new_aliens[r + 1, c] = True
                    self._aliens = new_aliens
                    self._alien_dir *= -1
                else:
                    # Shift sideways
                    new_aliens = np.zeros_like(self._aliens)
                    for r, c in alien_coords:
                        new_aliens[r, c + self._alien_dir] = True
                    self._aliens = new_aliens

            # Check if aliens reached cannon level
            if np.any(self._aliens[self.CANNON_ROW, :]):
                terminated = True
                reward += self.death_penalty

            # Chance for alien to drop bomb
            if len(alien_coords) > 0 and self.np_random.random() < 0.4:
                shooter = alien_coords[self.np_random.integers(len(alien_coords))]
                if len(self._alien_bombs) < 3:
                    self._alien_bombs.append([shooter[0] + 1, shooter[1]])

        # 5. Update Alien Bombs (move down)
        new_bombs = []
        for r, c in self._alien_bombs:
            nr = r + 1
            if nr >= self.ROWS:
                continue
            # Check hit shield
            if nr == self.SHIELD_ROW and self._shields[c] > 0:
                self._shields[c] -= 1
                continue
            # Check hit cannon
            if nr == self.CANNON_ROW and c == self._cannon_x:
                terminated = True
                reward += self.death_penalty
                continue
            new_bombs.append([nr, c])
        self._alien_bombs = new_bombs

        truncated = self._step_count >= self.max_steps
        info = {
            "score": self._score,
            "aliens_killed": self._aliens_killed,
            "step": self._step_count,
        }

        return self._get_obs(), reward, terminated, truncated, info

    def _get_obs(self) -> np.ndarray:
        obs = np.zeros((4, self.ROWS, self.COLS), dtype=np.float32)
        # Channel 0: Cannon
        obs[0, self.CANNON_ROW, self._cannon_x] = 1.0
        # Channel 1: Aliens
        obs[1] = self._aliens.astype(np.float32)
        # Channel 2: Alien Bombs
        for r, c in self._alien_bombs:
            if 0 <= r < self.ROWS and 0 <= c < self.COLS:
                obs[2, r, c] = 1.0
        # Channel 3: Player Lasers & Shields
        for r, c in self._player_lasers:
            if 0 <= r < self.ROWS and 0 <= c < self.COLS:
                obs[3, r, c] = 1.0
        for c in range(self.COLS):
            if self._shields[c] > 0:
                obs[3, self.SHIELD_ROW, c] = float(self._shields[c]) / 2.0

        return obs.flatten()

    def get_viewer_grid(self) -> List[int]:
        """Categorical grid for HUD canvas rendering (100 ints)."""
        grid = np.zeros((self.ROWS, self.COLS), dtype=int)
        # Shields: 5
        for c in range(self.COLS):
            if self._shields[c] > 0:
                grid[self.SHIELD_ROW, c] = 5
        # Aliens: 2
        grid[self._aliens] = 2
        # Player Lasers: 4
        for r, c in self._player_lasers:
            if 0 <= r < self.ROWS and 0 <= c < self.COLS:
                grid[r, c] = 4
        # Alien Bombs: 3
        for r, c in self._alien_bombs:
            if 0 <= r < self.ROWS and 0 <= c < self.COLS:
                grid[r, c] = 3
        # Cannon: 1
        grid[self.CANNON_ROW, self._cannon_x] = 1

        return grid.flatten().tolist()


def register():
    """Register Space Invaders environments with Gymnasium."""
    for env_id in ("MinAtar-SpaceInvaders-v0", "MinAtar-Space-Invaders-v0"):
        if env_id not in gym.envs.registration.registry:
            gym.register(
                id=env_id,
                entry_point="envs.minatar_space_invaders_env:MinAtarSpaceInvadersEnv",
            )


register()
