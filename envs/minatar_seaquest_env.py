from __future__ import annotations

from typing import Optional, Dict, List, Tuple
import numpy as np
import gymnasium as gym
from gymnasium import spaces


class MinAtarSeaquestEnv(gym.Env):
    """
    Gymnasium environment for MinAtar Seaquest (Young & Tian, 2019).
    Pure Python/NumPy implementation running at >50,000 steps/sec.

    Grid: 10 rows x 10 cols.
      Row 0: Surface water (oxygen refill & diver deposit)
      Rows 1..8: Ocean depths (enemies, divers, torpedoes)
      Row 9: Sea floor

    Actions:
      0: NOOP
      1: LEFT
      2: RIGHT
      3: UP
      4: DOWN
      5: FIRE (torpedo horizontally in facing direction)

    Observations:
      Box(0.0, 1.0, shape=(400,), dtype=np.float32)
      4 flattened 10x10 binary/normalized planes:
        [0..99]    Player submarine position & facing nozzle
        [100..199] Enemies (sharks and enemy subs)
        [200..299] Divers (swimming divers & held count gauge on row 9)
        [300..399] Torpedoes (player & enemy) & surface oxygen gauge (row 0)
    """

    metadata = {"render_modes": []}

    ROWS = 10
    COLS = 10
    SURFACE_ROW = 0
    MAX_DIVERS_HELD = 6

    def __init__(
        self,
        max_steps: int = 1500,
        render_mode: Optional[str] = None,
        enemy_kill_reward: float = 1.0,
        diver_pickup_reward: float = 0.5,
        diver_rescue_reward: float = 2.0,
        wave_clear_bonus: float = 5.0,
        death_penalty: float = -1.0,
        oxygen_max: int = 200,
        **kwargs,
    ):
        super().__init__()
        self.max_steps = max_steps
        self.render_mode = render_mode
        self.enemy_kill_reward = enemy_kill_reward
        self.diver_pickup_reward = diver_pickup_reward
        self.diver_rescue_reward = diver_rescue_reward
        self.wave_clear_bonus = wave_clear_bonus
        self.death_penalty = death_penalty
        self.oxygen_max = oxygen_max

        self.action_space = spaces.Discrete(6)  # NOOP, LEFT, RIGHT, UP, DOWN, FIRE
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(self.ROWS * self.COLS * 4,),
            dtype=np.float32,
        )

        # Player state
        self._sub_r: int = 5
        self._sub_c: int = 2
        self._facing: int = 1  # 1: right, -1: left
        self._oxygen: int = self.oxygen_max
        self._divers_held: int = 0

        # Entities
        # Enemy: {"r": int, "c": float, "dir": int, "type": "shark" | "sub", "timer": int, "freq": int}
        self._enemies: List[Dict] = []
        # Diver: {"r": int, "c": float, "dir": int, "timer": int, "freq": int}
        self._divers: List[Dict] = []
        # Torpedoes: {"r": int, "c": int, "dir": int}
        self._player_torpedoes: List[Dict] = []
        self._enemy_torpedoes: List[Dict] = []

        # Episode metrics
        self._score: float = 0.0
        self._enemies_killed: int = 0
        self._divers_saved: int = 0
        self._step_count: int = 0
        self._destroyed: bool = False
        self._rng: np.random.Generator = np.random.default_rng()

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        self._rng = np.random.default_rng(seed)

        self._sub_r = 5
        self._sub_c = 2
        self._facing = 1
        self._oxygen = self.oxygen_max
        self._divers_held = 0

        self._enemies.clear()
        self._divers.clear()
        self._player_torpedoes.clear()
        self._enemy_torpedoes.clear()

        # Seed initial enemy and diver
        self._spawn_enemy()
        self._spawn_diver()

        self._score = 0.0
        self._enemies_killed = 0
        self._divers_saved = 0
        self._step_count = 0
        self._destroyed = False

        return self._get_obs(), {
            "score": self._score,
            "enemies_killed": self._enemies_killed,
            "divers_saved": self._divers_saved,
            "oxygen": self._oxygen,
        }

    def _spawn_enemy(self):
        if len(self._enemies) >= 4:
            return
        r = int(self._rng.integers(1, 9))
        dir_val = int(self._rng.choice([-1, 1]))
        c = 0.0 if dir_val == 1 else float(self.COLS - 1)
        etype = "shark" if self._rng.random() < 0.5 else "sub"
        freq = 2 if etype == "shark" else 3
        self._enemies.append({
            "r": r,
            "c": c,
            "dir": dir_val,
            "type": etype,
            "timer": 0,
            "freq": freq,
        })

    def _spawn_diver(self):
        if len(self._divers) >= 2 or self._divers_held >= self.MAX_DIVERS_HELD:
            return
        r = int(self._rng.integers(2, 9))
        dir_val = int(self._rng.choice([-1, 1]))
        c = 0.0 if dir_val == 1 else float(self.COLS - 1)
        self._divers.append({
            "r": r,
            "c": c,
            "dir": dir_val,
            "timer": 0,
            "freq": 3,
        })

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        self._step_count += 1
        self._oxygen -= 1
        reward = 0.0
        terminated = False
        self._destroyed = False

        # 1. Player Submarine Movement & Action
        if action == 1:  # LEFT
            self._facing = -1
            self._sub_c = max(0, self._sub_c - 1)
        elif action == 2:  # RIGHT
            self._facing = 1
            self._sub_c = min(self.COLS - 1, self._sub_c + 1)
        elif action == 3:  # UP
            self._sub_r = max(0, self._sub_r - 1)
        elif action == 4:  # DOWN
            self._sub_r = min(self.ROWS - 1, self._sub_r + 1)
        elif action == 5:  # FIRE
            if len(self._player_torpedoes) < 2:
                self._player_torpedoes.append({
                    "r": self._sub_r,
                    "c": self._sub_c + self._facing,
                    "dir": self._facing,
                })

        # 2. Oxygen Depletion Check
        if self._oxygen <= 0:
            self._destroyed = True
            reward += self.death_penalty
            terminated = True
            truncated = self._step_count >= self.max_steps
            info = {
                "score": self._score,
                "enemies_killed": self._enemies_killed,
                "divers_saved": self._divers_saved,
                "oxygen": max(0, self._oxygen),
                "step": self._step_count,
            }
            return self._get_obs(), reward, terminated, truncated, info

        # 3. Surfacing Check (at surface row 0)
        if self._sub_r == self.SURFACE_ROW:
            if self._divers_held > 0:
                rescue_reward = self._divers_held * self.diver_rescue_reward
                if self._divers_held >= self.MAX_DIVERS_HELD:
                    rescue_reward += self.wave_clear_bonus
                reward += rescue_reward
                self._score += rescue_reward
                self._divers_saved += self._divers_held
                self._divers_held = 0
                self._oxygen = self.oxygen_max  # Tank refilled!
            else:
                # Surfacing without divers: small penalty and no oxygen refill
                reward -= 0.1

        # 4. Advance Player Torpedoes & Check Collision with Enemies
        new_torps = []
        for torp in self._player_torpedoes:
            tr = torp["r"]
            tc = torp["c"] + torp["dir"]
            hit = False
            for e_idx, e in enumerate(self._enemies):
                if e["r"] == tr and int(round(e["c"])) == tc:
                    hit = True
                    self._enemies.pop(e_idx)
                    self._enemies_killed += 1
                    self._score += self.enemy_kill_reward
                    reward += self.enemy_kill_reward
                    break
            if not hit and 0 <= tc < self.COLS:
                new_torps.append({"r": tr, "c": tc, "dir": torp["dir"]})
        self._player_torpedoes = new_torps

        # 5. Move Enemies & Maybe Launch Enemy Torpedoes
        new_enemies = []
        for e in self._enemies:
            e["timer"] += 1
            if e["timer"] >= e["freq"]:
                e["timer"] = 0
                e["c"] += e["dir"]
                # Submarines occasionally shoot towards player
                if e["type"] == "sub" and len(self._enemy_torpedoes) < 2:
                    if (e["dir"] == 1 and self._sub_c > e["c"]) or (e["dir"] == -1 and self._sub_c < e["c"]):
                        if self._rng.random() < 0.20:
                            self._enemy_torpedoes.append({
                                "r": e["r"],
                                "c": int(round(e["c"])) + e["dir"],
                                "dir": e["dir"],
                            })
            # Check if still inside screen
            if 0 <= e["c"] < self.COLS:
                new_enemies.append(e)
        self._enemies = new_enemies

        # 6. Move Enemy Torpedoes
        new_enemy_torps = []
        for torp in self._enemy_torpedoes:
            tc = torp["c"] + torp["dir"]
            if 0 <= tc < self.COLS:
                new_enemy_torps.append({"r": torp["r"], "c": tc, "dir": torp["dir"]})
        self._enemy_torpedoes = new_enemy_torps

        # 7. Move Swimming Divers & Check Pickup
        new_divers = []
        for d in self._divers:
            d["timer"] += 1
            if d["timer"] >= d["freq"]:
                d["timer"] = 0
                d["c"] += d["dir"]
            # Pickup check
            if d["r"] == self._sub_r and int(round(d["c"])) == self._sub_c:
                if self._divers_held < self.MAX_DIVERS_HELD:
                    self._divers_held += 1
                    reward += self.diver_pickup_reward
                    self._score += self.diver_pickup_reward
                continue  # picked up, remove
            if 0 <= d["c"] < self.COLS:
                new_divers.append(d)
        self._divers = new_divers

        # 8. Check Collision of Player with Enemies or Enemy Torpedoes
        for e in self._enemies:
            if e["r"] == self._sub_r and int(round(e["c"])) == self._sub_c:
                self._destroyed = True
                terminated = True
                reward += self.death_penalty
                break

        if not terminated:
            for torp in self._enemy_torpedoes:
                if torp["r"] == self._sub_r and torp["c"] == self._sub_c:
                    self._destroyed = True
                    terminated = True
                    reward += self.death_penalty
                    break

        # 9. Spawning
        if self._rng.random() < 0.08:
            self._spawn_enemy()
        if self._rng.random() < 0.05:
            self._spawn_diver()

        truncated = self._step_count >= self.max_steps
        info = {
            "score": self._score,
            "enemies_killed": self._enemies_killed,
            "divers_saved": self._divers_saved,
            "divers_held": self._divers_held,
            "oxygen": max(0, self._oxygen),
            "step": self._step_count,
        }

        return self._get_obs(), reward, terminated, truncated, info

    def _get_obs(self) -> np.ndarray:
        obs = np.zeros((4, self.ROWS, self.COLS), dtype=np.float32)

        # Channel 0: Player Submarine
        obs[0, self._sub_r, self._sub_c] = 1.0
        nozzle_c = self._sub_c + self._facing
        if 0 <= nozzle_c < self.COLS:
            obs[0, self._sub_r, nozzle_c] = 0.5

        # Channel 1: Enemies
        for e in self._enemies:
            c = int(round(e["c"]))
            if 0 <= e["r"] < self.ROWS and 0 <= c < self.COLS:
                obs[1, e["r"], c] = 0.7 if e["type"] == "shark" else 1.0

        # Channel 2: Divers & Held count gauge
        for d in self._divers:
            c = int(round(d["c"]))
            if 0 <= d["r"] < self.ROWS and 0 <= c < self.COLS:
                obs[2, d["r"], c] = 1.0
        # Row 9 encodes held divers (0 to 6)
        if self._divers_held > 0:
            obs[2, 9, :self._divers_held] = 1.0

        # Channel 3: Torpedoes & Oxygen gauge on row 0
        for torp in self._player_torpedoes:
            if 0 <= torp["r"] < self.ROWS and 0 <= torp["c"] < self.COLS:
                obs[3, torp["r"], torp["c"]] = 1.0
        for torp in self._enemy_torpedoes:
            if 0 <= torp["r"] < self.ROWS and 0 <= torp["c"] < self.COLS:
                obs[3, torp["r"], torp["c"]] = 0.5
        # Oxygen gauge normalized across columns 0..9 on surface row
        oxy_cols = max(0, min(self.COLS, int(np.ceil(self.COLS * (self._oxygen / self.oxygen_max)))))
        obs[3, self.SURFACE_ROW, :oxy_cols] = 1.0

        return obs.flatten()

    def get_viewer_grid(self) -> List[int]:
        """Categorical grid for HUD canvas rendering (100 ints)."""
        grid = np.zeros((self.ROWS, self.COLS), dtype=int)

        # Surface water / Oxygen: 6
        grid[self.SURFACE_ROW, :] = 6

        # Divers: 3
        for d in self._divers:
            c = int(round(d["c"]))
            if 0 <= d["r"] < self.ROWS and 0 <= c < self.COLS:
                grid[d["r"], c] = 3

        # Enemies: 2
        for e in self._enemies:
            c = int(round(e["c"]))
            if 0 <= e["r"] < self.ROWS and 0 <= c < self.COLS:
                grid[e["r"], c] = 2

        # Player Torpedoes: 4
        for torp in self._player_torpedoes:
            if 0 <= torp["r"] < self.ROWS and 0 <= torp["c"] < self.COLS:
                grid[torp["r"], torp["c"]] = 4

        # Enemy Torpedoes: 5
        for torp in self._enemy_torpedoes:
            if 0 <= torp["r"] < self.ROWS and 0 <= torp["c"] < self.COLS:
                grid[torp["r"], torp["c"]] = 5

        # Submarine: 1 (or 7 if destroyed)
        grid[self._sub_r, self._sub_c] = 7 if self._destroyed else 1

        return grid.flatten().tolist()


def register():
    """Register Seaquest environment with Gymnasium."""
    if "MinAtar-Seaquest-v0" not in gym.envs.registration.registry:
        gym.register(
            id="MinAtar-Seaquest-v0",
            entry_point="envs.minatar_seaquest_env:MinAtarSeaquestEnv",
        )


register()
