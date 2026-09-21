from __future__ import annotations

from typing import Dict, List, Optional, Set, Tuple
import numpy as np
import gymnasium as gym
from gymnasium import spaces


# 10×10 maze: # wall, . pellet, o power pellet, space empty
_MAZE = [
    "##########",
    "#o......o#",
    "#.##..##.#",
    "#........#",
    "#.##..##.#",
    "#........#",
    "#.##..##.#",
    "#........#",
    "#o......o#",
    "##########",
]


class GridPacManEnv(gym.Env):
    """
    Small-grid Pac-Man for ASTRA: 10×10 occupancy maze, pellets, power, ghosts.

    Actions:
      0: NOOP
      1: LEFT
      2: RIGHT
      3: UP
      4: DOWN

    Observations:
      Box(0.0, 1.0, shape=(400,), dtype=np.float32)
      4 flattened 10×10 planes: walls, player, pellets (power=1), ghosts
      (frightened ghosts = 0.5).
    """

    metadata = {"render_modes": []}

    ROWS = 10
    COLS = 10
    POWER_DURATION = 15

    def __init__(
        self,
        max_steps: int = 800,
        render_mode: Optional[str] = None,
        pellet_reward: float = 1.0,
        power_reward: float = 2.0,
        ghost_eat_reward: float = 5.0,
        clear_bonus: float = 10.0,
        death_penalty: float = -1.0,
        **kwargs,
    ):
        super().__init__()
        self.max_steps = max_steps
        self.render_mode = render_mode
        self.pellet_reward = pellet_reward
        self.power_reward = power_reward
        self.ghost_eat_reward = ghost_eat_reward
        self.clear_bonus = clear_bonus
        self.death_penalty = death_penalty

        self.action_space = spaces.Discrete(5)
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(self.ROWS * self.COLS * 4,),
            dtype=np.float32,
        )

        self._walls: Set[Tuple[int, int]] = set()
        self._home_pellets: Set[Tuple[int, int]] = set()
        self._home_power: Set[Tuple[int, int]] = set()
        self._parse_maze()

        self._pellets: Set[Tuple[int, int]] = set()
        self._power: Set[Tuple[int, int]] = set()
        self._ghosts: List[Dict] = []
        self._player_r = 7
        self._player_c = 4
        # Viewer heading: 0 RIGHT, 1 DOWN, 2 LEFT, 3 UP (mouth drawn facing right).
        self._player_dir = 0
        self._power_timer = 0
        self._score = 0.0
        self._ghosts_eaten = 0
        self._pellets_eaten = 0
        self._step_count = 0
        self._rng: np.random.Generator = np.random.default_rng()

    def _parse_maze(self) -> None:
        self._walls.clear()
        self._home_pellets.clear()
        self._home_power.clear()
        for r, row in enumerate(_MAZE):
            for c, ch in enumerate(row):
                if ch == "#":
                    self._walls.add((r, c))
                elif ch == ".":
                    self._home_pellets.add((r, c))
                elif ch == "o":
                    self._home_power.add((r, c))

    def _walkable(self, r: int, c: int) -> bool:
        return 0 <= r < self.ROWS and 0 <= c < self.COLS and (r, c) not in self._walls

    def _ghost_homes(self) -> List[Tuple[int, int]]:
        return [(1, 4), (1, 5), (3, 1), (3, 8)]

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        self._rng = np.random.default_rng(seed)
        self._player_r, self._player_c = 7, 4
        self._player_dir = 0
        self._power_timer = 0
        self._pellets = set(self._home_pellets)
        self._power = set(self._home_power)
        self._ghosts = [{"r": r, "c": c, "home": (r, c)} for r, c in self._ghost_homes()]
        self._score = 0.0
        self._ghosts_eaten = 0
        self._pellets_eaten = 0
        self._step_count = 0
        return self._get_obs(), self._info()

    def _move_player(self, action: int) -> None:
        dr, dc = 0, 0
        facing = None
        if action == 1:
            dc = -1
            facing = 2  # LEFT
        elif action == 2:
            dc = 1
            facing = 0  # RIGHT
        elif action == 3:
            dr = -1
            facing = 3  # UP
        elif action == 4:
            dr = 1
            facing = 1  # DOWN
        nr, nc = self._player_r + dr, self._player_c + dc
        if self._walkable(nr, nc):
            self._player_r, self._player_c = nr, nc
            if facing is not None:
                self._player_dir = facing

    def _move_ghosts(self) -> None:
        pr, pc = self._player_r, self._player_c
        frightened = self._power_timer > 0
        for g in self._ghosts:
            options: List[Tuple[int, int]] = []
            for dr, dc in ((0, -1), (0, 1), (-1, 0), (1, 0)):
                nr, nc = g["r"] + dr, g["c"] + dc
                if self._walkable(nr, nc):
                    options.append((nr, nc))
            if not options:
                continue
            if self._rng.random() < 0.2:
                g["r"], g["c"] = options[int(self._rng.integers(len(options)))]
                continue
            def dist(p: Tuple[int, int]) -> int:
                return abs(p[0] - pr) + abs(p[1] - pc)
            ranked = max(options, key=dist) if frightened else min(options, key=dist)
            g["r"], g["c"] = ranked

    def _resolve_ghost_collisions(self) -> Tuple[float, bool]:
        reward = 0.0
        pos = (self._player_r, self._player_c)
        for g in self._ghosts:
            if (g["r"], g["c"]) != pos:
                continue
            if self._power_timer > 0:
                reward += self.ghost_eat_reward
                self._ghosts_eaten += 1
                hr, hc = g["home"]
                g["r"], g["c"] = hr, hc
            else:
                return self.death_penalty, True
        return reward, False

    def step(self, action: int):
        self._move_player(int(action))
        reward = 0.0
        pos = (self._player_r, self._player_c)
        if pos in self._pellets:
            self._pellets.remove(pos)
            reward += self.pellet_reward
            self._pellets_eaten += 1
        if pos in self._power:
            self._power.remove(pos)
            reward += self.power_reward
            self._power_timer = self.POWER_DURATION

        r_hit, dead = self._resolve_ghost_collisions()
        reward += r_hit
        if not dead:
            self._move_ghosts()
            r_hit, dead = self._resolve_ghost_collisions()
            reward += r_hit

        if self._power_timer > 0:
            self._power_timer -= 1

        if not dead and not self._pellets and not self._power:
            reward += self.clear_bonus
            self._pellets = set(self._home_pellets)
            self._power = set(self._home_power)

        self._step_count += 1
        self._score += reward
        terminated = dead
        truncated = self._step_count >= self.max_steps
        return self._get_obs(), float(reward), terminated, truncated, self._info()

    def _get_obs(self) -> np.ndarray:
        planes = np.zeros((4, self.ROWS, self.COLS), dtype=np.float32)
        for r, c in self._walls:
            planes[0, r, c] = 1.0
        planes[1, self._player_r, self._player_c] = 1.0
        for r, c in self._pellets:
            planes[2, r, c] = 1.0
        for r, c in self._power:
            planes[2, r, c] = 1.0
        frightened = self._power_timer > 0
        for g in self._ghosts:
            planes[3, g["r"], g["c"]] = 0.5 if frightened else 1.0
        return planes.reshape(-1)

    def get_viewer_grid(self) -> list:
        grid = np.zeros((self.ROWS, self.COLS), dtype=np.int32)
        for r, c in self._walls:
            grid[r, c] = 2
        for r, c in self._pellets:
            grid[r, c] = 3
        for r, c in self._power:
            grid[r, c] = 4
        frightened = self._power_timer > 0
        for g in self._ghosts:
            grid[g["r"], g["c"]] = 6 if frightened else 5
        grid[self._player_r, self._player_c] = 1
        return grid.flatten().tolist()

    def _info(self) -> dict:
        return {
            "score": float(self._score),
            "ghosts_eaten": int(self._ghosts_eaten),
            "pellets_left": int(len(self._pellets) + len(self._power)),
            "pellets_eaten": int(self._pellets_eaten),
            "power_timer": int(self._power_timer),
            "player_dir": int(self._player_dir),
        }


def register():
    if "GridPacMan-v0" not in gym.envs.registration.registry:
        gym.register(
            id="GridPacMan-v0",
            entry_point="envs.grid_pacman_env:GridPacManEnv",
        )


register()
