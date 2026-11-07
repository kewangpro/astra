"""Gymnasium-backed AstraGym (Snake, MinAtar, …)."""
from __future__ import annotations

from typing import Any, Optional

import gymnasium as gym

from astragym.resolve import gymnasium_env_id


class GymnasiumGym:
    """Thin wrapper so learners see a stable ``gym_id`` + Gymnasium API."""

    def __init__(self, env_id: str, **env_kwargs: Any) -> None:
        try:
            from envs.register import register_for_env_id

            register_for_env_id(env_id)
        except Exception:
            pass
        self.gym_id = f"gymnasium:{env_id}"
        self.env_id = env_id
        self._env = gym.make(env_id, **env_kwargs)

    def reset(self, *, seed: Optional[int] = None) -> tuple[Any, dict]:
        return self._env.reset(seed=seed)

    def step(self, action: Any) -> tuple[Any, float, bool, bool, dict]:
        return self._env.step(action)

    def close(self) -> None:
        self._env.close()

    @property
    def observation_space(self) -> Any:
        return self._env.observation_space

    @property
    def action_space(self) -> Any:
        return self._env.action_space

    @property
    def raw_env(self) -> gym.Env:
        """Underlying Gymnasium env (SB3 / play)."""
        return self._env


def make_gymnasium(gym_id: str, **kwargs: Any) -> GymnasiumGym:
    eid = gymnasium_env_id(gym_id) or gym_id
    return GymnasiumGym(eid, **kwargs)
