"""Stable-Baselines3 learners bound to an AstraGym world."""
from __future__ import annotations

from typing import Any, Optional

import numpy as np


_ALGO_MAP = {
    "sb3_ppo": "PPO",
    "sb3_dqn": "DQN",
    "sb3_a2c": "A2C",
    "sb3_sac": "SAC",
    "sb3_td3": "TD3",
}


class SB3Learner:
    def __init__(
        self,
        learner_id: str,
        gym: Any,
        *,
        policy: str = "MlpPolicy",
        policy_kwargs: Optional[dict] = None,
        learning_rate: float = 3e-4,
        device: str = "cpu",
        **algo_kwargs: Any,
    ) -> None:
        if learner_id not in _ALGO_MAP:
            raise ValueError(f"not an SB3 learner_id: {learner_id!r}")
        self.learner_id = learner_id
        self.gym = gym
        algo_name = _ALGO_MAP[learner_id]
        import stable_baselines3 as sb3

        Algo = getattr(sb3, algo_name)
        env = getattr(gym, "raw_env", None) or gym
        kw = dict(algo_kwargs)
        kw.setdefault("learning_rate", learning_rate)
        kw.setdefault("verbose", 0)
        kw.setdefault("device", device)
        if policy_kwargs:
            kw["policy_kwargs"] = policy_kwargs
        self.model = Algo(policy, env, **kw)

    def learn(self, total_timesteps: int, **kwargs: Any) -> dict[str, Any]:
        self.model.learn(total_timesteps=int(total_timesteps), **kwargs)
        return {"total_timesteps": int(total_timesteps), "learner_id": self.learner_id}

    def predict(self, observation: Any, deterministic: bool = True) -> Any:
        action, _ = self.model.predict(observation, deterministic=deterministic)
        return action

    def save(self, path: str) -> str:
        self.model.save(path)
        return path

    def load(self, path: str) -> None:
        import stable_baselines3 as sb3

        algo_name = _ALGO_MAP[self.learner_id]
        Algo = getattr(sb3, algo_name)
        self.model = Algo.load(path, device=getattr(self.model, "device", "cpu"))


def make_sb3(learner_id: str, gym: Any, **kwargs: Any) -> SB3Learner:
    return SB3Learner(learner_id, gym, **kwargs)
