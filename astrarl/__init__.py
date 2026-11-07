"""AstraRL — learners for online RL (Epoch 9)."""
from __future__ import annotations

from typing import Any, Optional

from astrarl.mlx_learners import ENTRYPOINTS, entrypoint_for, gym_id_for_mlx_learner
from astrarl.protocol import Learner
from astrarl.resolve import infer_learner_id

__all__ = [
    "Learner",
    "make",
    "infer_learner_id",
    "entrypoint_for",
    "gym_id_for_mlx_learner",
    "ENTRYPOINTS",
]


def make(learner_id: str, gym: Any = None, **kwargs: Any) -> Any:
    """Construct an AstraRL learner. MLX ids return a descriptor dict (CLI shim)."""
    lid = str(learner_id)
    if lid.startswith("sb3_"):
        if gym is None:
            raise ValueError("SB3 learners require an AstraGym instance")
        from astrarl.sb3_learner import make_sb3

        return make_sb3(lid, gym, **kwargs)
    if lid in ENTRYPOINTS:
        return {
            "learner_id": lid,
            "entrypoint": ENTRYPOINTS[lid],
            "gym_id": gym_id_for_mlx_learner(lid, reward_mode=kwargs.get("reward_mode", "")),
            "gym": gym,
            "kwargs": kwargs,
        }
    raise ValueError(f"unknown learner_id: {learner_id!r}")
