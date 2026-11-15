"""AstraRL — learners for interactive / gym-coupled RL (incl. off-policy SB3).

Training center entrypoint: ``python -m astrarl.cli``. Also owns rollout backends,
DAPO-lite helpers, and GRPO math loaded by use-case plugins.
"""
from __future__ import annotations

from typing import Any, Optional

from astrarl.mlx_learners import ENTRYPOINTS, entrypoint_for, gym_id_for_mlx_learner
from astrarl.mlx_facade import MLXLearnerFacade, make_mlx_facade
from astrarl.protocol import Learner
from astrarl.resolve import infer_learner_id

__all__ = [
    "Learner",
    "MLXLearnerFacade",
    "make",
    "infer_learner_id",
    "entrypoint_for",
    "gym_id_for_mlx_learner",
    "ENTRYPOINTS",
]


def make(learner_id: str, gym: Any = None, **kwargs: Any) -> Any:
    """Construct an AstraRL learner (SB3 object or MLXLearnerFacade)."""
    lid = str(learner_id)
    if lid.startswith("sb3_"):
        if gym is None:
            raise ValueError("SB3 learners require an AstraGym instance")
        from astrarl.sb3_learner import make_sb3

        return make_sb3(lid, gym, **kwargs)
    if lid in ENTRYPOINTS:
        return make_mlx_facade(lid, gym, **kwargs)
    raise ValueError(f"unknown learner_id: {learner_id!r}")
