"""AstraTune — offline / dataset LM post-training platform.

Sibling to AstraGym (worlds) and AstraRL (online / gym-coupled learners).
Training center entrypoint: ``python -m astratune.cli``. Owns shared MLX LoRA I/O
and SFT/DPO/distill/RLCD math. Use-case plugins supply hooks via ``astra_plugin`` — products are not a second train host.
"""
from __future__ import annotations

from typing import Any

from astratune.mlx_learners import ENTRYPOINTS, entrypoint_for
from astratune.mlx_facade import MLXLearnerFacade, make_mlx_facade

__all__ = [
    "make",
    "ENTRYPOINTS",
    "entrypoint_for",
    "MLXLearnerFacade",
]


def make(learner_id: str, gym: Any = None, **kwargs: Any) -> Any:
    """Construct an AstraTune learner facade (MLX offline workers)."""
    lid = str(learner_id)
    if lid in ENTRYPOINTS:
        return make_mlx_facade(lid, gym, **kwargs)
    raise ValueError(f"unknown astratune learner_id: {learner_id!r}")
