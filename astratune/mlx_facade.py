"""MLX offline learner facade — AstraTune wiring over the shared astrarl core.

Dataset / preference workers (SFT, DPO, distill, RFT, STAR, RLCD). Shares
``astrarl.mlx_facade.MLXLearnerFacade``; this module only supplies AstraTune
ENTRYPOINTS, flag aliases, and DPO ``epochs`` timestep mapping.
"""
from __future__ import annotations

from typing import Any

from astrarl.mlx_facade import (
    FacadeConfig,
    MLXLearnerFacade,
    kwargs_to_argv as _shared_kwargs_to_argv,
)
from astratune.mlx_learners import ENTRYPOINTS, entrypoint_for

_FACADE_ONLY = frozenset({
    "finetune_dir",
    "python_bin",
    "extra_argv",
    "use_execv",
    "env",
    "cwd",
    "check",
    "gym_id",
    "learner_id",
    "task_type",
    "algorithm",
    "gym",
})

# argparse names that are not simple ``--{key.replace('_','-')}``.
_FLAG_ALIASES: dict[str, str] = {
    "k_collect": "--K-collect",
    "K_collect": "--K-collect",
}


def _astratune_map_timesteps(learner_id: str, merged: dict[str, Any], total_timesteps: int) -> None:
    if "iters" not in merged and learner_id != "mlx_dpo":
        merged["iters"] = int(total_timesteps)
    if "epochs" not in merged and learner_id == "mlx_dpo":
        merged["epochs"] = int(total_timesteps)


ASTRATUNE_FACADE_CONFIG = FacadeConfig(
    entrypoints=ENTRYPOINTS,
    entrypoint_for=entrypoint_for,
    facade_only=_FACADE_ONLY,
    flag_aliases=_FLAG_ALIASES,
    resolve_gym_id=None,
    map_total_timesteps=_astratune_map_timesteps,
    unknown_learner_fmt="not an AstraTune MLX learner_id: {learner_id!r}",
    predict_msg_fmt=(
        "{learner_id}: inference is served by Ensemble/JEV MLX endpoints, "
        "not astratune.MLXLearnerFacade.predict()"
    ),
)


def kwargs_to_argv(kwargs: dict[str, Any]) -> list[str]:
    """Map AstraTune facade/recipe kwargs to Ensemble CLI argv."""
    return _shared_kwargs_to_argv(
        kwargs,
        facade_only=_FACADE_ONLY,
        flag_aliases=_FLAG_ALIASES,
    )


def make_mlx_facade(learner_id: str, gym: Any = None, **kwargs: Any) -> MLXLearnerFacade:
    return MLXLearnerFacade(
        learner_id, gym, config=ASTRATUNE_FACADE_CONFIG, **kwargs,
    )


# Re-export class name for ``from astratune.mlx_facade import MLXLearnerFacade``.
__all__ = [
    "ASTRATUNE_FACADE_CONFIG",
    "MLXLearnerFacade",
    "kwargs_to_argv",
    "make_mlx_facade",
]
