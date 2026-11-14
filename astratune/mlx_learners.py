"""AstraTune MLX offline learner entrypoints (Ensemble thin ``*_train.py``).

Training loops live in ``astratune.train_*`` / ``ce_loop``; Ensemble CLIs stay
the Mini ``os.execv`` targets and inject use-case hooks.
"""
from __future__ import annotations

ENTRYPOINTS: dict[str, str] = {
    "mlx_sft": "sft_train.py",
    "mlx_dpo": "dpo_train.py",
    "mlx_distill": "distill_train.py",
    "mlx_rft": "rft_train.py",
    "mlx_star": "star_train.py",
    "mlx_rlcd": "rlcd_train.py",
}


def entrypoint_for(learner_id: str) -> str:
    if learner_id not in ENTRYPOINTS:
        raise KeyError(f"unknown AstraTune learner_id: {learner_id!r}")
    return ENTRYPOINTS[learner_id]
