"""AstraTune MLX offline learner ids — facades execv ``python -m astratune.cli <cmd>``.

Training loops live in ``astratune.train_*`` / ``ce_loop``. Use-case plugins supply
hooks via ``--use-case`` (plugin), not train scripts.
"""
from __future__ import annotations

ENTRYPOINTS: dict[str, str] = {
    "mlx_sft": "sft",
    "mlx_dpo": "dpo",
    "mlx_distill": "distill",
    "mlx_rft": "rft",
    "mlx_star": "star",
    "mlx_rlcd": "rlcd",
}

CLI_MODULE = "astratune.cli"


def entrypoint_for(learner_id: str) -> str:
    if learner_id not in ENTRYPOINTS:
        raise KeyError(f"unknown AstraTune learner_id: {learner_id!r}")
    return ENTRYPOINTS[learner_id]


def cli_argv(learner_id: str, python_bin: str) -> list[str]:
    """``[python, -m, astratune.cli, <subcommand>]``."""
    return [str(python_bin), "-m", CLI_MODULE, entrypoint_for(learner_id)]
