"""MLX online-RL learner ids — facades execv ``python -m astrarl.cli <cmd>``.

Training loops live in ``astrarl.train_*``. Ensemble supplies hooks via
``--use-case`` (plugin), not train scripts.
"""
from __future__ import annotations

# learner_id -> CLI subcommand
ENTRYPOINTS = {
    "mlx_grpo_routing": "grpo",
    "mlx_grpo_traj": "agrpo",
    "mlx_opd": "opd",
}

CLI_MODULE = "astrarl.cli"


def entrypoint_for(learner_id: str) -> str:
    if learner_id not in ENTRYPOINTS:
        raise ValueError(f"unknown MLX learner_id: {learner_id!r}")
    return ENTRYPOINTS[learner_id]


def cli_argv(learner_id: str, python_bin: str) -> list[str]:
    """``[python, -m, astrarl.cli, <subcommand>]``."""
    return [str(python_bin), "-m", CLI_MODULE, entrypoint_for(learner_id)]


def gym_id_for_mlx_learner(learner_id: str, *, reward_mode: str = "") -> str:
    if learner_id == "mlx_grpo_traj":
        mode = (reward_mode or "agent_gym").lower()
        if mode in ("e2e", "ensemble_e2e"):
            return "e2e_stubs"
        return "agent_gym_text"
    if learner_id in ("mlx_grpo_routing", "mlx_opd"):
        return "routing_prompt"
    raise ValueError(learner_id)
