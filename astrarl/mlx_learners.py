"""MLX online-RL learner ids — CLI entrypoints live in Ensemble finetune/.

These symbols document the contract and provide resolve helpers. Heavy training
remains in ``grpo_train.py`` / ``agent_grpo_train.py`` / ``opd_train.py`` so
Mini SSH preflight script names stay stable.
"""
from __future__ import annotations

ENTRYPOINTS = {
    "mlx_grpo_routing": "grpo_train.py",
    "mlx_grpo_traj": "agent_grpo_train.py",
    "mlx_opd": "opd_train.py",
}


def entrypoint_for(learner_id: str) -> str:
    if learner_id not in ENTRYPOINTS:
        raise ValueError(f"unknown MLX learner_id: {learner_id!r}")
    return ENTRYPOINTS[learner_id]


def gym_id_for_mlx_learner(learner_id: str, *, reward_mode: str = "") -> str:
    if learner_id == "mlx_grpo_traj":
        mode = (reward_mode or "agent_gym").lower()
        if mode in ("e2e", "ensemble_e2e"):
            return "e2e_stubs"
        return "agent_gym_text"
    if learner_id in ("mlx_grpo_routing", "mlx_opd"):
        return "routing_prompt"
    raise ValueError(learner_id)
