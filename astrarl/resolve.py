"""Infer learner_id from recipe / mission fields."""
from __future__ import annotations

from typing import Any, Optional


_SB3 = {
    "ppo": "sb3_ppo",
    "dqn": "sb3_dqn",
    "a2c": "sb3_a2c",
    "sac": "sb3_sac",
    "td3": "sb3_td3",
}


def infer_learner_id(
    *,
    task_type: Optional[str] = None,
    algorithm: Optional[str] = None,
    learner_id: Optional[str] = None,
    hyperparams: Optional[dict[str, Any]] = None,
) -> str:
    hp = hyperparams or {}
    if learner_id:
        return str(learner_id)
    if hp.get("learner_id"):
        return str(hp["learner_id"])

    tt = (task_type or "").lower()
    algo = (algorithm or hp.get("algorithm") or "").lower()

    if tt == "agrpo":
        return "mlx_grpo_traj"
    if tt == "grpo":
        return "mlx_grpo_routing"
    if tt == "opd":
        return "mlx_opd"
    if tt == "rl" or algo in _SB3:
        key = algo or "ppo"
        if key not in _SB3:
            raise ValueError(f"unknown SB3 algorithm: {algo!r}")
        return _SB3[key]

    raise ValueError(
        f"cannot infer learner_id from task_type={task_type!r} algorithm={algorithm!r}"
    )
