"""Infer gym_id from recipe / mission fields (backward compatible)."""
from __future__ import annotations

from typing import Any, Optional


def infer_gym_id(
    *,
    task_type: Optional[str] = None,
    env_id: Optional[str] = None,
    reward_mode: Optional[str] = None,
    gym_id: Optional[str] = None,
    hyperparams: Optional[dict[str, Any]] = None,
) -> str:
    """Resolve AstraGym id. Explicit ``gym_id`` always wins."""
    hp = hyperparams or {}
    if gym_id:
        return str(gym_id)
    if hp.get("gym_id"):
        return str(hp["gym_id"])

    tt = (task_type or "").lower()
    mode = (reward_mode or hp.get("reward_mode") or "").lower()
    eid = env_id or hp.get("env_id")

    if tt == "agrpo" or mode in ("agent_gym", "e2e", "ensemble_e2e"):
        if mode in ("e2e", "ensemble_e2e"):
            return "e2e_stubs"
        return "agent_gym_text"

    if tt in ("grpo", "opd"):
        return "routing_prompt"

    if tt == "rl" or eid:
        if eid in (
            "MultiTurnAgentGym-v0",
            "MultiTurnAgentGymText-v0",
            "AgentGym-v0",
        ) or (eid and "agentgym" in eid.lower().replace("-", "").replace("_", "")):
            if eid and "text" in eid.lower():
                return "agent_gym_text"
            return "agent_gym_vector"
        if eid:
            return f"gymnasium:{eid}"

    raise ValueError(
        f"cannot infer gym_id from task_type={task_type!r} env_id={env_id!r} "
        f"reward_mode={reward_mode!r}"
    )


def gymnasium_env_id(gym_id: str) -> Optional[str]:
    """Strip ``gymnasium:`` prefix; else None."""
    if gym_id.startswith("gymnasium:"):
        return gym_id.split(":", 1)[1]
    return None
