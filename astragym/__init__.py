"""AstraGym — interactive worlds for online RL (Epoch 9)."""
from __future__ import annotations

from typing import Any

from astragym.protocol import Gym
from astragym.resolve import gymnasium_env_id, infer_gym_id

__all__ = [
    "Gym",
    "make",
    "infer_gym_id",
    "gymnasium_env_id",
]


def make(gym_id: str, **kwargs: Any) -> Any:
    """Construct an AstraGym world by id."""
    gid = str(gym_id)
    if gid.startswith("gymnasium:"):
        from astragym.gymnasium_gym import make_gymnasium

        return make_gymnasium(gid, **kwargs)
    if gid == "agent_gym_vector":
        from astragym.agent_vector import make_agent_gym_vector

        return make_agent_gym_vector(**kwargs)
    if gid in ("agent_gym_text", "e2e_stubs"):
        from astragym.agent_text import make_agent_text

        return make_agent_text(gym_id=gid, **kwargs)
    if gid == "routing_prompt":
        from astragym.routing_prompt import make_routing_prompt

        return make_routing_prompt(**kwargs)
    # Bare env id convenience
    if ":" not in gid and gid.endswith("-v0"):
        from astragym.gymnasium_gym import make_gymnasium

        return make_gymnasium(f"gymnasium:{gid}", **kwargs)
    raise ValueError(f"unknown gym_id: {gym_id!r}")
