"""AgentGym vector observation world (SB3 / play)."""
from __future__ import annotations

from typing import Any, Optional

import gymnasium as gym

from astragym.gymnasium_gym import GymnasiumGym


def make_agent_gym_vector(**kwargs: Any) -> GymnasiumGym:
    """MultiTurnAgentGym-v0 via the shared gymnasium adapter."""
    env_kwargs = {k: v for k, v in kwargs.items() if k != "gym_id"}
    env_kwargs.setdefault("obs_mode", "vector")
    g = GymnasiumGym("MultiTurnAgentGym-v0", **env_kwargs)
    g.gym_id = "agent_gym_vector"
    return g


def make_agent_gym_text_env(**kwargs: Any) -> gym.Env:
    """Raw Gymnasium text-obs env (for callers that need gym.Env directly)."""
    from envs.register import register_for_env_id

    register_for_env_id("MultiTurnAgentGymText-v0")
    kw = {k: v for k, v in kwargs.items() if k != "gym_id"}
    kw.setdefault("obs_mode", "text")
    try:
        return gym.make("MultiTurnAgentGymText-v0", **kw)
    except Exception:
        register_for_env_id("MultiTurnAgentGym-v0")
        return gym.make("MultiTurnAgentGym-v0", obs_mode="text", **{
            k: v for k, v in kw.items() if k != "obs_mode"
        })
