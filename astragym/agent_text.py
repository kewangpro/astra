"""AgentGym text + e2e stub gyms for trajectory GRPO.

Uses Astra ``envs.agent_gym`` (shipped with the editable ``astra-rl-gym`` install).
Ensemble no longer vendors ``finetune/agent_gym_env.py``.
"""
from __future__ import annotations

from typing import Any, Optional


class AgentTextGym:
    """Multi-turn text observation gym used by ``mlx_grpo_traj``."""

    def __init__(
        self,
        *,
        gym_id: str = "agent_gym_text",
        tool_failure_prob: float = 0.0,
        max_turns: int = 8,
        e2e_cases: str = "",
        **_: Any,
    ) -> None:
        self.gym_id = gym_id
        self.tool_failure_prob = tool_failure_prob
        self.max_turns = max_turns
        self.e2e_cases = e2e_cases
        self._backend = self._load_backend()

    def _load_backend(self) -> Any:
        if self.gym_id == "e2e_stubs":
            try:
                from astragym.e2e_stubs import DEFAULT_E2E_STUBS, load_e2e_stub_cases

                path = self.e2e_cases
                if path:
                    cases = load_e2e_stub_cases(path)
                else:
                    cases = list(DEFAULT_E2E_STUBS)
                return {"kind": "e2e", "cases": cases}
            except Exception as exc:  # noqa: BLE001
                return {"kind": "e2e_unavailable", "error": str(exc)}
        from envs.agent_gym import AgentToolGym

        return {
            "kind": "astra",
            "env": AgentToolGym(tool_failure_prob=self.tool_failure_prob),
        }

    def reset(self, *, seed: Optional[int] = None) -> tuple[Any, dict]:
        b = self._backend
        if b.get("kind") == "e2e":
            return {"cases": b["cases"]}, {"n_cases": len(b["cases"])}
        if b.get("kind") == "e2e_unavailable":
            raise RuntimeError(f"e2e stubs unavailable: {b.get('error')}")
        env = b["env"]
        if seed is not None and hasattr(env, "reset"):
            try:
                return env.reset(seed=seed)
            except TypeError:
                return env.reset()
        return env.reset()

    def step(self, action: Any) -> tuple[Any, float, bool, bool, dict]:
        b = self._backend
        if b.get("kind") in ("e2e", "e2e_unavailable"):
            raise RuntimeError("e2e_stubs gym scores trajectories via reward_fn, not step()")
        return b["env"].step(action)

    def close(self) -> None:
        b = self._backend
        env = b.get("env")
        if env is not None and hasattr(env, "close"):
            env.close()

    @property
    def observation_space(self) -> Any:
        env = self._backend.get("env")
        return getattr(env, "observation_space", None)

    @property
    def action_space(self) -> Any:
        env = self._backend.get("env")
        return getattr(env, "action_space", None)

    @property
    def raw_env(self) -> Any:
        return self._backend.get("env")


def make_agent_text(gym_id: str = "agent_gym_text", **kwargs: Any) -> AgentTextGym:
    return AgentTextGym(gym_id=gym_id, **kwargs)
