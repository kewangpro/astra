"""AstraGym protocol — worlds, rewards, eval (no optimizers)."""
from __future__ import annotations

from typing import Any, Optional, Protocol, runtime_checkable


@runtime_checkable
class Gym(Protocol):
    """Minimal interactive world used by AstraRL learners and play HUD helpers."""

    gym_id: str

    def reset(self, *, seed: Optional[int] = None) -> tuple[Any, dict]:
        """Return (observation, info)."""
        ...

    def step(self, action: Any) -> tuple[Any, float, bool, bool, dict]:
        """Return (obs, reward, terminated, truncated, info)."""
        ...

    def close(self) -> None:
        ...

    @property
    def observation_space(self) -> Any:
        ...

    @property
    def action_space(self) -> Any:
        ...
