"""AstraRL protocol — learners / checkpoints (no scenario tables)."""
from __future__ import annotations

from typing import Any, Optional, Protocol, runtime_checkable


@runtime_checkable
class Learner(Protocol):
    learner_id: str

    def learn(self, total_timesteps: int, **kwargs: Any) -> dict[str, Any]:
        """Run training; return summary metrics."""
        ...

    def predict(self, observation: Any, deterministic: bool = True) -> Any:
        ...

    def save(self, path: str) -> str:
        ...

    def load(self, path: str) -> None:
        ...
