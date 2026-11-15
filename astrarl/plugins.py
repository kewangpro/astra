"""Legacy step-loop Protocols for AstraRL online LM training.

Prefer :mod:`astra_plugin` for CLI / ``run_*`` contracts (``UseCasePlugin``,
per-learner hook attribute sets). ``RewardPlugin`` / ``CaseSource`` remain for
lower-level GRPO step helpers that take method-based scorers.
"""
from __future__ import annotations

from typing import Any, Optional, Protocol, runtime_checkable


@runtime_checkable
class RewardPlugin(Protocol):
    """Scores completions and builds prompts for a training case (step-loop)."""

    def build_prompt(self, case: dict) -> str:
        """Full model-ready prompt string for ``case``."""
        ...

    def score(self, completion: str, case: dict) -> float:
        """Scalar reward in a use-case-defined range (routing: often [-0.5, 1.0])."""
        ...


@runtime_checkable
class CaseSource(Protocol):
    """Supplies the next training case (pool / curriculum owned by the use case)."""

    def sample(self) -> dict:
        ...
