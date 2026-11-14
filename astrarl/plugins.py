"""Use-case plugin contracts for AstraRL online LM training.

Ensemble (and future products) implement these; the platform GRPO step loop
calls them without importing product-specific scoring or prompts.
"""
from __future__ import annotations

from typing import Any, Optional, Protocol, runtime_checkable


@runtime_checkable
class RewardPlugin(Protocol):
    """Scores completions and builds prompts for a training case."""

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
