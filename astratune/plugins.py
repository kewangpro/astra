"""Use-case plugin contracts for AstraTune offline LM post-training.

Ensemble (and future products) implement these; ``python -m astratune.cli``
loads hooks via ``--use-case`` without duplicating platform math.
"""
from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class DatasetPlugin(Protocol):
    """Supervised (SFT / distill) prompt–completion examples."""

    def load_examples(self) -> list[dict[str, Any]]:
        """Return training rows (use-case schema: prompt/completion, case metadata, …)."""
        ...


@runtime_checkable
class PreferencePlugin(Protocol):
    """DPO-style (prompt, chosen, rejected) preference pairs."""

    def load_pairs(self) -> list[dict[str, Any]]:
        """Return preference pairs with prompt and chosen/rejected completions."""
        ...


@runtime_checkable
class DecisionCasesPlugin(Protocol):
    """RLCD decision JSONL (Noul / Choice / Score)."""

    def load_cases(self) -> list[dict[str, Any]]:
        """Return generic decision cases for calibrated readout training."""
        ...