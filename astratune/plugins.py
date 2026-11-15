"""Legacy data-source Protocols for AstraTune offline LM post-training.

Prefer :mod:`astra_plugin` for CLI / ``run_*`` contracts (``UseCasePlugin``,
``DpoHooks`` / ``RlcdHooks`` attribute sets). These Protocols remain for
optional data-only adapters that do not implement a full train-hook bag.
"""
from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class DatasetPlugin(Protocol):
    """Supervised (SFT / distill) prompt-completion examples."""

    def load_examples(self) -> list[dict[str, Any]]:
        """Return training rows (use-case schema: prompt/completion, case metadata, ...)."""
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
