"""
PivotEngine — Step 3.3 strategic pivot logic.

Detects plateaus and proposes hyperparameter adjustments via the Lead Agent.
"""
from __future__ import annotations

from typing import Optional

from backend.logging_config import get_logger

logger = get_logger(__name__)

PLATEAU_WINDOW = 3          # iterations with no improvement → plateau
PLATEAU_THRESHOLD = 0.01    # minimum relative improvement to count as progress

# Escalation: how many consecutive failed pivots before stepping up aggressiveness
ESCALATION_ARCH   = 2  # pivot count → suggest architecture change
ESCALATION_ALGO   = 4  # pivot count → allow algorithm switch
ESCALATION_REWARD = 6  # pivot count → allow reward shaping changes


class PivotEngine:
    """
    Tracks metric history and decides when a strategic pivot is needed.
    """

    def __init__(self, target_metric: dict) -> None:
        self.target_metric = target_metric
        self._metric_name = next(iter(target_metric), "")
        self._target_value = target_metric.get(self._metric_name, 0)
        self._history: list[dict] = []   # [{iteration, metric_name, value}]
        self._pivot_count: int = 0       # consecutive pivots without breakthrough
        self._best_at_last_pivot: Optional[float] = None

    def record(self, iteration: int, metrics: dict) -> None:
        self._history.append({"iteration": iteration, **metrics})

    def record_pivot(self) -> None:
        """Call each time a pivot is applied to track escalation."""
        current_best = self.best_metric_value()
        if (
            self._best_at_last_pivot is not None
            and current_best is not None
            and self._best_at_last_pivot > 0
            and (current_best - self._best_at_last_pivot) / self._best_at_last_pivot
               < PLATEAU_THRESHOLD
        ):
            self._pivot_count += 1
        else:
            self._pivot_count = 0
        self._best_at_last_pivot = current_best

    def escalation_level(self) -> int:
        """0=tweak HPs, 1=change arch, 2=allow algorithm switch, 3=reshape rewards."""
        if self._pivot_count >= ESCALATION_REWARD:
            return 3
        if self._pivot_count >= ESCALATION_ALGO:
            return 2
        if self._pivot_count >= ESCALATION_ARCH:
            return 1
        return 0

    def is_goal_met(self, metrics: dict) -> bool:
        if not self._metric_name:
            return False
        value = self._resolve_metric(self._metric_name, metrics)
        if value is None:
            return False
        return value >= self._target_value

    def _resolve_metric(self, name: str, metrics: dict) -> Optional[float]:
        """Look up metric by name with fallback to suffix-match (e.g. 'accuracy' matches 'validation_accuracy')."""
        if name in metrics:
            return metrics[name]
        for key, val in metrics.items():
            if key.endswith(f"_{name}") or key.startswith(f"{name}_") or key == name:
                return val
        return None

    def needs_pivot(self) -> bool:
        """Return True if the last PLATEAU_WINDOW iterations show no meaningful improvement."""
        values = [
            self._resolve_metric(self._metric_name, h)
            for h in self._history[-PLATEAU_WINDOW:]
            if self._resolve_metric(self._metric_name, h) is not None
        ]
        if len(values) < PLATEAU_WINDOW:
            return False
        if values[0] == 0:
            return True
        relative_improvement = (values[-1] - values[0]) / abs(values[0])
        stalled = relative_improvement < PLATEAU_THRESHOLD
        if stalled:
            logger.info(
                "PivotEngine: plateau detected over %d iterations "
                "(improvement=%.4f < threshold=%.4f)",
                PLATEAU_WINDOW, relative_improvement, PLATEAU_THRESHOLD,
            )
        return stalled

    def best_metric_value(self) -> Optional[float]:
        values = [
            self._resolve_metric(self._metric_name, h)
            for h in self._history
            if self._resolve_metric(self._metric_name, h) is not None
        ]
        return max(values) if values else None

    def history_snapshot(self) -> list[dict]:
        return list(self._history)
