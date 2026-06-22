"""
PivotEngine — Step 3.3 strategic pivot logic.

Detects plateaus and proposes hyperparameter adjustments via the Lead Agent.
"""
from __future__ import annotations

from typing import Optional

from backend.logging_config import get_logger

logger = get_logger(__name__)

PLATEAU_WINDOW = 3          # iterations with no improvement → plateau
# Don't pivot when the window's best is within this fraction of the all-time peak.
# A brief dip (e.g. iter scores 151→127→126 while peak=164) should self-correct.
PIVOT_COMPETITIVE_THRESHOLD = 0.85
# A pivot resets the escalation counter only when the best metric improves by
# this much relative to its value at the last pivot. Raised to 5% so small
# oscillations in the running average don't keep resetting escalation back to 0.
ESCALATION_RESET_THRESHOLD = 0.05

# Escalation: how many consecutive failed pivots before stepping up aggressiveness
ESCALATION_ARCH   = 2  # pivot count → suggest architecture change
ESCALATION_ALGO   = 4  # pivot count → allow algorithm switch
ESCALATION_REWARD = 6  # pivot count → allow reward shaping changes

# After an arch/algo pivot, if the new config's best is still this much below
# the pre-pivot best after PLATEAU_WINDOW iters, revert to the old checkpoint.
PIVOT_REGRESSION_THRESHOLD = 0.20


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
        # Post-pivot regression tracking (arch/algo pivots only)
        self._pivot_applied: bool = False
        self._pre_pivot_best: Optional[float] = None
        self._post_pivot_best: Optional[float] = None
        self._iters_since_pivot: int = 0
        self._best_policy_kwargs: Optional[dict] = None

    def record(self, iteration: int, metrics: dict, policy_kwargs: Optional[dict] = None) -> None:
        self._history.append({"iteration": iteration, **metrics})
        v = self._resolve_metric(self._metric_name, metrics)
        current_best = self.best_metric_value()
        if v is not None and (current_best is None or v >= current_best):
            if policy_kwargs is not None:
                # Explicit arch at new best — always record it.
                self._best_policy_kwargs = policy_kwargs
            elif self._best_policy_kwargs is None:
                # First best ever seen with no explicit arch: record {} as "default MLP" sentinel
                # so the pivot prompt can tell the LLM to stick with the default rather than
                # cycling through arbitrary net_arch values.
                self._best_policy_kwargs = {}
            # else: keep existing explicit arch — don't overwrite it with {} just because
            # a higher-scoring iter happened to have no policy_kwargs.
        if self._pivot_applied:
            self._iters_since_pivot += 1
            if v is not None and (self._post_pivot_best is None or v > self._post_pivot_best):
                self._post_pivot_best = v

    def record_pivot(self) -> None:
        """Call each time a pivot is applied to track escalation."""
        current_best = self.best_metric_value()
        if (
            self._best_at_last_pivot is not None
            and current_best is not None
            and self._best_at_last_pivot > 0
            and (current_best - self._best_at_last_pivot) / self._best_at_last_pivot
               < ESCALATION_RESET_THRESHOLD
        ):
            self._pivot_count += 1
        else:
            self._pivot_count = 0
        self._best_at_last_pivot = current_best

    @property
    def pivot_count(self) -> int:
        return self._pivot_count

    def restore_pivot_count(self, count: int) -> None:
        """Seed pivot_count from persisted state after a server restart."""
        self._pivot_count = count

    def restore_best_at_last_pivot(self, value: float) -> None:
        """Seed _best_at_last_pivot so record_pivot() doesn't reset escalation on restart."""
        self._best_at_last_pivot = value

    def restore_history(self, entries: list[dict]) -> None:
        """Replay per-iteration goal metric entries so needs_pivot() has full context on restart.

        Each entry must be {iteration: int, <metric_name>: float}.
        Entries are appended in iteration order; duplicates (from the startup seed) are skipped.
        """
        existing_iters = {h.get("iteration") for h in self._history}
        for entry in entries:
            if entry.get("iteration") not in existing_iters:
                self._history.append(entry)
                existing_iters.add(entry.get("iteration"))

    def record_arch_pivot_baseline(self) -> None:
        """Call before applying an arch or algo pivot to arm regression detection."""
        self._pre_pivot_best = self.best_metric_value()
        self._post_pivot_best = None
        self._iters_since_pivot = 0
        self._pivot_applied = True

    def restore_arch_pivot_baseline(self, pre_pivot_best: float) -> None:
        """Re-arm regression detector after a restart using the persisted pre-pivot best."""
        self._pre_pivot_best = pre_pivot_best
        self._post_pivot_best = None
        self._iters_since_pivot = 0
        self._pivot_applied = True
        logger.info(
            "PivotEngine: arch/algo pivot armed — pre_pivot_best=%.3f",
            self._pre_pivot_best if self._pre_pivot_best is not None else float("nan"),
        )

    def should_revert_pivot(self) -> bool:
        """True if the new config is still materially worse after PLATEAU_WINDOW iters.

        Clears the regression window if the new config is performing adequately so that
        future arch changes can be tracked independently.
        """
        if not self._pivot_applied:
            return False
        if self._iters_since_pivot < PLATEAU_WINDOW:
            return False
        pre = self._pre_pivot_best
        post = self._post_pivot_best
        if pre is None or pre <= 0 or post is None:
            # Not enough info — don't revert, just clear tracking
            self._pivot_applied = False
            return False
        regression = (pre - post) / pre
        if regression > PIVOT_REGRESSION_THRESHOLD:
            logger.info(
                "PivotEngine: post-pivot regression detected — pre=%.3f post=%.3f regression=%.1f%% > threshold=%.0f%%",
                pre, post, regression * 100, PIVOT_REGRESSION_THRESHOLD * 100,
            )
            return True
        # New config recovered — clear tracking without reverting
        logger.info(
            "PivotEngine: post-pivot recovery confirmed — pre=%.3f post=%.3f regression=%.1f%%",
            pre, post, regression * 100,
        )
        self._pivot_applied = False
        self._pre_pivot_best = None
        self._post_pivot_best = None
        self._iters_since_pivot = 0
        return False

    def revert_escalation(self) -> None:
        """De-escalate after reverting a bad arch/algo pivot."""
        self._pivot_count = max(0, self._pivot_count - 1)
        self._pivot_applied = False
        self._pre_pivot_best = None
        self._post_pivot_best = None
        self._iters_since_pivot = 0
        logger.info("PivotEngine: escalation reverted — pivot_count=%d", self._pivot_count)

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
        """Return True only if the metric has not improved at all over the last PLATEAU_WINDOW iterations."""
        values = [
            self._resolve_metric(self._metric_name, h)
            for h in self._history[-PLATEAU_WINDOW:]
            if self._resolve_metric(self._metric_name, h) is not None
        ]
        if len(values) < PLATEAU_WINDOW:
            return False
        stalled = values[-1] <= values[0]
        if not stalled:
            return False
        # Guard: if the window's best is a competitive dip BELOW the all-time peak,
        # this is likely temporary variance, not a real plateau. Don't pivot yet.
        # Only applies when window_best < all_time_best (i.e., we're in a dip).
        # Stuck-at-peak (window_best == all_time_best) still triggers a pivot.
        all_time_best = self.best_metric_value()
        if all_time_best is not None and all_time_best > 0:
            window_best = max(values)
            if 0 < window_best < all_time_best and window_best >= all_time_best * PIVOT_COMPETITIVE_THRESHOLD:
                logger.info(
                    "PivotEngine: dip detected but window_best=%.4f is within %.0f%% of "
                    "all-time best=%.4f — suppressing pivot",
                    window_best, PIVOT_COMPETITIVE_THRESHOLD * 100, all_time_best,
                )
                return False
        logger.info(
            "PivotEngine: plateau detected over %d iterations "
            "(latest=%.4f <= earliest=%.4f, no improvement)",
            PLATEAU_WINDOW, values[-1], values[0],
        )
        return True

    def best_metric_value(self) -> Optional[float]:
        best_entry = self._best_entry()
        return best_entry[1] if best_entry else None

    def best_metric_iteration(self) -> Optional[int]:
        """Return the iteration index at which the best metric was recorded."""
        best_entry = self._best_entry()
        if best_entry is None:
            return None
        iteration = best_entry[0]
        return None if iteration == -1 else iteration  # -1 is the seed entry

    def best_policy_kwargs(self) -> Optional[dict]:
        """Return the policy_kwargs (net_arch etc.) that produced the best metric value."""
        return self._best_policy_kwargs

    def restore_best_policy_kwargs(self, kwargs: Optional[dict]) -> None:
        """Seed _best_policy_kwargs from persisted state after a server restart."""
        self._best_policy_kwargs = kwargs

    def _best_entry(self) -> Optional[tuple]:
        """Return (iteration, value) for the history entry with the highest metric."""
        best_val: Optional[float] = None
        best_iter: Optional[int] = None
        for h in self._history:
            v = self._resolve_metric(self._metric_name, h)
            if v is not None and (best_val is None or v > best_val):
                best_val = v
                best_iter = h.get("iteration")
        return (best_iter, best_val) if best_val is not None else None

    def history_snapshot(self) -> list[dict]:
        return list(self._history)
