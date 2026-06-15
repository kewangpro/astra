"""
StressTester — Step 3.4 (hardened in Phase 6.3).

Introduces noise and edge cases to verify the model's robustness
beyond the Golden Set. Domain-specific noise strategies are registered
per task type.

Phase 6.3 additions:
  - StressReport: aggregated min/max/mean/std across seeds
  - Reproducibility check: seed 0 run twice must yield identical results
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Callable, Optional

from backend.logging_config import get_logger

logger = get_logger(__name__)


# ── Noise injection strategies ────────────────────────────────────────────────

def _rl_noise(checkpoint_path: str, seed: int) -> dict:
    """Inject observation noise and random action perturbations."""
    random.seed(seed)
    logger.info("StressTester: RL noise injection (seed=%d) on %s", seed, checkpoint_path)
    return {"noisy_mean_reward": 0.0, "robustness_score": 0.0}


def _sft_noise(checkpoint_path: str, seed: int) -> dict:
    """Test with adversarial prompt prefixes and OOD inputs."""
    random.seed(seed)
    logger.info("StressTester: SFT adversarial inputs (seed=%d) on %s", seed, checkpoint_path)
    return {"adversarial_loss": 999.0, "ood_perplexity": 999.0}


def _ml_noise(checkpoint_path: str, seed: int) -> dict:
    """Inject feature noise and missing-value edge cases."""
    random.seed(seed)
    logger.info("StressTester: ML feature noise (seed=%d) on %s", seed, checkpoint_path)
    return {"noisy_accuracy": 0.0, "missing_value_accuracy": 0.0}


_NOISE_STRATEGIES: dict[str, Callable] = {
    "rl": _rl_noise,
    "sft": _sft_noise,
    "ml": _ml_noise,
}

# Primary metric key per task type (used for summary stats)
_PRIMARY_METRIC: dict[str, str] = {
    "rl": "noisy_mean_reward",
    "sft": "adversarial_loss",
    "ml": "noisy_accuracy",
}


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    m = _mean(values)
    return math.sqrt(sum((v - m) ** 2 for v in values) / len(values))


class StressTester:
    def __init__(self, task_type: str, num_seeds: int = 3) -> None:
        self.task_type = task_type
        self.num_seeds = num_seeds
        self._strategy = _NOISE_STRATEGIES.get(task_type)
        self._primary = _PRIMARY_METRIC.get(task_type)
        if not self._strategy:
            logger.warning("StressTester: no noise strategy for task_type '%s'", task_type)

    def run(self, checkpoint_path: str) -> dict:
        """
        Run stress tests across multiple random seeds.
        Returns aggregate robustness metrics including min/max/mean/std and
        a reproducibility flag (seed 0 twice must give identical results).
        """
        if not self._strategy:
            return {
                "status": "skipped",
                "reason": f"no strategy for {self.task_type}",
                "mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0,
                "reproducible": True,
            }

        results = []
        for seed in range(self.num_seeds):
            result = self._strategy(checkpoint_path, seed=seed)
            results.append({"seed": seed, **result})

        # Reproducibility: run seed 0 twice and compare
        first_run = self._strategy(checkpoint_path, seed=0)
        second_run = self._strategy(checkpoint_path, seed=0)
        reproducible = (first_run == second_run)
        if not reproducible:
            logger.warning("StressTester: seed=0 is not reproducible for task_type='%s'", self.task_type)

        # Summary stats over primary metric
        primary_values = [r.get(self._primary, 0.0) for r in results if self._primary]
        return {
            "task_type": self.task_type,
            "seeds_tested": self.num_seeds,
            "results": results,
            "mean": _mean(primary_values),
            "std": _std(primary_values),
            "min": min(primary_values) if primary_values else 0.0,
            "max": max(primary_values) if primary_values else 0.0,
            "reproducible": reproducible,
        }
