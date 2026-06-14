"""
StressTester — Step 3.4.

Introduces noise and edge cases to verify the model's robustness
beyond the Golden Set. Domain-specific noise strategies are registered
per task type.
"""
from __future__ import annotations

import random
from typing import Callable, Optional

from backend.logging_config import get_logger

logger = get_logger(__name__)


# ── Noise injection strategies ────────────────────────────────────────────────

def _rl_noise(checkpoint_path: str, seed: int) -> dict:
    """Inject observation noise and random action perturbations."""
    random.seed(seed)
    logger.info("StressTester: RL noise injection (seed=%d) on %s", seed, checkpoint_path)
    return {"noisy_mean_reward": 0.0, "robustness_score": 0.0}  # stub for Phase 6


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


class StressTester:
    def __init__(self, task_type: str, num_seeds: int = 3) -> None:
        self.task_type = task_type
        self.num_seeds = num_seeds
        self._strategy = _NOISE_STRATEGIES.get(task_type)
        if not self._strategy:
            logger.warning("StressTester: no noise strategy for task_type '%s'", task_type)

    def run(self, checkpoint_path: str) -> dict:
        """
        Run stress tests across multiple random seeds.
        Returns aggregate robustness metrics.
        """
        if not self._strategy:
            return {"status": "skipped", "reason": f"no strategy for {self.task_type}"}

        results = []
        for seed in range(self.num_seeds):
            result = self._strategy(checkpoint_path, seed=seed)
            results.append({"seed": seed, **result})

        return {"task_type": self.task_type, "seeds_tested": self.num_seeds, "results": results}
