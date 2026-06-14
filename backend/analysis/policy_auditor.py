"""
PolicyAuditor — Step 3.5.

Computes action-distribution histograms for RL policies to detect
mode collapse (all actions converging to one) or action bias.
"""
from __future__ import annotations

from typing import Optional

from backend.logging_config import get_logger

logger = get_logger(__name__)


class PolicyAuditor:
    """
    Samples a policy over a set of observations and builds an action histogram.
    """

    def __init__(self, checkpoint_path: str, n_actions: int) -> None:
        self.checkpoint_path = checkpoint_path
        self.n_actions = n_actions
        self._model = None

    def _load_model(self):
        try:
            import torch
            self._model = torch.load(self.checkpoint_path, map_location="cpu")
            self._model.eval()
        except Exception as e:
            logger.error("PolicyAuditor: failed to load model: %s", e)
            raise

    def compute_histogram(self, observations: list, n_samples: Optional[int] = None) -> dict:
        """
        Run the policy over observations and return an action frequency histogram.
        Detects mode collapse if any single action exceeds 80% frequency.
        """
        if self._model is None:
            self._load_model()

        try:
            import torch
            counts = [0] * self.n_actions
            obs_batch = observations[:n_samples] if n_samples else observations

            with torch.no_grad():
                for obs in obs_batch:
                    tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
                    logits = self._model(tensor)
                    action = logits.argmax(dim=-1).item()
                    if 0 <= action < self.n_actions:
                        counts[action] += 1

            total = sum(counts)
            frequencies = [c / total if total > 0 else 0.0 for c in counts]
            max_freq = max(frequencies) if frequencies else 0.0
            mode_collapsed = max_freq > 0.8

            if mode_collapsed:
                logger.warning(
                    "PolicyAuditor: mode collapse detected — dominant action frequency=%.2f",
                    max_freq,
                )

            return {
                "n_actions": self.n_actions,
                "counts": counts,
                "frequencies": frequencies,
                "mode_collapsed": mode_collapsed,
                "dominant_action": frequencies.index(max_freq),
                "dominant_frequency": max_freq,
                "entropy": self._entropy(frequencies),
            }

        except Exception as e:
            logger.error("PolicyAuditor: histogram computation failed: %s", e)
            return {"error": str(e)}

    @staticmethod
    def _entropy(frequencies: list[float]) -> float:
        import math
        return -sum(p * math.log2(p) for p in frequencies if p > 0)
