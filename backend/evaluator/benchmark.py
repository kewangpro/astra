"""
BenchmarkSuite — Step 3.4.

Runs the trained model against a fixed "Golden Set" of domain challenges.
The Golden Set is defined per domain (snake, tetris, nlp) and is immutable
across runs to ensure comparable results.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

from backend.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class GoldenChallenge:
    name: str
    domain: str
    description: str
    evaluate_fn: Callable[[str], dict]  # takes checkpoint_path, returns {metric: value}
    pass_threshold: dict = field(default_factory=dict)


# ── Built-in Golden Sets ───────────────────────────────────────────────────────

def _snake_eval(checkpoint_path: str) -> dict:
    """Evaluate a Snake RL model. Stub: Phase 6 hardens with real env rollouts."""
    logger.info("BenchmarkSuite: running Snake Golden Set on %s", checkpoint_path)
    return {"mean_reward": 0.0, "max_length": 0}   # replaced by real eval in Phase 6


def _tetris_eval(checkpoint_path: str) -> dict:
    logger.info("BenchmarkSuite: running Tetris Golden Set on %s", checkpoint_path)
    return {"mean_score": 0.0, "lines_cleared": 0}


def _nlp_eval(checkpoint_path: str) -> dict:
    logger.info("BenchmarkSuite: running NLP Golden Set on %s", checkpoint_path)
    return {"eval_loss": 999.0, "perplexity": 999.0}


GOLDEN_SETS: dict[str, list[GoldenChallenge]] = {
    "snake": [
        GoldenChallenge(
            name="snake_baseline",
            domain="snake",
            description="Achieve mean_reward ≥ 20 on 16×12 grid",
            evaluate_fn=_snake_eval,
            pass_threshold={"mean_reward": 20},
        ),
    ],
    "tetris": [
        GoldenChallenge(
            name="tetris_baseline",
            domain="tetris",
            description="Clear ≥ 10 lines on standard board",
            evaluate_fn=_tetris_eval,
            pass_threshold={"lines_cleared": 10},
        ),
    ],
    "nlp": [
        GoldenChallenge(
            name="nlp_loss",
            domain="nlp",
            description="Achieve eval_loss ≤ 1.5 on validation set",
            evaluate_fn=_nlp_eval,
            pass_threshold={"eval_loss": 1.5},
        ),
    ],
}


class BenchmarkSuite:
    def __init__(self, domain: str) -> None:
        self.domain = domain
        self.challenges = GOLDEN_SETS.get(domain, [])
        if not self.challenges:
            logger.warning("BenchmarkSuite: no Golden Set defined for domain '%s'", domain)

    def run(self, checkpoint_path: str) -> dict:
        """
        Run all Golden Challenges for this domain.
        Returns {"passed": int, "failed": int, "results": [...]}.
        """
        passed = 0
        failed = 0
        results = []

        for challenge in self.challenges:
            metrics = challenge.evaluate_fn(checkpoint_path)
            challenge_passed = all(
                metrics.get(k, 0) >= v
                for k, v in challenge.pass_threshold.items()
            )
            status = "passed" if challenge_passed else "failed"
            if challenge_passed:
                passed += 1
            else:
                failed += 1
            results.append({
                "name": challenge.name,
                "status": status,
                "metrics": metrics,
                "threshold": challenge.pass_threshold,
            })
            logger.info("BenchmarkSuite: %s → %s (%s)", challenge.name, status, metrics)

        return {"passed": passed, "failed": failed, "results": results}
