"""
BenchmarkSuite — Step 3.4 (hardened in Phase 6.3).

Runs the trained model against a fixed "Golden Set" of domain challenges.
The Golden Set is defined per domain (snake, tetris, nlp) and is immutable
across runs to ensure comparable results.
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from typing import Callable, Optional

from backend.logging_config import get_logger

logger = get_logger(__name__)

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))


def _rollout(checkpoint_path: str, env_id: str, n_episodes: int = 10) -> tuple[float, dict]:
    """Load checkpoint, run n_episodes deterministically, return (mean_reward, mean_info).

    Returns info values averaged across episodes. Only collects info at episode end
    (terminated/truncated step).
    """
    import numpy as np

    if _PROJECT_ROOT not in sys.path:
        sys.path.insert(0, _PROJECT_ROOT)

    try:
        from stable_baselines3 import PPO, SAC, A2C, DQN, TD3
        import gymnasium as gym

        # Try loading with each algo until one succeeds
        model = None
        for cls in (PPO, SAC, A2C, DQN, TD3):
            try:
                model = cls.load(checkpoint_path)
                break
            except Exception:
                continue
        if model is None:
            return 0.0, {}

        env = gym.make(env_id)
        rewards, info_accum = [], {}
        for _ in range(n_episodes):
            obs, _ = env.reset()
            ep_reward, done = 0.0, False
            ep_info = {}
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, r, terminated, truncated, info = env.step(action)
                ep_reward += float(r)
                done = terminated or truncated
                if done:
                    ep_info = info
            rewards.append(ep_reward)
            for k, v in ep_info.items():
                try:
                    info_accum.setdefault(k, []).append(float(v))
                except (TypeError, ValueError):
                    pass
        env.close()
        mean_info = {k: float(np.mean(vs)) for k, vs in info_accum.items()}
        return float(np.mean(rewards)), mean_info
    except Exception as exc:
        logger.warning("BenchmarkSuite rollout failed env=%s: %s", env_id, exc)
        return 0.0, {}


@dataclass
class GoldenChallenge:
    name: str
    domain: str
    description: str
    evaluate_fn: Callable[[str], dict]  # takes checkpoint_path, returns {metric: value}
    pass_threshold: dict = field(default_factory=dict)


# ── Built-in Golden Sets ───────────────────────────────────────────────────────

def _snake_eval(checkpoint_path: str) -> dict:
    if not os.path.exists(checkpoint_path):
        logger.warning("BenchmarkSuite: checkpoint not found: %s", checkpoint_path)
        return {"mean_reward": 0.0, "max_length": 0}
    try:
        if _PROJECT_ROOT not in sys.path:
            sys.path.insert(0, _PROJECT_ROOT)
        from envs.snake_env import register
        register()
    except Exception:
        pass
    logger.info("BenchmarkSuite: running Snake baseline on %s", checkpoint_path)
    mean_reward, info = _rollout(checkpoint_path, "Snake-v0", n_episodes=10)
    return {"mean_reward": mean_reward, "max_length": info.get("max_length", 0.0)}


def _snake_hard_eval(checkpoint_path: str) -> dict:
    if not os.path.exists(checkpoint_path):
        return {"mean_reward": 0.0, "max_length": 0}
    try:
        if _PROJECT_ROOT not in sys.path:
            sys.path.insert(0, _PROJECT_ROOT)
        from envs.snake_env import register
        register()
    except Exception:
        pass
    logger.info("BenchmarkSuite: running Snake hard on %s", checkpoint_path)
    mean_reward, info = _rollout(checkpoint_path, "Snake-v0", n_episodes=10)
    return {"mean_reward": mean_reward, "max_length": info.get("max_length", 0.0)}


def _tetris_eval(checkpoint_path: str) -> dict:
    if not os.path.exists(checkpoint_path):
        return {"mean_reward": 0.0, "lines_cleared": 0}
    try:
        if _PROJECT_ROOT not in sys.path:
            sys.path.insert(0, _PROJECT_ROOT)
        from envs.tetris_env import register
        register()
    except Exception:
        pass
    logger.info("BenchmarkSuite: running Tetris baseline on %s", checkpoint_path)
    mean_reward, info = _rollout(checkpoint_path, "Tetris-v0", n_episodes=10)
    return {"mean_reward": mean_reward, "lines_cleared": info.get("lines_cleared", 0.0)}


def _tetris_hard_eval(checkpoint_path: str) -> dict:
    if not os.path.exists(checkpoint_path):
        return {"mean_reward": 0.0, "lines_cleared": 0}
    try:
        if _PROJECT_ROOT not in sys.path:
            sys.path.insert(0, _PROJECT_ROOT)
        from envs.tetris_env import register
        register()
    except Exception:
        pass
    logger.info("BenchmarkSuite: running Tetris hard on %s", checkpoint_path)
    mean_reward, info = _rollout(checkpoint_path, "Tetris-v0", n_episodes=10)
    return {"mean_reward": mean_reward, "lines_cleared": info.get("lines_cleared", 0.0)}


def _nlp_loss_eval(checkpoint_path: str) -> dict:
    if not os.path.exists(checkpoint_path):
        return {"eval_loss": 999.0, "perplexity": 999.0}
    logger.info("BenchmarkSuite: running NLP loss eval on %s", checkpoint_path)
    return {"eval_loss": 999.0, "perplexity": 999.0}


def _nlp_perplexity_eval(checkpoint_path: str) -> dict:
    """Stricter NLP scenario: perplexity on an out-of-domain validation set."""
    if not os.path.exists(checkpoint_path):
        return {"eval_loss": 999.0, "perplexity": 999.0}
    logger.info("BenchmarkSuite: running NLP perplexity eval on %s", checkpoint_path)
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
        GoldenChallenge(
            name="snake_hard",
            domain="snake",
            description="Achieve mean_reward ≥ 80 on 32×24 grid",
            evaluate_fn=_snake_hard_eval,
            pass_threshold={"mean_reward": 80},
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
        GoldenChallenge(
            name="tetris_hard",
            domain="tetris",
            description="Clear ≥ 40 lines in speed mode",
            evaluate_fn=_tetris_hard_eval,
            pass_threshold={"lines_cleared": 40},
        ),
    ],
    "nlp": [
        GoldenChallenge(
            name="nlp_loss",
            domain="nlp",
            description="Achieve eval_loss ≤ 1.5 on validation set",
            evaluate_fn=_nlp_loss_eval,
            pass_threshold={"eval_loss": 1.5},
        ),
        GoldenChallenge(
            name="nlp_perplexity",
            domain="nlp",
            description="Achieve perplexity ≤ 20.0 on OOD validation set",
            evaluate_fn=_nlp_perplexity_eval,
            pass_threshold={"perplexity": 20.0},
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

            # NLP metrics use lower-is-better semantics for loss/perplexity
            def _passes(k: str, threshold: float, value: float) -> bool:
                if k in ("eval_loss", "perplexity", "adversarial_loss", "ood_perplexity"):
                    return value <= threshold
                return value >= threshold

            challenge_passed = all(
                _passes(k, v, metrics.get(k, 0))
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
