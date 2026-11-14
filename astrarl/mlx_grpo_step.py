"""One GRPO group step: rollout → score (plugin) → advantages → optional skip.

Platform-owned dynamic sampling + advantage math. The use case supplies
``RewardPlugin`` and tokenization/update hooks stay with the trainer CLI for now.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional

from astrarl.grpo_math import compute_advantages
from astrarl.plugins import RewardPlugin
from astrarl.rollout import RolloutBackend, rewards_are_flat


@dataclass
class GroupRollout:
    """Result of generating and scoring one prompt group."""

    completions: list[str]
    rewards: list[float]
    advantages: list[float]
    skipped: bool = False
    skip_reason: str = ""
    dynamic_resamples: int = 0


def collect_group(
    *,
    rollout: RolloutBackend,
    plugin: RewardPlugin,
    case: dict,
    k: int,
    temperature: float,
    max_tokens: int,
    reward_ema: float,
    ema_alpha: float = 0.1,
    dynamic_sample_retries: int = 0,
) -> tuple[GroupRollout, float]:
    """Generate K completions, score via plugin, apply DAPO-lite dynamic sampling.

    Returns ``(group, updated_reward_ema)``. When ``group.skipped`` is True the
    caller should not run a policy update (flat rewards after retries).
    """
    prompt = plugin.build_prompt(case)
    dynamic_resamples = 0

    def _once() -> tuple[list[str], list[float]]:
        comps = list(
            rollout.generate(
                prompt, n=k, temperature=temperature, max_tokens=max_tokens,
            )
        )
        rews = [float(plugin.score(c, case)) for c in comps]
        return comps, rews

    completions, rewards = _once()
    if dynamic_sample_retries > 0 and rewards_are_flat(rewards):
        for _ in range(dynamic_sample_retries):
            dynamic_resamples += 1
            completions, rewards = _once()
            if not rewards_are_flat(rewards):
                break
        else:
            return (
                GroupRollout(
                    completions=completions,
                    rewards=rewards,
                    advantages=[],
                    skipped=True,
                    skip_reason="flat_rewards",
                    dynamic_resamples=dynamic_resamples,
                ),
                reward_ema,
            )

    batch_mean = sum(rewards) / max(len(rewards), 1)
    new_ema = (1.0 - ema_alpha) * reward_ema + ema_alpha * batch_mean
    advantages = compute_advantages(rewards, baseline=new_ema)
    return (
        GroupRollout(
            completions=completions,
            rewards=rewards,
            advantages=advantages,
            skipped=False,
            dynamic_resamples=dynamic_resamples,
        ),
        new_ema,
    )
