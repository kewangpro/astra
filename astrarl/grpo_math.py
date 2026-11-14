"""Generic GRPO math — advantages and PPO-clipped loss (platform-owned).

Ensemble use cases supply rewards via a reward plugin; this module owns the
update math shared across routing GRPO / agent GRPO.
"""
from __future__ import annotations

import math
from typing import Any


def compute_advantages(rewards: list[float], baseline: float = 0.0) -> list[float]:
    """A_i = (r_i - baseline) / (std + eps). Uses running mean as baseline so
    all-fail batches still produce a negative gradient signal instead of silence."""
    if not rewards:
        return []
    variance = sum((r - baseline) ** 2 for r in rewards) / len(rewards)
    std = math.sqrt(variance) + 1e-8
    return [(r - baseline) / std for r in rewards]


def completion_token_logps(model: Any, prompt_ids: list[int], comp_ids: list[int]) -> Any:
    """Per-token log-probs of the completion tokens under model (MLX).

    Softmax is applied only on the completion logit slice to avoid allocating a
    full-sequence vocab tensor on long prompts (RFT / STAR / distill).
    """
    import mlx.core as mx
    import mlx.nn as nn

    full_ids = mx.array(prompt_ids + comp_ids, dtype=mx.uint32)
    inputs = full_ids[:-1]
    targets = full_ids[1:]
    logits = model(inputs[None])[0]
    comp_start = len(prompt_ids) - 1
    comp_logits = logits[comp_start : comp_start + len(comp_ids)]
    comp_targets = targets[comp_start : comp_start + len(comp_ids)]
    log_probs = nn.log_softmax(comp_logits.astype(mx.float32), axis=-1)
    return log_probs[mx.arange(comp_targets.shape[0]), comp_targets]


def seq_log_prob(model: Any, prompt_ids: list[int], comp_ids: list[int]) -> Any:
    """Mean completion-token log-prob (length-normalised GRPO objective)."""
    import mlx.core as mx

    return mx.mean(completion_token_logps(model, prompt_ids, comp_ids))


def seq_log_prob_sum(model: Any, prompt_ids: list[int], comp_ids: list[int]) -> Any:
    """Summed completion-token log-prob (DPO-style sequence log-likelihood)."""
    import mlx.core as mx

    return mx.sum(completion_token_logps(model, prompt_ids, comp_ids))


def ref_log_prob(model: Any, prompt_ids: list[int], comp_ids: list[int]) -> float:
    """Log-prob of a completion under the frozen reference policy (no gradient)."""
    import mlx.core as mx

    lp = seq_log_prob(model, prompt_ids, comp_ids)
    mx.eval(lp)
    return float(lp)


def grpo_loss_single(
    model: Any,
    prompt_ids: list[int],
    comp_ids: list[int],
    advantage: float,
    old_log_prob: float,
    clip_epsilon_low: float,
    clip_epsilon_high: float,
) -> tuple[Any, int]:
    """PPO-clipped GRPO loss for one (prompt, completion, advantage) triple.

    DAPO-lite clip-higher: clip into [1-ε_low, 1+ε_high] (symmetric when equal).
    """
    import mlx.core as mx

    new_lp = seq_log_prob(model, prompt_ids, comp_ids)
    ratio = mx.exp(new_lp - old_log_prob)
    clipped = mx.clip(ratio, 1.0 - clip_epsilon_low, 1.0 + clip_epsilon_high)
    adv = mx.array(float(advantage))
    loss = -mx.minimum(ratio * adv, clipped * adv)
    return loss, len(comp_ids)
