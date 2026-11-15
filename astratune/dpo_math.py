"""DPO preference loss math (platform-owned, summed sequence log-likelihood).

log π(y|x) is the **summed** completion-token log-prob (sequence log-likelihood),
not the mean reduction GRPO uses. That matches Rafailov et al. and product
``dpo_train`` / ``astrarl.grpo_math.seq_log_prob_sum``.
"""
from __future__ import annotations

import math
from typing import Any


def dpo_loss_from_logprobs(
    policy_chosen: float,
    policy_rejected: float,
    ref_chosen: float,
    ref_rejected: float,
    beta: float,
) -> float:
    """Scalar DPO loss from pre-computed sequence log-probs (no MLX).

    loss = -log σ(β · ((log π_θ(c|x) − log π_ref(c|x))
                      − (log π_θ(r|x) − log π_ref(r|x))))
    """
    chosen_ratio = policy_chosen - ref_chosen
    rejected_ratio = policy_rejected - ref_rejected
    logit = beta * (chosen_ratio - rejected_ratio)
    # Stable -log(sigmoid(logit)) = log(1 + exp(-logit))
    if logit >= 0:
        return math.log1p(math.exp(-logit))
    return -logit + math.log1p(math.exp(logit))


def dpo_loss_mlx(
    chosen_lp: Any,
    rejected_lp: Any,
    ref_chosen_lp: float,
    ref_rejected_lp: float,
    beta: float,
) -> Any:
    """DPO loss when policy log-probs are already MLX arrays (differentiable)."""
    import mlx.core as mx
    import mlx.nn as nn

    chosen_ratio = chosen_lp - float(ref_chosen_lp)
    rejected_ratio = rejected_lp - float(ref_rejected_lp)
    logit = mx.array(float(beta)) * (chosen_ratio - rejected_ratio)
    return -nn.log_sigmoid(logit)


def dpo_loss(
    model: Any,
    prompt_ids: list[int],
    chosen_ids: list[int],
    rejected_ids: list[int],
    ref_chosen_lp: float,
    ref_rejected_lp: float,
    beta: float,
) -> Any:
    """Standard DPO loss for one pair — matches legacy ``dpo_train.dpo_loss``."""
    from astrarl.grpo_math import seq_log_prob_sum

    chosen_lp = seq_log_prob_sum(model, prompt_ids, chosen_ids)
    rejected_lp = seq_log_prob_sum(model, prompt_ids, rejected_ids)
    return dpo_loss_mlx(
        chosen_lp,
        rejected_lp,
        ref_chosen_lp,
        ref_rejected_lp,
        beta,
    )