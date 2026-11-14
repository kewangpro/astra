"""RLCD strictly proper scores, LOO advantages, and readout grads (platform-owned)."""
from __future__ import annotations

import math
from typing import Any, Sequence


def softmax(logits: Sequence[float]) -> list[float]:
    if not logits:
        return []
    peak = max(logits)
    exps = [math.exp(z - peak) for z in logits]
    total = sum(exps)
    return [e / total for e in exps]


def proper_score(u: list[float], y: int, score: str = "brier") -> float:
    """Strictly proper score, higher is better. Brier J = 2 u_y − ||u||²."""
    if score == "log":
        return math.log(max(u[y], 1e-12))
    return 2.0 * u[y] - sum(p * p for p in u)


def standard_brier(u: list[float], y: int) -> float:
    """||u − e_y||². Lower is better. Equals 1 − J for the Brier score above."""
    return sum((p - (1.0 if i == y else 0.0)) ** 2 for i, p in enumerate(u))


def readout_logit_grads(u: list[float], y: int, score: str = "brier") -> list[float]:
    """∂J/∂z for u = softmax(z). Sums to 0 (softmax is shift-invariant).

    Pathwise stop-grad coefficient for per-option readout backprop (see
    ``rlcd_train`` stage-1 loop).
    """
    g = [0.0] * len(u)
    if score == "log":
        g[y] = 1.0 / max(u[y], 1e-12)
    else:
        for i, p in enumerate(u):
            g[i] = -2.0 * p
        g[y] += 2.0
    dot = sum(gi * pi for gi, pi in zip(g, u))
    return [pi * (gi - dot) for gi, pi in zip(g, u)]


def leave_one_out_advantages(
    rewards: Sequence[float],
    log_ratios: Sequence[float],
    beta: float,
) -> list[float]:
    """A_i = (1/M) (R_i − β log π/π_ref) minus the leave-one-out mean.

    M >= 2. Equal rewards and a zero KL give advantage 0. A rationale the
    policy has up-weighted relative to the anchor (positive log-ratio) is
    penalised.
    """
    m = len(rewards)
    if m < 2:
        raise ValueError(f"leave-one-out baseline needs M >= 2, got {m}")
    if len(log_ratios) != m:
        raise ValueError("log_ratios must align with rewards")
    shaped = [(r - beta * lr) / m for r, lr in zip(rewards, log_ratios)]
    advantages = []
    for i, s in enumerate(shaped):
        others = [shaped[j] for j in range(m) if j != i]
        advantages.append(s - sum(others) / len(others))
    return advantages


def fit_prompt(prompt_ids: list[int], comp_ids: list[int], limit: int) -> list[int]:
    """Keep the completion intact and tail-truncate the prompt to ``limit``."""
    if not limit or len(prompt_ids) + len(comp_ids) <= limit:
        return prompt_ids
    keep = max(1, limit - len(comp_ids))
    return prompt_ids[-keep:]


def pathwise_option_loss(
    model: Any,
    prompt_ids: list[int],
    option_ids: list[int],
    coef: float,
    limit: int,
) -> Any:
    """−coef · z, with coef = (∂J/∂z) / M. Minimising this ascends J."""
    import mlx.core as mx

    from astrarl.grpo_math import completion_token_logps

    prompt_ids = fit_prompt(prompt_ids, option_ids, limit)
    z = mx.sum(completion_token_logps(model, prompt_ids, option_ids))
    return -mx.array(float(coef), dtype=mx.float32) * z


def score_function_loss(
    model: Any,
    prompt_ids: list[int],
    rationale_ids: list[int],
    coef: float,
    limit: int,
) -> Any:
    """−coef · log π(r), coef = c · A_i. A_i already includes the 1/M factor."""
    import mlx.core as mx

    from astrarl.grpo_math import completion_token_logps

    prompt_ids = fit_prompt(prompt_ids, rationale_ids, limit)
    logp = mx.sum(completion_token_logps(model, prompt_ids, rationale_ids))
    return -mx.array(float(coef), dtype=mx.float32) * logp


__all__ = [
    "softmax",
    "proper_score",
    "standard_brier",
    "readout_logit_grads",
    "leave_one_out_advantages",
    "fit_prompt",
    "pathwise_option_loss",
    "score_function_loss",
]
