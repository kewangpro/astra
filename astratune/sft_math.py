"""SFT / distillation cross-entropy helpers (platform-owned).

Masked completion-only CE matches ``mlx_lm.lora --mask-prompt`` and Ensemble
``distill_train`` / ``rft_train``: mean negative log-likelihood over the
selected token positions.
"""
from __future__ import annotations

from typing import Any, Sequence


def mean_neg_log_likelihood(token_log_probs: Sequence[float]) -> float:
    """``-mean(log π(token))`` over an explicit per-token log-prob list (tests / mocks)."""
    if not token_log_probs:
        raise ValueError("token_log_probs must be non-empty")
    return -sum(token_log_probs) / len(token_log_probs)


def masked_ce_loss_mlx(model: Any, prompt_ids: list[int], comp_ids: list[int]) -> Any:
    """Mean NLL over completion tokens only (prompt masked out) — MLX array scalar."""
    import mlx.core as mx

    from astrarl.grpo_math import completion_token_logps

    return -mx.mean(completion_token_logps(model, prompt_ids, comp_ids))


def full_sequence_ce_loss_mlx(model: Any, prompt_ids: list[int], comp_ids: list[int]) -> Any:
    """Mean NLL over prompt+completion (``--no-mask-prompt`` behaviour)."""
    import mlx.core as mx
    import mlx.nn as nn

    full_ids = mx.array(prompt_ids + comp_ids, dtype=mx.uint32)
    inputs = full_ids[:-1]
    targets = full_ids[1:]
    logits = model(inputs[None])[0]
    log_probs = nn.log_softmax(logits.astype(mx.float32), axis=-1)
    tok_lp = log_probs[mx.arange(targets.shape[0]), targets]
    return -mx.mean(tok_lp)