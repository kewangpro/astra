"""OPD (on-policy distillation) advantage math — platform-owned.

Ensemble ``opd_train`` wires student/teacher logprobs + routing rewards;
this module owns reverse-KL → advantage reductions (no MLX weights).
"""
from __future__ import annotations

from typing import Sequence, Union

ArrayLike = Union[Sequence[float], object]


def _as_float_list(tok_lps: ArrayLike) -> list[float]:
    if hasattr(tok_lps, "tolist"):
        return [float(x) for x in tok_lps.tolist()]  # type: ignore[union-attr]
    return [float(x) for x in tok_lps]  # type: ignore[arg-type]


def mean_reverse_kl(
    teacher_tok_lps: ArrayLike,
    student_tok_lps: ArrayLike,
) -> float | None:
    """Token-mean reverse KL = mean(log π_S − log π_T). None if unusable."""
    t = _as_float_list(teacher_tok_lps)
    s = _as_float_list(student_tok_lps)
    if not t or not s or len(t) != len(s):
        return None
    return sum(ss - tt for tt, ss in zip(t, s)) / len(t)


def kl_policy_advantage(rkl: float, kl_coef: float) -> float:
    """Policy-gradient advantage for minimising reverse KL on student tokens.

    rkl = mean(log π_S − log π_T). The PG advantage that *increases* student
    log-prob when the teacher is ahead is
        A = −kl_coef · rkl = kl_coef · mean(log π_T − log π_S).

    Do NOT pathwise-differentiate mean(log π_S − log π_T): with stop-grad
    teacher tokens that gradient is just ∇ mean(log π_S) and minimising it
    collapses the student.
    """
    return -float(kl_coef) * float(rkl)


def clip_advantage(adv: float, adv_clip: float | None) -> float:
    """Symmetric clamp on the PPO advantage (0 / negative / None disables)."""
    if adv_clip is None or float(adv_clip) <= 0.0:
        return float(adv)
    c = float(adv_clip)
    return max(-c, min(c, float(adv)))


def opd_sequence_advantage(
    teacher_tok_lps: ArrayLike,
    student_tok_lps: ArrayLike,
    reward: float,
    kl_coef: float,
    outcome_bonus: float,
) -> float:
    """Uncentered sequence advantage (KL PG + raw outcome). For tests/diagnostics.

    Training centers only the outcome term within each prompt group; use
    ``kl_policy_advantage`` + ``center_group_advantages`` in the loop instead.
    """
    rkl = mean_reverse_kl(teacher_tok_lps, student_tok_lps)
    if rkl is None:
        return float(outcome_bonus) * float(reward)
    return kl_policy_advantage(rkl, kl_coef) + float(outcome_bonus) * float(reward)


def mean_token_logprob(tok_lps: Sequence[float]) -> float:
    """Mean completion-token log-prob — same reduction as GRPO seq log-prob."""
    if not tok_lps:
        return 0.0
    return sum(float(x) for x in tok_lps) / len(tok_lps)


def center_group_advantages(advantages: Sequence[float]) -> list[float]:
    """Subtract the within-group mean (GRPO-style).

    Used ONLY for the optional routing-outcome PPO term — absolute reward scale
    is arbitrary, so relative ranking within the K rollouts is the right signal.
    Do NOT center reverse-KL: the absolute gap toward the teacher is the distill
    objective.
    """
    if not advantages:
        return []
    mean_a = sum(advantages) / len(advantages)
    return [a - mean_a for a in advantages]


__all__ = [
    "mean_reverse_kl",
    "kl_policy_advantage",
    "clip_advantage",
    "opd_sequence_advantage",
    "mean_token_logprob",
    "center_group_advantages",
]
