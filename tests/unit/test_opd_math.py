"""Pure-Python tests for astrarl.opd_math (no MLX weights)."""
from __future__ import annotations

import pytest

from astrarl.opd_math import (
    center_group_advantages,
    clip_advantage,
    kl_policy_advantage,
    mean_reverse_kl,
    mean_token_logprob,
    opd_sequence_advantage,
)


def test_mean_reverse_kl_is_student_minus_teacher() -> None:
    teacher = [-1.0, -2.0]
    student = [-2.0, -3.0]
    assert mean_reverse_kl(teacher, student) == pytest.approx(-1.0)


def test_mean_reverse_kl_none_on_empty_or_mismatch() -> None:
    assert mean_reverse_kl([], []) is None
    assert mean_reverse_kl([-0.1], [-0.1, -0.2]) is None


def test_kl_policy_advantage_is_neg_rkl() -> None:
    assert kl_policy_advantage(-1.0, kl_coef=1.0) == pytest.approx(1.0)
    assert kl_policy_advantage(1.5, kl_coef=1.0) == pytest.approx(-1.5)


def test_clip_advantage() -> None:
    assert clip_advantage(5.0, 2.0) == 2.0
    assert clip_advantage(-5.0, 2.0) == -2.0
    assert clip_advantage(1.0, 0.0) == 1.0
    assert clip_advantage(1.0, None) == 1.0


def test_opd_sequence_advantage_kl_plus_outcome() -> None:
    teacher = [-0.1, -0.2]
    student = [-1.1, -1.2]  # rkl = -1.0
    adv = opd_sequence_advantage(
        teacher, student, reward=1.0, kl_coef=1.0, outcome_bonus=0.5,
    )
    # kl_policy_advantage(-1, 1) = 1.0; + 0.5 * 1.0
    assert adv == pytest.approx(1.5)


def test_opd_sequence_advantage_fallback_on_bad_lps() -> None:
    assert opd_sequence_advantage(
        [], [], reward=1.0, kl_coef=2.0, outcome_bonus=0.3,
    ) == pytest.approx(0.3)


def test_center_group_advantages() -> None:
    assert center_group_advantages([1.0, 3.0]) == pytest.approx([ -1.0, 1.0 ])


def test_mean_token_logprob() -> None:
    assert mean_token_logprob([-1.0, -3.0]) == pytest.approx(-2.0)
    assert mean_token_logprob([]) == 0.0
