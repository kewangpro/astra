"""Pure-Python tests for AstraTune loss / score math (no MLX weights)."""
from __future__ import annotations

import math

import pytest

from astratune.dpo_math import dpo_loss_from_logprobs
from astratune.rlcd_math import proper_score, readout_logit_grads, standard_brier


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def test_dpo_loss_from_logprobs_matches_sigmoid_formula() -> None:
    policy_c, policy_r = -3.0, -5.0
    ref_c, ref_r = -3.5, -4.8
    beta = 0.1
    margin = (policy_c - ref_c) - (policy_r - ref_r)
    logit = beta * margin
    expected = -math.log(_sigmoid(logit))
    assert dpo_loss_from_logprobs(policy_c, policy_r, ref_c, ref_r, beta) == pytest.approx(
        expected
    )


def test_dpo_loss_prefers_higher_chosen_margin() -> None:
    ref_c, ref_r = 0.0, 0.0
    beta = 0.2
    good = dpo_loss_from_logprobs(2.0, -1.0, ref_c, ref_r, beta)
    bad = dpo_loss_from_logprobs(-1.0, 2.0, ref_c, ref_r, beta)
    assert good < bad


def test_rlcd_brier_proper_score_identity() -> None:
    u = [0.7, 0.2, 0.1]
    y = 0
    assert proper_score(u, y) == pytest.approx(1.0 - standard_brier(u, y))


def test_rlcd_brier_extremes() -> None:
    assert proper_score([1.0, 0.0], 0) == pytest.approx(1.0)
    assert proper_score([0.0, 1.0], 0) == pytest.approx(-1.0)
    assert standard_brier([1.0, 0.0], 0) == pytest.approx(0.0)
    assert standard_brier([0.0, 1.0], 0) == pytest.approx(2.0)


def test_rlcd_log_score() -> None:
    assert proper_score([1.0, 0.0], 0, "log") == pytest.approx(0.0)
    assert proper_score([0.5, 0.5], 1, "log") == pytest.approx(-math.log(2.0))


def test_readout_logit_grads_sum_to_zero() -> None:
    u = [0.5, 0.3, 0.2]
    grads = readout_logit_grads(u, 1, "brier")
    assert sum(grads) == pytest.approx(0.0, abs=1e-9)