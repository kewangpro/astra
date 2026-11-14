"""Stdout contract + checkpoint selection helpers (no MLX)."""
from __future__ import annotations

from astratune.stdout_contract import (
    BASELINE_PREFIX,
    PASS_RATE_PREFIX,
    format_pass_rate,
    format_step_loss,
    is_better_checkpoint,
)


def test_format_pass_rate() -> None:
    assert format_pass_rate("Pass rate", 0.818, 9, 11).startswith(PASS_RATE_PREFIX)
    assert "81.8%" in format_pass_rate("Baseline", 0.818, 9, 11)
    assert format_pass_rate("Baseline", 0.5, 1, 2).startswith(BASELINE_PREFIX)


def test_format_step_loss() -> None:
    assert format_step_loss(3, 10, 0.12345, 12.0).startswith("Step 3/10 loss=0.1235")


def test_is_better_checkpoint_static_wins() -> None:
    assert is_better_checkpoint(0.9, 0.8, (10, 12), (9, 12))
    assert not is_better_checkpoint(0.7, 0.8, (10, 12), (9, 12))


def test_is_better_checkpoint_blended_tiebreak() -> None:
    assert is_better_checkpoint(0.8, 0.8, (10, 12), (9, 12))
    assert not is_better_checkpoint(0.8, 0.8, (9, 12), (10, 12))
