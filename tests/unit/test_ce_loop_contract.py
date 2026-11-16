"""Stdout contract + checkpoint selection helpers (no MLX)."""
from __future__ import annotations

import re

from astratune.stdout_contract import (
    BASELINE_PASS_RATE_PREFIX,
    BASELINE_PREFIX,
    GRPO_STEP_LOSS_RE,
    PASS_RATE_PREFIX,
    STEP_LOSS_RE,
    format_grpo_step_loss,
    format_pass_rate,
    format_step_loss,
    is_better_checkpoint,
)


def test_format_pass_rate() -> None:
    assert format_pass_rate("Pass rate", 0.818, 9, 11).startswith(PASS_RATE_PREFIX)
    assert "81.8%" in format_pass_rate("Baseline", 0.818, 9, 11)
    assert format_pass_rate("Baseline", 0.5, 1, 2).startswith(BASELINE_PREFIX)
    assert format_pass_rate("Baseline pass rate", 0.853, 58, 68).startswith(
        BASELINE_PASS_RATE_PREFIX
    )


def test_format_step_loss() -> None:
    assert format_step_loss(3, 10, 0.12345, 12.0).startswith("Step 3/10 loss=0.1235")
    assert re.search(STEP_LOSS_RE, format_step_loss(1, 3, -1.0, 40.0))


def test_format_grpo_step_loss_matches_contract_regex() -> None:
    line = format_grpo_step_loss(1, 3, -1.0, baseline=0.1)
    assert "| loss=-1.0000" in line
    assert re.search(GRPO_STEP_LOSS_RE, line)


def test_is_better_checkpoint_static_wins() -> None:
    assert is_better_checkpoint(0.9, 0.8, (10, 12), (9, 12))
    assert not is_better_checkpoint(0.7, 0.8, (10, 12), (9, 12))


def test_is_better_checkpoint_blended_tiebreak() -> None:
    assert is_better_checkpoint(0.8, 0.8, (10, 12), (9, 12))
    assert not is_better_checkpoint(0.8, 0.8, (9, 12), (10, 12))
