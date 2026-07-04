"""Unit tests for LoopStateMachine._tail_remote_pass_rate — the astra-side
remote log tailing that replaces telemetry POSTs for dpo/grpo missions."""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.loop.state_machine import LoopStateMachine, _PASS_RATE_RE


def _bare_state_machine() -> LoopStateMachine:
    """Construct a LoopStateMachine instance without running __init__ (which
    needs DB/evaluator/etc dependencies not relevant to this method)."""
    sm = LoopStateMachine.__new__(LoopStateMachine)
    sm._sandbox = MagicMock()
    return sm


# ── _PASS_RATE_RE ──────────────────────────────────────────────────────────────

def test_pass_rate_regex_matches_standard_line():
    match = _PASS_RATE_RE.search("Pass rate: 91.7% (122/133)")
    assert match is not None
    assert match.group(1) == "91.7"


def test_pass_rate_regex_matches_baseline_line():
    match = _PASS_RATE_RE.search("Baseline: 75.0% (15/20)")
    assert match is None  # only matches the "Pass rate:" prefix, not "Baseline:"


# ── _tail_remote_pass_rate ────────────────────────────────────────────────────

class TestTailRemotePassRate:
    def test_no_new_output_returns_step_unchanged(self):
        sm = _bare_state_machine()
        sm._sandbox.tail_new_output.return_value = ""
        step = asyncio.get_event_loop().run_until_complete(
            sm._tail_remote_pass_rate("mission-1", 5)
        )
        assert step == 5

    def test_single_pass_rate_line_emits_one_metric_and_increments_step(self):
        sm = _bare_state_machine()
        sm._sandbox.tail_new_output.return_value = "Some log noise\nPass rate: 60.0% (12/20)\n"

        with patch("backend.loop.state_machine.emit_metric", new_callable=AsyncMock) as mock_emit:
            step = asyncio.get_event_loop().run_until_complete(
                sm._tail_remote_pass_rate("mission-1", 0)
            )

        assert step == 1
        mock_emit.assert_awaited_once_with("mission-1", "pass_rate", 0.6, step=0, iteration=0)

    def test_multiple_pass_rate_lines_increment_step_each_time(self):
        sm = _bare_state_machine()
        sm._sandbox.tail_new_output.return_value = (
            "Pass rate: 50.0% (10/20)\n"
            "some other output\n"
            "Pass rate: 55.0% (11/20)\n"
        )

        with patch("backend.loop.state_machine.emit_metric", new_callable=AsyncMock) as mock_emit:
            step = asyncio.get_event_loop().run_until_complete(
                sm._tail_remote_pass_rate("mission-1", 3)
            )

        assert step == 5   # started at 3, two matches
        assert mock_emit.await_count == 2
        first_call = mock_emit.await_args_list[0]
        assert first_call.args == ("mission-1", "pass_rate", 0.5)
        assert first_call.kwargs == {"step": 3, "iteration": 3}

    def test_tail_exception_returns_step_unchanged(self):
        """A tail failure (e.g. transient SSH error) must not crash the poll loop."""
        sm = _bare_state_machine()
        sm._sandbox.tail_new_output.side_effect = RuntimeError("ssh timeout")

        step = asyncio.get_event_loop().run_until_complete(
            sm._tail_remote_pass_rate("mission-1", 2)
        )
        assert step == 2

    def test_unsupported_backend_returning_none_is_a_noop(self):
        """tail_new_output() returns None for backends without live tailing."""
        sm = _bare_state_machine()
        sm._sandbox.tail_new_output.return_value = None
        step = asyncio.get_event_loop().run_until_complete(
            sm._tail_remote_pass_rate("mission-1", 7)
        )
        assert step == 7
