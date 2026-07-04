"""Unit tests for LoopStateMachine._tail_remote_metrics — the astra-side
remote log tailing that replaces telemetry POSTs for dpo/grpo missions,
covering both the "pass_rate" goal metric and the "loss" training signal."""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.loop.state_machine import (
    LoopStateMachine, _PASS_RATE_RE, _GRPO_LOSS_RE, _DPO_LOSS_RE, _COLLECT_PROGRESS_RE,
)


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


# ── _GRPO_LOSS_RE / _DPO_LOSS_RE ───────────────────────────────────────────────

def test_grpo_loss_regex_matches_step_line():
    line = "Step   25/300 | loss=0.6421 | baseline=0.812 | rewards=[1.0 0.3] | route_x | 120s"
    match = _GRPO_LOSS_RE.search(line)
    assert match is not None
    assert match.group(1) == "25"
    assert match.group(2) == "0.6421"


def test_dpo_loss_regex_matches_epoch_line():
    line = "\n=== Epoch 2/3 done  avg_loss=0.5891 ===\n"
    match = _DPO_LOSS_RE.search(line)
    assert match is not None
    assert match.group(1) == "2"
    assert match.group(2) == "0.5891"


# ── _tail_remote_metrics ───────────────────────────────────────────────────────

class TestTailRemoteMetrics:
    def test_no_new_output_returns_step_unchanged(self):
        sm = _bare_state_machine()
        sm._sandbox.tail_new_output.return_value = ""
        step = asyncio.get_event_loop().run_until_complete(
            sm._tail_remote_metrics("mission-1", "grpo", 5)
        )
        assert step == 5

    def test_single_pass_rate_line_emits_one_metric_and_increments_step(self):
        sm = _bare_state_machine()
        sm._sandbox.tail_new_output.return_value = "Some log noise\nPass rate: 60.0% (12/20)\n"

        with patch("backend.loop.state_machine.emit_metric", new_callable=AsyncMock) as mock_emit:
            step = asyncio.get_event_loop().run_until_complete(
                sm._tail_remote_metrics("mission-1", "grpo", 0)
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
                sm._tail_remote_metrics("mission-1", "grpo", 3)
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
            sm._tail_remote_metrics("mission-1", "grpo", 2)
        )
        assert step == 2

    def test_unsupported_backend_returning_none_is_a_noop(self):
        """tail_new_output() returns None for backends without live tailing."""
        sm = _bare_state_machine()
        sm._sandbox.tail_new_output.return_value = None
        step = asyncio.get_event_loop().run_until_complete(
            sm._tail_remote_metrics("mission-1", "grpo", 7)
        )
        assert step == 7

    def test_grpo_step_line_emits_loss_metric_with_own_step_number(self):
        sm = _bare_state_machine()
        sm._sandbox.tail_new_output.return_value = (
            "Step   25/300 | loss=0.6421 | baseline=0.812 | rewards=[1.0] | case | 120s\n"
        )

        with patch("backend.loop.state_machine.emit_metric", new_callable=AsyncMock) as mock_emit:
            asyncio.get_event_loop().run_until_complete(
                sm._tail_remote_metrics("mission-1", "grpo", 0)
            )

        mock_emit.assert_awaited_once_with("mission-1", "loss", 0.6421, step=25, iteration=25)

    def test_dpo_epoch_line_emits_loss_metric_with_epoch_number(self):
        sm = _bare_state_machine()
        sm._sandbox.tail_new_output.return_value = "\n=== Epoch 2/3 done  avg_loss=0.5891 ===\n"

        with patch("backend.loop.state_machine.emit_metric", new_callable=AsyncMock) as mock_emit:
            asyncio.get_event_loop().run_until_complete(
                sm._tail_remote_metrics("mission-1", "dpo", 0)
            )

        mock_emit.assert_awaited_once_with("mission-1", "loss", 0.5891, step=2, iteration=2)

    def test_grpo_task_type_does_not_match_dpo_style_loss_line(self):
        """Wrong regex for the task type must not accidentally match."""
        sm = _bare_state_machine()
        sm._sandbox.tail_new_output.return_value = "\n=== Epoch 2/3 done  avg_loss=0.5891 ===\n"

        with patch("backend.loop.state_machine.emit_metric", new_callable=AsyncMock) as mock_emit:
            asyncio.get_event_loop().run_until_complete(
                sm._tail_remote_metrics("mission-1", "grpo", 0)
            )

        mock_emit.assert_not_awaited()

    def test_pass_rate_and_loss_both_emitted_from_same_tail(self):
        sm = _bare_state_machine()
        sm._sandbox.tail_new_output.return_value = (
            "Step   25/300 | loss=0.6421 | baseline=0.812 | rewards=[1.0] | case | 120s\n"
            "Pass rate: 78.0% (23/30)\n"
        )

        with patch("backend.loop.state_machine.emit_metric", new_callable=AsyncMock) as mock_emit:
            asyncio.get_event_loop().run_until_complete(
                sm._tail_remote_metrics("mission-1", "grpo", 0)
            )

        assert mock_emit.await_count == 2
        names = [call.args[1] for call in mock_emit.await_args_list]
        assert "pass_rate" in names
        assert "loss" in names


# ── _COLLECT_PROGRESS_RE / pair-collection status ──────────────────────────────

def test_collect_progress_regex_matches_dpo_collection_line():
    match = _COLLECT_PROGRESS_RE.search("  [40/66]  37 pairs  (3310s)")
    assert match is not None
    assert match.groups() == ("40", "66", "37", "3310")


class TestCollectProgressStatus:
    def test_collection_line_emits_status_not_metric(self):
        sm = _bare_state_machine()
        sm._sandbox.tail_new_output.return_value = "  [40/66]  37 pairs  (3310s)\n"

        with patch("backend.loop.state_machine.emit_metric", new_callable=AsyncMock) as mock_metric, \
             patch("backend.loop.state_machine.emit_status", new_callable=AsyncMock) as mock_status:
            asyncio.get_event_loop().run_until_complete(
                sm._tail_remote_metrics("mission-1", "dpo", 0)
            )

        mock_metric.assert_not_awaited()
        mock_status.assert_awaited_once()
        args, kwargs = mock_status.await_args
        assert "40/66" in args[1]
        assert "37 pairs" in args[1]
        assert "55m" in args[1]
        assert kwargs["event_type"] == "info"

    def test_multiple_collection_lines_only_emit_latest(self):
        sm = _bare_state_machine()
        sm._sandbox.tail_new_output.return_value = (
            "  [10/66]  10 pairs  (834s)\n"
            "  [20/66]  19 pairs  (1675s)\n"
        )

        with patch("backend.loop.state_machine.emit_status", new_callable=AsyncMock) as mock_status:
            asyncio.get_event_loop().run_until_complete(
                sm._tail_remote_metrics("mission-1", "dpo", 0)
            )

        mock_status.assert_awaited_once()
        assert "20/66" in mock_status.await_args.args[1]
