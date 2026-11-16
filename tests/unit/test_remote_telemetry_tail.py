"""Unit tests for LoopStateMachine._tail_remote_metrics — mirror sandbox.log
as type=log and extract typed metrics (pass_rate / loss) for charts."""
from __future__ import annotations

import asyncio
import json
from collections import Counter
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.loop.state_machine import (
    LoopStateMachine,
    _PASS_RATE_RE,
    _GRPO_LOSS_RE,
    _DPO_LOSS_RE,
    _DISTILL_LOSS_RE,
    filter_sandbox_log_lines,
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


def test_pass_rate_regex_matches_grpo_baseline_pass_rate_line():
    """astrarl train_grpo_routing prints ``Baseline pass rate:`` (lowercase p)."""
    match = _PASS_RATE_RE.search("Baseline pass rate: 85.3% (58/68)")
    assert match is not None
    assert match.group(1) == "85.3"
    assert match.group(2) == "58"
    assert match.group(3) == "68"


def test_pass_rate_regex_ignores_bare_baseline_without_pass_rate():
    match = _PASS_RATE_RE.search("Baseline: 75.0% (15/20)")
    assert match is None


def test_pass_rate_regex_ignores_held_out_line():
    """Held-out is informational; must not inflate the goal pass_rate series."""
    assert _PASS_RATE_RE.search("Held-out pass rate: 80.0% (8/10)") is None
    assert _PASS_RATE_RE.search("Baseline pass rate: 85.3% (58/68)") is not None


# ── _GRPO_LOSS_RE / _DPO_LOSS_RE ───────────────────────────────────────────────

def test_grpo_loss_regex_matches_step_line():
    line = "Step   25/300 | loss=0.6421 | baseline=0.812 | rewards=[1.0 0.3] | route_x | 120s"
    match = _GRPO_LOSS_RE.search(line)
    assert match is not None
    assert match.group(1) == "25"
    assert match.group(2) == "0.6421"


def test_grpo_loss_regex_matches_negative_sentinel_loss():
    """Live Mini smoke printed ``loss=-1.0000``; unsigned ``[\\d.]+`` missed it."""
    line = "Step    1/3 | loss=-1.0000 | baseline=0.100 | rewards=[1.0 1.0] | route_x | 40s"
    match = _GRPO_LOSS_RE.search(line)
    assert match is not None
    assert match.group(1) == "1"
    assert match.group(2) == "-1.0000"


def test_dpo_loss_regex_matches_epoch_line():
    line = "\n=== Epoch 2/3 done  avg_loss=0.5891 ===\n"
    match = _DPO_LOSS_RE.search(line)
    assert match is not None
    assert match.group(1) == "2"
    assert match.group(2) == "0.5891"


def test_distill_loss_regex_matches_step_line():
    line = "Step   40/500  epoch=1  loss=0.4210  312s"
    match = _DISTILL_LOSS_RE.search(line)
    assert match is not None
    assert match.group(1) == "40"
    assert match.group(2) == "0.4210"


def test_distill_loss_regex_matches_opd_negative_and_rkl():
    """OPD prints `loss=… rkl=…`; reverse KL can be negative."""
    line = "Step 12/200 loss=-0.3125 rkl=-0.3125  88s"
    match = _DISTILL_LOSS_RE.search(line)
    assert match is not None
    assert match.group(1) == "12"
    assert match.group(2) == "-0.3125"


# ── filter_sandbox_log_lines ───────────────────────────────────────────────────

class TestFilterSandboxLogLines:
    def test_keeps_unknown_banner(self):
        lines = filter_sandbox_log_lines(
            "Caching reference log-probs for 59 pairs...\n"
            "  [20/59]  ref log-probs cached\n"
        )
        assert lines == [
            "Caching reference log-probs for 59 pairs...",
            "[20/59]  ref log-probs cached",
        ]

    def test_drops_empty_fetching_tqdm_and_consecutive_dupes(self):
        text = (
            "\n"
            "Loading model: gemma\n"
            "Loading model: gemma\n"
            "Fetching 13 files: 100%|██████████| 13/13 [00:00<?, ?it/s]\n"
            "45%|████| 45/100 [00:01<00:01, 12.0it/s]\n"
            "Pass rate: 88.2% (60/68)\n"
        )
        assert filter_sandbox_log_lines(text) == [
            "Loading model: gemma",
            "Pass rate: 88.2% (60/68)",
        ]

    def test_cr_progress_keeps_last_segment(self):
        lines = filter_sandbox_log_lines("foo\rbar\rbaz final\n")
        assert lines == ["baz final"]


# ── _tail_remote_metrics ───────────────────────────────────────────────────────

class TestTailRemoteMetrics:
    def test_no_new_output_returns_step_unchanged(self):
        sm = _bare_state_machine()
        sm._sandbox.tail_new_output.return_value = ""
        step = asyncio.get_event_loop().run_until_complete(
            sm._tail_remote_metrics("mission-1", "grpo", 5)
        )
        assert step == 5

    def test_pass_rate_line_emits_metric_and_increments_step(self):
        sm = _bare_state_machine()
        sm._sandbox.tail_new_output.return_value = "Pass rate: 91.7% (122/133)\n"

        with patch("backend.loop.state_machine.emit_metric", new_callable=AsyncMock) as mock_emit, \
             patch("backend.loop.state_machine.emit_log", new_callable=AsyncMock):
            step = asyncio.get_event_loop().run_until_complete(
                sm._tail_remote_metrics("mission-1", "grpo", 0)
            )

        assert step == 1
        mock_emit.assert_awaited_once_with("mission-1", "pass_rate", pytest.approx(0.917), step=0, iteration=0)

    def test_pass_rate_tags_real_mission_iteration(self):
        sm = _bare_state_machine()
        sm._sandbox.tail_new_output.return_value = "Pass rate: 80.0% (8/10)\n"

        with patch("backend.loop.state_machine.emit_metric", new_callable=AsyncMock) as mock_emit, \
             patch("backend.loop.state_machine.emit_log", new_callable=AsyncMock):
            asyncio.get_event_loop().run_until_complete(
                sm._tail_remote_metrics("mission-1", "grpo", 3, current_iteration=7)
            )

        mock_emit.assert_awaited_once_with("mission-1", "pass_rate", 0.8, step=3, iteration=7)

    def test_dpo_epoch_avg_loss_emitted(self):
        sm = _bare_state_machine()
        sm._sandbox.tail_new_output.return_value = "\n=== Epoch 2/3 done  avg_loss=0.5891 ===\n"

        with patch("backend.loop.state_machine.emit_metric", new_callable=AsyncMock) as mock_emit, \
             patch("backend.loop.state_machine.emit_log", new_callable=AsyncMock):
            asyncio.get_event_loop().run_until_complete(
                sm._tail_remote_metrics("mission-1", "dpo", 0, current_iteration=1)
            )

        mock_emit.assert_awaited_once_with("mission-1", "avg_loss", 0.5891, step=2, iteration=1)

    def test_dpo_epoch_then_another_chunk_increments_independently(self):
        sm = _bare_state_machine()
        sm._sandbox.tail_new_output.return_value = "\n=== Epoch 1/3 done  avg_loss=0.7 ===\n"

        with patch("backend.loop.state_machine.emit_metric", new_callable=AsyncMock), \
             patch("backend.loop.state_machine.emit_log", new_callable=AsyncMock):
            step = asyncio.get_event_loop().run_until_complete(
                sm._tail_remote_metrics("mission-1", "dpo", 0)
            )
            assert step == 0  # epoch avg does not bump pass_rate_step
            sm._sandbox.tail_new_output.return_value = "Pass rate: 50.0% (1/2)\n"
            step = asyncio.get_event_loop().run_until_complete(
                sm._tail_remote_metrics("mission-1", "dpo", step)
            )
            assert step == 1

    def test_exception_from_tail_returns_step_unchanged(self):
        sm = _bare_state_machine()
        sm._sandbox.tail_new_output.side_effect = RuntimeError("ssh failed")
        step = asyncio.get_event_loop().run_until_complete(
            sm._tail_remote_metrics("mission-1", "grpo", 2)
        )
        assert step == 2

    def test_no_matching_lines_returns_step_unchanged(self):
        sm = _bare_state_machine()
        sm._sandbox.tail_new_output.return_value = "hello world\n"

        with patch("backend.loop.state_machine.emit_metric", new_callable=AsyncMock) as mock_emit, \
             patch("backend.loop.state_machine.emit_log", new_callable=AsyncMock):
            step = asyncio.get_event_loop().run_until_complete(
                sm._tail_remote_metrics("mission-1", "grpo", 7)
            )
        assert step == 7
        mock_emit.assert_not_awaited()

    def test_grpo_loss_emitted_with_script_step_and_mission_iteration(self):
        sm = _bare_state_machine()
        sm._sandbox.tail_new_output.return_value = (
            "Step   25/300 | loss=0.6421 | baseline=0.812 | rewards=[1.0] | case | 120s\n"
        )

        with patch("backend.loop.state_machine.emit_metric", new_callable=AsyncMock) as mock_emit, \
             patch("backend.loop.state_machine.emit_log", new_callable=AsyncMock):
            asyncio.get_event_loop().run_until_complete(
                sm._tail_remote_metrics("mission-1", "grpo", 0, current_iteration=3)
            )

        mock_emit.assert_awaited_once_with("mission-1", "loss", 0.6421, step=25, iteration=3)

    def test_grpo_baseline_pass_rate_and_negative_loss_emitted(self):
        sm = _bare_state_machine()
        sm._sandbox.tail_new_output.return_value = (
            "--- Baseline eval (step 0) ---\n"
            "Baseline pass rate: 85.3% (58/68)\n"
            "Held-out pass rate: 80.0% (8/10)\n"
            "Step    1/3 | loss=-1.0000 | baseline=0.100 | rewards=[1.0 1.0] | case | 40s\n"
            "Step    2/3 | loss=-1.0000 | baseline=0.190 | rewards=[1.0 1.0] | case | 85s\n"
            "--- Eval at step 3 ---\n"
        )

        with patch("backend.loop.state_machine.emit_metric", new_callable=AsyncMock) as mock_emit, \
             patch("backend.loop.state_machine.emit_log", new_callable=AsyncMock) as mock_log:
            step = asyncio.get_event_loop().run_until_complete(
                sm._tail_remote_metrics("mission-1", "grpo", 0, current_iteration=0)
            )

        assert step == 1
        pr = [c for c in mock_emit.await_args_list if c.args[1] == "pass_rate"]
        assert len(pr) == 1
        assert pr[0].args[2] == pytest.approx(0.853)
        losses = [(c.args[2], c.kwargs["step"]) for c in mock_emit.await_args_list if c.args[1] == "loss"]
        assert (-1.0, 1) in losses
        assert (-1.0, 2) in losses
        log_msgs = [c.args[1] for c in mock_log.await_args_list]
        assert "--- Baseline eval (step 0) ---" in log_msgs
        assert "Baseline pass rate: 85.3% (58/68)" in log_msgs
        assert "Held-out pass rate: 80.0% (8/10)" in log_msgs
        assert any(m.startswith("Step") for m in log_msgs)

    def test_unknown_banner_mirrored_without_status_regex(self):
        """Ref-LP cache (and any future banner) must appear as type=log."""
        sm = _bare_state_machine()
        sm._sandbox.tail_new_output.return_value = (
            "Caching reference log-probs for 59 pairs...\n"
            "  [40/59]  ref log-probs cached\n"
        )

        with patch("backend.loop.state_machine.emit_metric", new_callable=AsyncMock) as mock_metric, \
             patch("backend.loop.state_machine.emit_log", new_callable=AsyncMock) as mock_log, \
             patch("backend.loop.state_machine.emit_status", new_callable=AsyncMock) as mock_status:
            asyncio.get_event_loop().run_until_complete(
                sm._tail_remote_metrics("mission-1", "dpo", 0)
            )

        mock_metric.assert_not_awaited()
        mock_status.assert_not_awaited()
        msgs = [c.args[1] for c in mock_log.await_args_list]
        assert "Caching reference log-probs for 59 pairs..." in msgs
        assert "[40/59]  ref log-probs cached" in msgs

    def test_dpo_epoch_line_emits_avg_loss_not_loss(self):
        """Epoch avg must not reuse the ``loss`` series (would collide with
        per-step loss at step=epoch on the Metric History x-axis)."""
        sm = _bare_state_machine()
        sm._sandbox.tail_new_output.return_value = "\n=== Epoch 2/3 done  avg_loss=0.5891 ===\n"

        with patch("backend.loop.state_machine.emit_metric", new_callable=AsyncMock) as mock_emit, \
             patch("backend.loop.state_machine.emit_log", new_callable=AsyncMock):
            asyncio.get_event_loop().run_until_complete(
                sm._tail_remote_metrics("mission-1", "dpo", 0, current_iteration=3)
            )

        mock_emit.assert_awaited_once_with("mission-1", "avg_loss", 0.5891, step=2, iteration=3)

    def test_dpo_per_step_loss_lines_emit_multiple_loss_metrics(self):
        sm = _bare_state_machine()
        sm._sandbox.tail_new_output.return_value = (
            "Step   50/59  epoch=1  loss=0.5120  4800s\n"
            "Step   51/59  epoch=1  loss=0.4981  4830s\n"
            "Step   52/59  epoch=1  loss=0.4681  5083s\n"
            "\n--- Eval at step 45 ---\n"
            "Pass rate: 88.2% (60/68)\n"
            "New best: 88.2% → saved to /tmp/best\n"
        )

        with patch("backend.loop.state_machine.emit_metric", new_callable=AsyncMock) as mock_emit, \
             patch("backend.loop.state_machine.emit_log", new_callable=AsyncMock) as mock_log:
            asyncio.get_event_loop().run_until_complete(
                sm._tail_remote_metrics("mission-1", "dpo", 0, current_iteration=1)
            )

        loss_calls = [c for c in mock_emit.await_args_list if c.args[1] == "loss"]
        assert len(loss_calls) == 3
        assert loss_calls[-1].args == ("mission-1", "loss", 0.4681)
        assert loss_calls[-1].kwargs == {"step": 52, "iteration": 1}
        pr_calls = [c for c in mock_emit.await_args_list if c.args[1] == "pass_rate"]
        assert len(pr_calls) == 1
        assert pr_calls[0].args[2] == pytest.approx(0.882)
        log_msgs = [c.args[1] for c in mock_log.await_args_list]
        assert "New best: 88.2% → saved to /tmp/best" in log_msgs
        assert "--- Eval at step 45 ---" in log_msgs

    def test_dpo_step_and_epoch_emit_distinct_metric_names(self):
        sm = _bare_state_machine()
        sm._sandbox.tail_new_output.return_value = (
            "Step   59/59  epoch=1  loss=0.5324  5303s\n"
            "\n=== Epoch 1/1 done  avg_loss=0.5971 ===\n"
        )

        with patch("backend.loop.state_machine.emit_metric", new_callable=AsyncMock) as mock_emit, \
             patch("backend.loop.state_machine.emit_log", new_callable=AsyncMock):
            asyncio.get_event_loop().run_until_complete(
                sm._tail_remote_metrics("mission-1", "dpo", 0, current_iteration=0)
            )

        loss_vals = [(c.args[2], c.kwargs["step"]) for c in mock_emit.await_args_list if c.args[1] == "loss"]
        avg_vals = [(c.args[2], c.kwargs["step"]) for c in mock_emit.await_args_list if c.args[1] == "avg_loss"]
        assert (0.5324, 59) in loss_vals
        assert (0.5971, 1) in avg_vals

    def test_grpo_task_type_does_not_match_dpo_style_loss_line(self):
        """Wrong regex for the task type must not accidentally match."""
        sm = _bare_state_machine()
        sm._sandbox.tail_new_output.return_value = "\n=== Epoch 2/3 done  avg_loss=0.5891 ===\n"

        with patch("backend.loop.state_machine.emit_metric", new_callable=AsyncMock) as mock_emit, \
             patch("backend.loop.state_machine.emit_log", new_callable=AsyncMock):
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

        with patch("backend.loop.state_machine.emit_metric", new_callable=AsyncMock) as mock_emit, \
             patch("backend.loop.state_machine.emit_log", new_callable=AsyncMock):
            asyncio.get_event_loop().run_until_complete(
                sm._tail_remote_metrics("mission-1", "grpo", 0)
            )

        assert mock_emit.await_count == 2
        names = [call.args[1] for call in mock_emit.await_args_list]
        assert "pass_rate" in names
        assert "loss" in names

    def test_collect_progress_mirrored_not_as_metric(self):
        sm = _bare_state_machine()
        sm._sandbox.tail_new_output.return_value = "  [40/66]  37 pairs  (3310s)\n"

        with patch("backend.loop.state_machine.emit_metric", new_callable=AsyncMock) as mock_metric, \
             patch("backend.loop.state_machine.emit_log", new_callable=AsyncMock) as mock_log:
            asyncio.get_event_loop().run_until_complete(
                sm._tail_remote_metrics("mission-1", "dpo", 0)
            )

        mock_metric.assert_not_awaited()
        mock_log.assert_awaited_once()
        assert "[40/66]  37 pairs  (3310s)" in mock_log.await_args.args[1]

    def test_multiple_progress_lines_all_mirrored(self):
        sm = _bare_state_machine()
        sm._sandbox.tail_new_output.return_value = (
            "  [10/66]  10 pairs  (834s)\n"
            "  [20/66]  19 pairs  (1675s)\n"
        )

        with patch("backend.loop.state_machine.emit_log", new_callable=AsyncMock) as mock_log:
            asyncio.get_event_loop().run_until_complete(
                sm._tail_remote_metrics("mission-1", "dpo", 0)
            )

        msgs = [c.args[1] for c in mock_log.await_args_list]
        assert len(msgs) == 2
        assert "10/66" in msgs[0]
        assert "20/66" in msgs[1]


# ── catch-up ───────────────────────────────────────────────────────────────────

class TestSandboxLogCatchUp:
    def test_catch_up_skips_already_emitted_log_lines(self, tmp_path, monkeypatch):
        from backend.config import settings

        mid = "mission-catchup"
        mission_dir = Path(tmp_path) / "missions" / mid
        mission_dir.mkdir(parents=True)
        tel = mission_dir / "telemetry.jsonl"
        tel.write_text(
            json.dumps({"type": "log", "name": "Loading model: gemma", "source": "sandbox"}) + "\n"
        )
        monkeypatch.setattr(settings, "data_path", str(tmp_path))

        sm = _bare_state_machine()
        sm._sandbox.read_full_log.return_value = (
            "Loading model: gemma\n"
            "Caching reference log-probs for 59 pairs...\n"
        )

        with patch("backend.loop.state_machine.emit_log", new_callable=AsyncMock) as mock_log:
            asyncio.get_event_loop().run_until_complete(
                sm._catch_up_sandbox_log_stream(mid)
            )

        msgs = [c.args[1] for c in mock_log.await_args_list]
        assert "Loading model: gemma" not in msgs
        assert "Caching reference log-probs for 59 pairs..." in msgs

    def test_mirror_skip_counts_multiset(self):
        sm = _bare_state_machine()
        # Non-consecutive so the noise filter does not collapse the two hellos.
        skip = Counter({"hello": 1})
        with patch("backend.loop.state_machine.emit_log", new_callable=AsyncMock) as mock_log:
            asyncio.get_event_loop().run_until_complete(
                sm._mirror_sandbox_log_lines(
                    "m1", "hello\nmid\nhello\nworld\n", skip_counts=skip,
                )
            )
        msgs = [c.args[1] for c in mock_log.await_args_list]
        assert msgs == ["mid", "hello", "world"]


# ── Metric Gap fallback ────────────────────────────────────────────────────────

class TestLivePassRateFallbackConsumption:
    def test_pop_consumes_and_clears_the_tracked_value(self):
        sm = _bare_state_machine()
        sm._live_pass_rate_best = {"mission-1": 0.742}

        consumed = sm._live_pass_rate_best.pop("mission-1", None)

        assert consumed == pytest.approx(0.742)
        assert "mission-1" not in sm._live_pass_rate_best

    def test_pop_returns_none_when_bare_eval_never_produced_a_reading(self):
        sm = _bare_state_machine()
        sm._live_pass_rate_best = {}

        consumed = getattr(sm, "_live_pass_rate_best", {}).pop("mission-1", None)

        assert consumed is None

    def test_fresh_iteration_clears_prior_iterations_tracked_value(self):
        sm = _bare_state_machine()
        sm._sandbox = MagicMock()
        sm._sandbox.is_alive.return_value = False
        sm._sandbox.get_log_path.return_value = "/nonexistent/path/for/this/test.log"
        sm._live_pass_rate_best = {"mission-1": 0.988}

        asyncio.get_event_loop().run_until_complete(
            sm._wait_for_sandbox("mission-1", task_type="dpo")
        )

        assert "mission-1" not in sm._live_pass_rate_best

    def test_wait_for_sandbox_threads_current_iteration_into_tail(self):
        sm = _bare_state_machine()
        sm._sandbox = MagicMock()
        sm._sandbox.is_alive.side_effect = [True, False]
        sm._sandbox.get_log_path.return_value = "/nonexistent/path/for/this/test.log"
        sm._sandbox.tail_new_output.return_value = "Pass rate: 90.0% (9/10)\n"

        with patch("backend.loop.state_machine.emit_metric", new_callable=AsyncMock) as mock_emit, \
             patch("backend.loop.state_machine.emit_log", new_callable=AsyncMock), \
             patch("asyncio.sleep", new_callable=AsyncMock):
            asyncio.get_event_loop().run_until_complete(
                sm._wait_for_sandbox("mission-1", task_type="dpo", current_iteration=4)
            )

        mock_emit.assert_awaited_once_with("mission-1", "pass_rate", 0.9, step=0, iteration=4)


# ── distill: live tail is static-scale, goal metric is blended ───────────────

class TestDistillLiveTailScale:
    def test_distill_live_pass_rate_emitted_under_static_name(self):
        sm = _bare_state_machine()
        sm._sandbox.tail_new_output.return_value = "Pass rate: 81.8% (9/11)\n"

        with patch("backend.loop.state_machine.emit_metric", new_callable=AsyncMock) as mock_emit, \
             patch("backend.loop.state_machine.emit_log", new_callable=AsyncMock):
            asyncio.get_event_loop().run_until_complete(
                sm._tail_remote_metrics("mission-1", "distill", 0)
            )

        mock_emit.assert_awaited_once_with(
            "mission-1", "pass_rate_static_live", 0.818, step=0, iteration=0
        )

    def test_opd_live_pass_rate_emitted_under_static_name(self):
        sm = _bare_state_machine()
        sm._sandbox.tail_new_output.return_value = "Pass rate: 81.8% (9/11)\n"

        with patch("backend.loop.state_machine.emit_metric", new_callable=AsyncMock) as mock_emit, \
             patch("backend.loop.state_machine.emit_log", new_callable=AsyncMock):
            asyncio.get_event_loop().run_until_complete(
                sm._tail_remote_metrics("mission-1", "opd", 0)
            )

        mock_emit.assert_awaited_once_with(
            "mission-1", "pass_rate_static_live", 0.818, step=0, iteration=0
        )

    def test_dpo_live_pass_rate_keeps_the_plain_name(self):
        sm = _bare_state_machine()
        sm._sandbox.tail_new_output.return_value = "Pass rate: 81.8% (9/11)\n"

        with patch("backend.loop.state_machine.emit_metric", new_callable=AsyncMock) as mock_emit, \
             patch("backend.loop.state_machine.emit_log", new_callable=AsyncMock):
            asyncio.get_event_loop().run_until_complete(
                sm._tail_remote_metrics("mission-1", "dpo", 0)
            )

        mock_emit.assert_awaited_once_with(
            "mission-1", "pass_rate", 0.818, step=0, iteration=0
        )
