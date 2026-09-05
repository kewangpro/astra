"""Unit tests for LoopStateMachine._run_bare_eval — the post-training
authoritative pass_rate check for dpo/grpo missions (analogous to
_run_goal_metric_eval for RL missions, which can't be used here since
adapters are .safetensors, not a Gym-rollout-compatible SB3/actor_critic
checkpoint)."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from backend.loop.state_machine import LoopStateMachine


def _bare_state_machine() -> LoopStateMachine:
    sm = LoopStateMachine.__new__(LoopStateMachine)
    return sm


def _plan(**overrides) -> dict:
    base = {"task_type": "grpo", "hyperparameters": {}}
    base.update(overrides)
    return base


class TestRunBareEval:
    def test_returns_none_when_sandbox_host_not_configured(self):
        sm = _bare_state_machine()
        with patch("backend.loop.state_machine.settings.sandbox_host", ""):
            result = sm._run_bare_eval("mission-abc12345", _plan(), 5)
        assert result is None

    def test_parses_pass_rate_from_ssh_output(self):
        sm = _bare_state_machine()
        with patch("backend.loop.state_machine.settings.sandbox_host", "mac-mini.local"), \
             patch("backend.loop.state_machine.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                stdout="Pass rate: 82.5% (55/66)  [11.3 min]\n", stderr="",
            )
            result = sm._run_bare_eval("mission-abc12345", _plan(), 5)

        assert result == pytest.approx(0.825)

    def test_parses_static_skill_split_metric_when_present(self):
        """bare_eval.py switches to a static/dynamic-MCP split report whenever the
        case pool has MCP cases (true for every real dpo/grpo/distill mission) —
        it then prints "Static-skill routing:", never "Pass rate:", so this must
        not silently fall through to "no parseable pass rate" (real incident:
        mission 551839b7's bare_eval ran its full ~19 min and produced a genuine
        result that _run_bare_eval failed to parse both times it ran)."""
        sm = _bare_state_machine()
        stdout = (
            "Model:    mlx-community/gemma-3-12b-it-4bit\n"
            "Static-skill routing: 10/11 (90.9%)  ← fine-tune target metric  [19.3 min]\n"
            "Dynamic MCP routing:  0/7 (0.0%)  ← needs MCP skills in the prompt (separate project)\n"
            "Blended (all cases):  68/78 (87.2%)  ← historical bare-oracle number, not the selector\n"
        )
        with patch("backend.loop.state_machine.settings.sandbox_host", "mac-mini.local"), \
             patch("backend.loop.state_machine.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(stdout=stdout, stderr="")
            result = sm._run_bare_eval("mission-abc12345", _plan(task_type="distill"), 2)

        assert result == pytest.approx(0.909)

    def test_falls_back_to_blended_pass_rate_when_no_static_split(self):
        """A non-split bare_eval run (e.g. --ids scoped to pure static cases) has
        no "Static-skill routing:" line at all — must still parse the plain
        "Pass rate:" line rather than treating its absence as a hard requirement."""
        sm = _bare_state_machine()
        with patch("backend.loop.state_machine.settings.sandbox_host", "mac-mini.local"), \
             patch("backend.loop.state_machine.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                stdout="Pass rate: 100.0% (3/3)  [2.1 min]\nAll routing cases pass ✓\n", stderr="",
            )
            result = sm._run_bare_eval("mission-abc12345", _plan(), 5)

        assert result == pytest.approx(1.0)

    def test_ssh_command_uses_finetune_dir_and_astra_adapter_path(self):
        sm = _bare_state_machine()
        with patch("backend.loop.state_machine.settings.sandbox_host", "mac-mini.local"), \
             patch("backend.loop.state_machine.subprocess.run") as mock_run:
            # First call: the best/ existence check (see _resolve_adapter_or_bare).
            # Second call: the actual bare_eval.py invocation.
            mock_run.side_effect = [
                MagicMock(stdout="yes\n", stderr=""),
                MagicMock(stdout="Pass rate: 80.0% (16/20)\n", stderr=""),
            ]
            sm._run_bare_eval("mission-abc12345", _plan(), 5)

        call_args = mock_run.call_args_list[-1].args[0]
        assert call_args[0] == "ssh"
        assert call_args[1] == "mac-mini.local"
        cmd = call_args[2]
        assert "cd /Users/kewang/finetune" in cmd
        assert "bare_eval.py" in cmd
        # Iteration-scoped (see finetune_checkpoint_dir()'s docstring) — a real
        # incident showed a mission-level-only path let every iteration's
        # --save-dir write clobber whatever the previous iteration had left
        # there, so bare_eval could silently score the wrong iteration.
        assert "--adapter adapters/astra_mission-_iter5/best" in cmd
        assert "--prompt-template backend/prompts/conductor_min.md" in cmd

    def test_returns_none_when_no_pass_rate_line_found(self):
        sm = _bare_state_machine()
        with patch("backend.loop.state_machine.settings.sandbox_host", "mac-mini.local"), \
             patch("backend.loop.state_machine.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(stdout="Traceback...\nSomeError\n", stderr="")
            result = sm._run_bare_eval("mission-abc12345", _plan(), 5)
        assert result is None

    def test_returns_none_on_subprocess_exception(self):
        sm = _bare_state_machine()
        with patch("backend.loop.state_machine.settings.sandbox_host", "mac-mini.local"), \
             patch("backend.loop.state_machine.subprocess.run", side_effect=RuntimeError("ssh timeout")):
            result = sm._run_bare_eval("mission-abc12345", _plan(), 5)
        assert result is None

    def test_uses_dpo_recipe_finetune_dir_for_dpo_task_type(self):
        sm = _bare_state_machine()
        with patch("backend.loop.state_machine.settings.sandbox_host", "mac-mini.local"), \
             patch("backend.loop.state_machine.subprocess.run") as mock_run:
            mock_run.side_effect = [
                MagicMock(stdout="yes\n", stderr=""),
                MagicMock(stdout="Pass rate: 70.0% (14/20)\n", stderr=""),
            ]
            sm._run_bare_eval("mission-xyz98765", _plan(task_type="dpo"), 12)

        cmd = mock_run.call_args_list[-1].args[0][2]
        assert "cd /Users/kewang/finetune" in cmd
        assert "--adapter adapters/astra_mission-_iter12/best" in cmd

    def test_falls_back_to_bare_dir_when_best_checkpoint_missing(self):
        """dpo_train.py only writes <save-dir>/best/ when it tracks a
        best-during-training checkpoint that beats the final epoch's own
        output; when the final epoch is itself the best, no best/ subdir is
        written at all (confirmed ~50% of runs via filesystem audit). Real
        incident: hardcoding /best unconditionally crashed dpo_train.py
        instantly with FileNotFoundError the first time a best/-less
        iteration became the mission's new best — silently looping the
        mission for ~7h/558 iterations with no checkpoint ever produced."""
        sm = _bare_state_machine()
        with patch("backend.loop.state_machine.settings.sandbox_host", "mac-mini.local"), \
             patch("backend.loop.state_machine.subprocess.run") as mock_run:
            mock_run.side_effect = [
                MagicMock(stdout="no\n", stderr=""),
                MagicMock(stdout="Pass rate: 83.3% (55/66)\n", stderr=""),
            ]
            result = sm._run_bare_eval("mission-abc12345", _plan(task_type="dpo"), 134)

        assert result == pytest.approx(0.833)
        cmd = mock_run.call_args_list[-1].args[0][2]
        assert "--adapter adapters/astra_mission-_iter134" in cmd
        assert "--adapter adapters/astra_mission-_iter134/best" not in cmd


class TestDistillHeldOutMetric:
    """LoopStateMachine._distill_held_out_metric — distill's goal metric comes
    from the training log's held-out "Best pass rate during training:" line, NOT
    a full-set bare_eval.py run (which leaks training cases and reports a
    different number than the floor's held-out "Baseline:" line — that mismatch
    doom-looped mission b790c69d for ~9h)."""

    def _sm_with_log(self, tmp_path, contents):
        sm = _bare_state_machine()
        log = tmp_path / "sandbox.log"
        log.write_text(contents)
        sm._sandbox = MagicMock()
        sm._sandbox.get_log_path.return_value = str(log)
        return sm

    def test_parses_best_pass_rate_during_training(self, tmp_path):
        sm = self._sm_with_log(tmp_path,
            "Baseline: 87.5% (7/8)\n"
            "Pass rate: 87.5% (7/8)\n"
            "Pass rate: 100.0% (8/8)\n"
            "=== Final Eval ===\n"
            "Pass rate: 87.5% (7/8)\n"
            "Best pass rate during training: 100.0%\n"
        )
        assert sm._distill_held_out_metric("m-123") == pytest.approx(1.0)

    def test_falls_back_to_last_pass_rate_line(self, tmp_path):
        # crashed before the summary line — use the final in-log held-out eval
        sm = self._sm_with_log(tmp_path,
            "Baseline: 87.5% (7/8)\nPass rate: 87.5% (7/8)\nPass rate: 75.0% (6/8)\n"
        )
        assert sm._distill_held_out_metric("m-123") == pytest.approx(0.75)

    def test_returns_none_when_log_missing(self, tmp_path):
        sm = _bare_state_machine()
        sm._sandbox = MagicMock()
        sm._sandbox.get_log_path.return_value = str(tmp_path / "nope.log")
        assert sm._distill_held_out_metric("m-123") is None

    def test_returns_none_when_no_pass_rate_at_all(self, tmp_path):
        sm = self._sm_with_log(tmp_path, "Loading model...\nTraceback\n")
        assert sm._distill_held_out_metric("m-123") is None


class TestResolveAdapterOrBare:
    """LoopStateMachine._resolve_adapter_or_bare — the existence-checked
    resolution used by both _run_bare_eval and the dpo/grpo warm-start
    chaining decision. See finetune_checkpoint_dir_relative()'s docstring for
    why an unconditional /best append is unsafe."""

    def test_appends_best_when_it_exists_remotely(self):
        sm = _bare_state_machine()
        with patch("backend.loop.state_machine.settings.sandbox_host", "mac-mini.local"), \
             patch("backend.loop.state_machine.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(stdout="yes\n", stderr="")
            result = sm._resolve_adapter_or_bare("/Users/kewang/finetune", "adapters/astra_abc12345_iter7")
        assert result == "adapters/astra_abc12345_iter7/best"

    def test_falls_back_to_bare_dir_when_best_missing_remotely(self):
        sm = _bare_state_machine()
        with patch("backend.loop.state_machine.settings.sandbox_host", "mac-mini.local"), \
             patch("backend.loop.state_machine.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(stdout="no\n", stderr="")
            result = sm._resolve_adapter_or_bare("/Users/kewang/finetune", "adapters/astra_abc12345_iter7")
        assert result == "adapters/astra_abc12345_iter7"

    def test_falls_back_to_bare_dir_on_ssh_exception(self):
        sm = _bare_state_machine()
        with patch("backend.loop.state_machine.settings.sandbox_host", "mac-mini.local"), \
             patch("backend.loop.state_machine.subprocess.run", side_effect=RuntimeError("ssh timeout")):
            result = sm._resolve_adapter_or_bare("/Users/kewang/finetune", "adapters/astra_abc12345_iter7")
        assert result == "adapters/astra_abc12345_iter7"

    def test_falls_back_to_bare_dir_when_sandbox_host_not_configured(self):
        sm = _bare_state_machine()
        with patch("backend.loop.state_machine.settings.sandbox_host", ""):
            result = sm._resolve_adapter_or_bare("/Users/kewang/finetune", "adapters/astra_abc12345_iter7")
        assert result == "adapters/astra_abc12345_iter7"

    def test_checks_for_adapters_safetensors_specifically_not_just_the_dir(self):
        """A best/ dir could theoretically exist without weights written into
        it yet (partial write, race). Check for the actual weights file dpo_
        train.py's own loader requires, matching bare_eval's own error
        message ('is missing adapters.safetensors and/or adapter_config.json')."""
        sm = _bare_state_machine()
        with patch("backend.loop.state_machine.settings.sandbox_host", "mac-mini.local"), \
             patch("backend.loop.state_machine.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(stdout="yes\n", stderr="")
            sm._resolve_adapter_or_bare("/Users/kewang/finetune", "adapters/astra_abc12345_iter7")
        cmd = mock_run.call_args.args[0][2]
        assert "adapters/astra_abc12345_iter7/best/adapters.safetensors" in cmd
