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
            result = sm._run_bare_eval("mission-abc12345", _plan())
        assert result is None

    def test_parses_pass_rate_from_ssh_output(self):
        sm = _bare_state_machine()
        with patch("backend.loop.state_machine.settings.sandbox_host", "mac-mini.local"), \
             patch("backend.loop.state_machine.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                stdout="Pass rate: 82.5% (55/66)  [11.3 min]\n", stderr="",
            )
            result = sm._run_bare_eval("mission-abc12345", _plan())

        assert result == pytest.approx(0.825)

    def test_ssh_command_uses_finetune_dir_and_astra_adapter_path(self):
        sm = _bare_state_machine()
        with patch("backend.loop.state_machine.settings.sandbox_host", "mac-mini.local"), \
             patch("backend.loop.state_machine.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(stdout="Pass rate: 80.0% (16/20)\n", stderr="")
            sm._run_bare_eval("mission-abc12345", _plan())

        call_args = mock_run.call_args_list[0].args[0]
        assert call_args[0] == "ssh"
        assert call_args[1] == "mac-mini.local"
        cmd = call_args[2]
        assert "cd /Users/kewang/finetune" in cmd
        assert "bare_eval.py" in cmd
        assert "--adapter adapters/astra_mission-" in cmd
        assert "--prompt-template backend/prompts/conductor_min.md" in cmd

    def test_returns_none_when_no_pass_rate_line_found(self):
        sm = _bare_state_machine()
        with patch("backend.loop.state_machine.settings.sandbox_host", "mac-mini.local"), \
             patch("backend.loop.state_machine.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(stdout="Traceback...\nSomeError\n", stderr="")
            result = sm._run_bare_eval("mission-abc12345", _plan())
        assert result is None

    def test_returns_none_on_subprocess_exception(self):
        sm = _bare_state_machine()
        with patch("backend.loop.state_machine.settings.sandbox_host", "mac-mini.local"), \
             patch("backend.loop.state_machine.subprocess.run", side_effect=RuntimeError("ssh timeout")):
            result = sm._run_bare_eval("mission-abc12345", _plan())
        assert result is None

    def test_uses_dpo_recipe_finetune_dir_for_dpo_task_type(self):
        sm = _bare_state_machine()
        with patch("backend.loop.state_machine.settings.sandbox_host", "mac-mini.local"), \
             patch("backend.loop.state_machine.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(stdout="Pass rate: 70.0% (14/20)\n", stderr="")
            sm._run_bare_eval("mission-xyz98765", _plan(task_type="dpo"))

        cmd = mock_run.call_args_list[0].args[0][2]
        assert "cd /Users/kewang/finetune" in cmd
        assert "--adapter adapters/astra_mission-" in cmd
