"""Unit tests for SubprocessSandbox in sandbox/subprocess_sandbox.py."""
from __future__ import annotations

import resource
from unittest.mock import MagicMock, patch

import pytest

from backend.sandbox.base import SandboxConfig, SandboxStatus
from backend.sandbox.subprocess_sandbox import SubprocessSandbox, _apply_resource_limits


# ── _apply_resource_limits ────────────────────────────────────────────────────

class TestApplyResourceLimits:
    def test_sets_rlimit_as(self):
        with patch("backend.sandbox.subprocess_sandbox.resource") as mock_resource:
            mock_resource.RLIMIT_AS = resource.RLIMIT_AS
            mock_resource.RLIM_INFINITY = resource.RLIM_INFINITY
            mock_resource.getrlimit.return_value = (resource.RLIM_INFINITY, resource.RLIM_INFINITY)
            _apply_resource_limits(memory_limit_gb=4.0, cpu_count=2)
            expected_bytes = int(4.0 * 1024 ** 3)
            mock_resource.setrlimit.assert_called_once_with(
                resource.RLIMIT_AS, (expected_bytes, expected_bytes)
            )

    def test_clamps_to_hard_limit_when_too_high(self):
        hard = int(2.0 * 1024 ** 3)
        with patch("backend.sandbox.subprocess_sandbox.resource") as mock_resource:
            mock_resource.RLIMIT_AS = resource.RLIMIT_AS
            mock_resource.RLIM_INFINITY = resource.RLIM_INFINITY
            # Simulate ValueError on first setrlimit, then getrlimit returns finite hard limit
            mock_resource.setrlimit.side_effect = [ValueError, None]
            mock_resource.getrlimit.return_value = (hard, hard)
            _apply_resource_limits(memory_limit_gb=8.0, cpu_count=2)
            # Second call should clamp to hard limit
            assert mock_resource.setrlimit.call_count == 2
            mock_resource.setrlimit.assert_called_with(resource.RLIMIT_AS, (hard, hard))


# ── SubprocessSandbox.is_pid_alive ────────────────────────────────────────────

class TestIsPidAlive:
    def _sandbox(self) -> SubprocessSandbox:
        config = SandboxConfig(
            mission_id="test-mission",
            script_path="/tmp/fake_script.py",
            data_dir="/tmp/fake_data",
        )
        return SubprocessSandbox(config)

    def test_existing_pid_returns_true(self):
        import os
        sandbox = self._sandbox()
        assert sandbox.is_pid_alive(os.getpid()) is True

    def test_nonexistent_pid_returns_false(self):
        sandbox = self._sandbox()
        # PID 99999999 is very unlikely to exist
        assert sandbox.is_pid_alive(99999999) is False


# ── SubprocessSandbox.is_alive ────────────────────────────────────────────────

class TestIsAlive:
    def _sandbox(self) -> SubprocessSandbox:
        config = SandboxConfig(
            mission_id="test-mission",
            script_path="/tmp/fake_script.py",
            data_dir="/tmp/fake_data",
        )
        return SubprocessSandbox(config)

    def test_false_when_no_process_attached(self):
        sandbox = self._sandbox()
        assert sandbox.is_alive() is False

    def test_true_when_process_still_running(self):
        sandbox = self._sandbox()
        mock_process = MagicMock()
        mock_process.poll.return_value = None   # still running
        sandbox._process = mock_process
        assert sandbox.is_alive() is True

    def test_false_when_process_has_exited(self):
        sandbox = self._sandbox()
        mock_process = MagicMock()
        mock_process.poll.return_value = 0      # exited with code 0
        sandbox._process = mock_process
        assert sandbox.is_alive() is False


# ── SubprocessSandbox.get_sandbox_id ─────────────────────────────────────────

class TestGetSandboxId:
    def _sandbox(self) -> SubprocessSandbox:
        config = SandboxConfig(
            mission_id="test-mission",
            script_path="/tmp/fake_script.py",
            data_dir="/tmp/fake_data",
        )
        return SubprocessSandbox(config)

    def test_returns_none_without_process(self):
        sandbox = self._sandbox()
        assert sandbox.get_sandbox_id() is None

    def test_returns_pid_as_string(self):
        sandbox = self._sandbox()
        mock_process = MagicMock()
        mock_process.pid = 12345
        sandbox._process = mock_process
        assert sandbox.get_sandbox_id() == "12345"


# ── SubprocessSandbox.terminate ───────────────────────────────────────────────

class TestTerminate:
    def _sandbox(self) -> SubprocessSandbox:
        config = SandboxConfig(
            mission_id="test-mission",
            script_path="/tmp/fake_script.py",
            data_dir="/tmp/fake_data",
        )
        return SubprocessSandbox(config)

    def test_terminate_sets_status_stopped(self):
        sandbox = self._sandbox()
        mock_process = MagicMock()
        mock_process.poll.return_value = None
        sandbox._process = mock_process
        sandbox.terminate()
        assert sandbox.status == SandboxStatus.STOPPED

    def test_terminate_calls_process_terminate(self):
        sandbox = self._sandbox()
        mock_process = MagicMock()
        mock_process.poll.return_value = None
        sandbox._process = mock_process
        sandbox.terminate()
        mock_process.terminate.assert_called_once()

    def test_terminate_with_no_process_does_not_raise(self):
        sandbox = self._sandbox()
        sandbox.terminate()   # should not raise
        assert sandbox.status == SandboxStatus.STOPPED

    def test_terminate_kills_when_timeout(self):
        import subprocess
        sandbox = self._sandbox()
        mock_process = MagicMock()
        mock_process.poll.return_value = None
        mock_process.wait.side_effect = subprocess.TimeoutExpired(cmd="fake", timeout=10)
        sandbox._process = mock_process
        sandbox.terminate()
        mock_process.kill.assert_called_once()
