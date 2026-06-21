"""Unit tests for SandboxManager.recover() and launch() lifecycle fixes."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from backend.sandbox.base import SandboxConfig, SandboxStatus
from backend.sandbox.subprocess_sandbox import SubprocessSandbox


def _make_manager():
    """Return a fresh SandboxManager with mocked backend detection."""
    with patch("backend.sandbox.manager._detect_backend", return_value="subprocess"):
        from backend.sandbox.manager import SandboxManager
        return SandboxManager()


def _config(mission_id: str = "test-mission") -> SandboxConfig:
    return SandboxConfig(
        mission_id=mission_id,
        script_path="/tmp/fake.py",
        data_dir="/tmp/fake_data",
    )


# ── SandboxManager.recover ────────────────────────────────────────────────────

class TestRecover:
    def test_reattach_sets_reattach_pid(self):
        """recover() must store the pid on the sandbox so terminate() can kill it."""
        mgr = _make_manager()
        with patch("backend.sandbox.manager.psutil.pid_exists", return_value=True), \
             patch("backend.sandbox.manager.os.path.abspath", return_value="/tmp/fake_data"):
            outcome = mgr.recover("test-mission", subprocess_pid=12345, container_id=None)

        assert outcome == "reattached"
        sandbox = mgr._sandboxes.get("test-mission")
        assert sandbox is not None
        assert sandbox._reattach_pid == 12345

    def test_reattach_returns_dead_when_pid_gone(self):
        mgr = _make_manager()
        with patch("backend.sandbox.manager.psutil.pid_exists", return_value=False):
            outcome = mgr.recover("test-mission", subprocess_pid=12345, container_id=None)

        assert outcome == "dead"
        assert "test-mission" not in mgr._sandboxes

    def test_recover_no_pid_no_container_returns_dead(self):
        mgr = _make_manager()
        outcome = mgr.recover("test-mission", subprocess_pid=None, container_id=None)
        assert outcome == "dead"


# ── SandboxManager.launch — kills existing sandbox first ─────────────────────

class TestLaunchKillsExisting:
    def test_launch_terminates_alive_existing_sandbox(self, tmp_path):
        """If a sandbox is already registered and alive, launch() must kill it first."""
        mgr = _make_manager()

        # Plant a fake live sandbox
        stale = MagicMock(spec=SubprocessSandbox)
        stale.is_alive.return_value = True
        mgr._sandboxes["test-mission"] = stale

        # Mock the actual launch so we don't spawn a real process
        new_sandbox = MagicMock(spec=SubprocessSandbox)
        new_sandbox.get_sandbox_id.return_value = "9999"

        script = tmp_path / "train.py"
        script.write_text("pass")

        with patch("backend.sandbox.manager.SubprocessSandbox", return_value=new_sandbox), \
             patch("backend.sandbox.manager.os.makedirs"):
            mgr.launch("test-mission", str(script))

        stale.terminate.assert_called_once()

    def test_launch_skips_terminate_when_existing_sandbox_dead(self, tmp_path):
        """Dead sandbox is evicted without calling terminate()."""
        mgr = _make_manager()

        stale = MagicMock(spec=SubprocessSandbox)
        stale.is_alive.return_value = False
        mgr._sandboxes["test-mission"] = stale

        new_sandbox = MagicMock(spec=SubprocessSandbox)
        new_sandbox.get_sandbox_id.return_value = "9999"

        script = tmp_path / "train.py"
        script.write_text("pass")

        with patch("backend.sandbox.manager.SubprocessSandbox", return_value=new_sandbox), \
             patch("backend.sandbox.manager.os.makedirs"):
            mgr.launch("test-mission", str(script))

        stale.terminate.assert_not_called()
