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

    def test_container_reattach_registers_sandbox_for_liveness_polling(self):
        """A reattached container must be registered in self._sandboxes with
        its container_id set — otherwise the resumed loop's is_alive() lookup
        finds nothing and immediately (and incorrectly) treats a genuinely
        running container as dead."""
        mgr = _make_manager()
        with patch("backend.sandbox.container_sandbox.ContainerSandbox.is_container_alive", return_value=True):
            outcome = mgr.recover("test-mission", subprocess_pid=None, container_id="abc123def456")
            assert mgr.is_alive("test-mission") is True

        assert outcome == "reattached"
        sandbox = mgr._sandboxes.get("test-mission")
        assert sandbox is not None
        assert sandbox._container_id == "abc123def456"

    def test_container_recover_returns_dead_when_gone(self):
        mgr = _make_manager()
        with patch("backend.sandbox.container_sandbox.ContainerSandbox.is_container_alive", return_value=False):
            outcome = mgr.recover("test-mission", subprocess_pid=None, container_id="abc123def456")

        assert outcome == "dead"
        assert "test-mission" not in mgr._sandboxes


# ── SandboxManager.recover — remote_pid (SSH/dpo/grpo) ────────────────────────

class TestRecoverRemotePid:
    def test_reattaches_when_remote_process_alive(self):
        mgr = _make_manager()
        with patch("backend.sandbox.manager.settings.sandbox_host", "mac-mini.local"), \
             patch("backend.sandbox.manager.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(stdout="alive\n")
            outcome = mgr.recover("test-mission", subprocess_pid=None, container_id=None, remote_pid=14516)

        assert outcome == "reattached"
        sandbox = mgr._sandboxes.get("test-mission")
        assert sandbox is not None
        assert sandbox._remote_pid == 14516

    def test_returns_dead_when_remote_process_gone(self):
        mgr = _make_manager()
        with patch("backend.sandbox.manager.settings.sandbox_host", "mac-mini.local"), \
             patch("backend.sandbox.manager.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(stdout="dead\n")
            outcome = mgr.recover("test-mission", subprocess_pid=None, container_id=None, remote_pid=14516)

        assert outcome == "dead"
        assert "test-mission" not in mgr._sandboxes

    def test_returns_dead_when_sandbox_host_unconfigured(self):
        """Can't verify a remote pid with no host configured — must not assume alive."""
        mgr = _make_manager()
        with patch("backend.sandbox.manager.settings.sandbox_host", ""):
            outcome = mgr.recover("test-mission", subprocess_pid=None, container_id=None, remote_pid=14516)
        assert outcome == "dead"

    def test_reattach_syncs_tail_offset_to_current_remote_log_size(self):
        """Reattaching mid-run must not re-tail the entire historical remote
        log as if it were new output — that would re-emit every already-seen
        pass_rate/loss event in one burst. The tail offset must be seeded to
        the remote log's current size before any tail_new_output() call."""
        mgr = _make_manager()
        with patch("backend.sandbox.manager.settings.sandbox_host", "mac-mini.local"), \
             patch("backend.sandbox.manager.subprocess.run") as mock_run:
            mock_run.side_effect = [
                MagicMock(stdout="alive\n"),    # kill -0 liveness check
                MagicMock(stdout="4096\n"),     # wc -c remote log size
            ]
            mgr.recover("test-mission", subprocess_pid=None, container_id=None, remote_pid=14516)

        sandbox = mgr._sandboxes.get("test-mission")
        assert sandbox._tail_offset == 4096

    def test_reattached_sandbox_terminate_kills_the_real_remote_pid(self):
        """Confirms the reattached SSHSandbox's terminate() targets the actual
        training pid (post os.execv fix) — not an orphan-prone wrapper pid."""
        mgr = _make_manager()
        with patch("backend.sandbox.manager.settings.sandbox_host", "mac-mini.local"), \
             patch("backend.sandbox.manager.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(stdout="alive\n")
            mgr.recover("test-mission", subprocess_pid=None, container_id=None, remote_pid=14516)

        with patch("backend.sandbox.ssh_sandbox.subprocess.run") as mock_ssh_run:
            mgr.terminate("test-mission")

        kill_call = mock_ssh_run.call_args_list[0].args[0]
        assert "kill -TERM 14516" in kill_call[3]


# ── SandboxManager.launch — kills existing sandbox first ─────────────────────

class TestLaunchKillsExisting:
    def test_launch_terminates_alive_existing_sandbox(self, tmp_path):
        """If a sandbox is already registered and alive, launch() must kill it first."""
        mgr = _make_manager()

        # Plant a fake live sandbox
        stale = MagicMock(spec=SubprocessSandbox)
        stale.is_alive.return_value = True
        stale.status = SandboxStatus.STOPPED  # terminate() succeeded
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

    def test_launch_aborts_and_restores_tracking_when_termination_unconfirmed(self, tmp_path):
        """If terminate() can't confirm the old sandbox actually died (e.g. the
        SSH host was unreachable — SSHSandbox.terminate() then leaves status
        untouched, not STOPPED), launch() must not silently discard tracking
        and proceed anyway — that would risk two processes running for the
        same mission, and orphan the untracked one exactly like a real
        incident where SandboxManager.terminate() discarding regardless of
        outcome lost track of a still-possibly-alive remote_pid."""
        mgr = _make_manager()

        stale = MagicMock(spec=SubprocessSandbox)
        stale.is_alive.return_value = True
        stale.status = SandboxStatus.RUNNING  # terminate() could not confirm death
        mgr._sandboxes["test-mission"] = stale

        script = tmp_path / "train.py"
        script.write_text("pass")

        with patch("backend.sandbox.manager.SubprocessSandbox"), \
             patch("backend.sandbox.manager.os.makedirs"):
            with pytest.raises(RuntimeError, match="Could not confirm termination"):
                mgr.launch("test-mission", str(script))

        stale.terminate.assert_called_once()
        # Restored, not orphaned — a future launch/reattach can still find it.
        assert mgr._sandboxes["test-mission"] is stale


# ── SandboxManager.terminate ──────────────────────────────────────────────────

class TestManagerTerminate:
    def test_terminate_discards_tracking_when_confirmed_stopped(self):
        mgr = _make_manager()
        sandbox = MagicMock(spec=SubprocessSandbox)
        sandbox.status = SandboxStatus.STOPPED
        mgr._sandboxes["test-mission"] = sandbox

        mgr.terminate("test-mission")

        sandbox.terminate.assert_called_once()
        assert "test-mission" not in mgr._sandboxes

    def test_terminate_keeps_tracking_when_kill_unconfirmed(self):
        """A real incident: SSHSandbox.terminate() couldn't reach the host
        (network blip), logged a warning, and left status untouched instead
        of claiming STOPPED. SandboxManager.terminate() previously discarded
        the sandbox from tracking regardless, orphaning a possibly-still-alive
        remote_pid — the same class of bug as an earlier is_alive() gap, just
        one layer up the call stack. Must now preserve tracking instead."""
        mgr = _make_manager()
        sandbox = MagicMock(spec=SubprocessSandbox)
        sandbox.status = SandboxStatus.RUNNING  # kill could not be confirmed
        mgr._sandboxes["test-mission"] = sandbox

        mgr.terminate("test-mission")

        sandbox.terminate.assert_called_once()
        assert mgr._sandboxes["test-mission"] is sandbox


# ── SandboxManager.launch — fine-tune task types pinned to Mac Mini ──────────

class TestFinetuneRemoteDispatch:
    def test_dpo_raises_when_sandbox_host_not_configured(self, tmp_path):
        """dpo/grpo must hard-fail, not silently fall back to a local backend."""
        mgr = _make_manager()
        script = tmp_path / "train.py"
        script.write_text("pass")

        with patch("backend.sandbox.manager.settings.sandbox_host", ""):
            with pytest.raises(RuntimeError, match="sandbox_host"):
                mgr.launch("test-mission", str(script), task_type="dpo")

    def test_grpo_raises_when_sandbox_host_not_configured(self, tmp_path):
        mgr = _make_manager()
        script = tmp_path / "train.py"
        script.write_text("pass")

        with patch("backend.sandbox.manager.settings.sandbox_host", ""):
            with pytest.raises(RuntimeError, match="sandbox_host"):
                mgr.launch("test-mission", str(script), task_type="grpo")

    def test_dpo_forces_ssh_backend_when_sandbox_host_configured(self, tmp_path):
        mgr = _make_manager()
        script = tmp_path / "train.py"
        script.write_text("pass")

        new_sandbox = MagicMock()
        new_sandbox.get_sandbox_id.return_value = None

        with patch("backend.sandbox.manager.settings.sandbox_host", "mac-mini.local"), \
             patch("backend.sandbox.manager.settings.sandbox_data_path", "/tmp/astra"), \
             patch("backend.sandbox.manager.SSHSandbox", return_value=new_sandbox) as mock_ssh, \
             patch("backend.sandbox.manager.os.makedirs"):
            mgr.launch("test-mission", str(script), task_type="dpo")

        mock_ssh.assert_called_once()
        new_sandbox.launch.assert_called_once()

    def test_rl_task_type_unaffected_by_finetune_check(self, tmp_path):
        """Non-finetune task types must not trigger the sandbox_host requirement."""
        mgr = _make_manager()
        script = tmp_path / "train.py"
        script.write_text("pass")

        new_sandbox = MagicMock()
        new_sandbox.get_sandbox_id.return_value = "9999"

        with patch("backend.sandbox.manager.settings.sandbox_host", ""), \
             patch("backend.sandbox.manager.SubprocessSandbox", return_value=new_sandbox), \
             patch("backend.sandbox.manager.os.makedirs"):
            mgr.launch("test-mission", str(script), task_type="rl")   # must not raise

    def test_tail_new_output_returns_none_when_no_sandbox(self):
        mgr = _make_manager()
        assert mgr.tail_new_output("nonexistent-mission") is None

    def test_tail_new_output_returns_none_when_backend_unsupported(self):
        """SubprocessSandbox has no tail_new_output — must not raise."""
        mgr = _make_manager()
        mgr._sandboxes["test-mission"] = MagicMock(spec=SubprocessSandbox)
        assert mgr.tail_new_output("test-mission") is None

    def test_tail_new_output_delegates_to_ssh_sandbox(self):
        mgr = _make_manager()
        ssh_sandbox = MagicMock()
        ssh_sandbox.tail_new_output.return_value = "Pass rate: 80.0% (16/20)\n"
        mgr._sandboxes["test-mission"] = ssh_sandbox

        result = mgr.tail_new_output("test-mission")

        assert result == "Pass rate: 80.0% (16/20)\n"
        ssh_sandbox.tail_new_output.assert_called_once()

    def test_get_sandbox_id_returns_none_when_no_sandbox(self):
        mgr = _make_manager()
        assert mgr.get_sandbox_id("nonexistent-mission") is None

    def test_get_sandbox_id_delegates_to_sandbox(self):
        """Used for display/logging (e.g. 'remote_pid=14515' for SSH-dispatched
        missions), not for the Mission Store's subprocess_pid field."""
        mgr = _make_manager()
        ssh_sandbox = MagicMock()
        ssh_sandbox.get_sandbox_id.return_value = "14515"
        mgr._sandboxes["test-mission"] = ssh_sandbox

        result = mgr.get_sandbox_id("test-mission")

        assert result == "14515"
        ssh_sandbox.get_sandbox_id.assert_called_once()

    def test_launch_with_no_task_type_unaffected(self, tmp_path):
        """Backward compatibility: omitting task_type must not raise or change behavior."""
        mgr = _make_manager()
        script = tmp_path / "train.py"
        script.write_text("pass")

        new_sandbox = MagicMock()
        new_sandbox.get_sandbox_id.return_value = "9999"

        with patch("backend.sandbox.manager.settings.sandbox_host", ""), \
             patch("backend.sandbox.manager.SubprocessSandbox", return_value=new_sandbox), \
             patch("backend.sandbox.manager.os.makedirs"):
            mgr.launch("test-mission", str(script))   # no task_type kwarg at all
