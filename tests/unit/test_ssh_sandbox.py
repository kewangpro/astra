"""Unit tests for SSHSandbox in sandbox/ssh_sandbox.py."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from backend.sandbox.base import SandboxConfig, SandboxStatus
from backend.sandbox.ssh_sandbox import SSHSandbox


def _sandbox(tmp_path) -> SSHSandbox:
    config = SandboxConfig(
        mission_id="test-mission",
        script_path="/tmp/fake_train.py",
        data_dir=str(tmp_path),
    )
    return SSHSandbox(config, host="mac-mini.local", remote_data_root="/tmp/astra")


# ── SSHSandbox.launch ─────────────────────────────────────────────────────────

class TestLaunch:
    def test_launch_creates_remote_dirs_and_transfers_script(self, tmp_path):
        sandbox = _sandbox(tmp_path)
        with patch("backend.sandbox.ssh_sandbox.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(stdout="12345\n")
            sandbox.launch()

        calls = mock_run.call_args_list
        mkdir_call = calls[0].args[0]
        assert mkdir_call[:2] == ["ssh", "mac-mini.local"]
        assert "mkdir -p" in mkdir_call[2]

        scp_call = calls[1].args[0]
        assert scp_call[0] == "scp"
        assert scp_call[1] == "/tmp/fake_train.py"

    def test_launch_sets_remote_pid_and_status(self, tmp_path):
        sandbox = _sandbox(tmp_path)
        with patch("backend.sandbox.ssh_sandbox.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(stdout="54321\n")
            sandbox.launch()
        assert sandbox._remote_pid == 54321
        assert sandbox.status == SandboxStatus.RUNNING


# ── SSHSandbox.is_alive ───────────────────────────────────────────────────────

class TestIsAlive:
    def test_false_when_no_remote_pid(self, tmp_path):
        sandbox = _sandbox(tmp_path)
        assert sandbox.is_alive() is False

    def test_true_when_remote_process_alive(self, tmp_path):
        sandbox = _sandbox(tmp_path)
        sandbox._remote_pid = 111
        with patch("backend.sandbox.ssh_sandbox.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(stdout="alive\n")
            assert sandbox.is_alive() is True

    def test_false_and_syncs_back_when_remote_process_dead(self, tmp_path):
        sandbox = _sandbox(tmp_path)
        sandbox._remote_pid = 111
        with patch("backend.sandbox.ssh_sandbox.subprocess.run") as mock_run, \
             patch.object(sandbox, "_sync_back") as mock_sync:
            mock_run.return_value = MagicMock(stdout="dead\n")
            assert sandbox.is_alive() is False
            mock_sync.assert_called_once()

    def test_true_and_no_sync_back_when_ssh_connection_fails(self, tmp_path):
        """A connection failure (e.g. 'No route to host', empty/garbage stdout)
        must NOT be treated as 'process is dead' — that previously caused a
        still-running remote training job to be dropped from tracking. Fail
        safe: assume alive, don't sync back / declare completion."""
        sandbox = _sandbox(tmp_path)
        sandbox._remote_pid = 111
        with patch("backend.sandbox.ssh_sandbox.subprocess.run") as mock_run, \
             patch.object(sandbox, "_sync_back") as mock_sync:
            mock_run.return_value = MagicMock(stdout="", stderr="ssh: connect to host mac-mini.local port 22: Undefined error: 0\n")
            assert sandbox.is_alive() is True
            mock_sync.assert_not_called()


# ── SSHSandbox.terminate ──────────────────────────────────────────────────────

class TestTerminate:
    def test_terminate_sends_graceful_then_force_kill_sequence(self, tmp_path):
        """terminate() must give the remote process a chance to exit gracefully
        (SIGTERM) before force-killing (SIGKILL), matching SubprocessSandbox."""
        sandbox = _sandbox(tmp_path)
        sandbox._remote_pid = 111
        with patch("backend.sandbox.ssh_sandbox.subprocess.run") as mock_run, \
             patch.object(sandbox, "_sync_back"):
            mock_run.return_value = MagicMock(returncode=0)
            sandbox.terminate()

        kill_call = mock_run.call_args_list[0].args[0]
        remote_cmd = kill_call[2]
        assert "kill -TERM 111" in remote_cmd
        assert "kill -9 111" in remote_cmd
        # TERM must be issued before the fallback KILL in the remote command string
        assert remote_cmd.index("kill -TERM") < remote_cmd.index("kill -9")

    def test_terminate_sets_status_stopped(self, tmp_path):
        sandbox = _sandbox(tmp_path)
        sandbox._remote_pid = 111
        with patch("backend.sandbox.ssh_sandbox.subprocess.run") as mock_run, \
             patch.object(sandbox, "_sync_back"):
            mock_run.return_value = MagicMock(returncode=0)
            sandbox.terminate()
        assert sandbox.status == SandboxStatus.STOPPED

    def test_terminate_calls_sync_back(self, tmp_path):
        sandbox = _sandbox(tmp_path)
        sandbox._remote_pid = 111
        with patch("backend.sandbox.ssh_sandbox.subprocess.run") as mock_run, \
             patch.object(sandbox, "_sync_back") as mock_sync:
            mock_run.return_value = MagicMock(returncode=0)
            sandbox.terminate()
        mock_sync.assert_called_once()

    def test_terminate_with_no_remote_pid_does_not_raise(self, tmp_path):
        sandbox = _sandbox(tmp_path)
        with patch("backend.sandbox.ssh_sandbox.subprocess.run"), \
             patch.object(sandbox, "_sync_back"):
            sandbox.terminate()   # should not raise
        assert sandbox.status == SandboxStatus.STOPPED

    def test_terminate_does_not_claim_stopped_when_ssh_unreachable(self, tmp_path):
        """If the kill command itself can't reach the host, the remote process
        was likely never signaled — don't claim STOPPED or sync_back for a
        kill we couldn't deliver, so an unreachable-but-still-alive process
        stays visible (remote_pid retained) instead of silently orphaned."""
        sandbox = _sandbox(tmp_path)
        sandbox._remote_pid = 111
        with patch("backend.sandbox.ssh_sandbox.subprocess.run") as mock_run, \
             patch.object(sandbox, "_sync_back") as mock_sync:
            mock_run.return_value = MagicMock(returncode=255)
            sandbox.terminate()
        mock_sync.assert_not_called()
        assert sandbox.status != SandboxStatus.STOPPED
        assert sandbox._remote_pid == 111


# ── SSHSandbox.sync_tail_offset_to_current ────────────────────────────────────

class TestSyncTailOffsetToCurrent:
    def test_sets_offset_to_remote_log_size(self, tmp_path):
        sandbox = _sandbox(tmp_path)
        with patch("backend.sandbox.ssh_sandbox.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(stdout="4096\n")
            sandbox.sync_tail_offset_to_current()
        assert sandbox._tail_offset == 4096

    def test_defaults_to_zero_on_unparseable_output(self, tmp_path):
        """If the remote size can't be determined, fail safe with offset 0
        (may re-emit some historical output as new — acceptable — rather than
        crash the reattach path)."""
        sandbox = _sandbox(tmp_path)
        with patch("backend.sandbox.ssh_sandbox.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(stdout="")
            sandbox.sync_tail_offset_to_current()
        assert sandbox._tail_offset == 0


# ── SSHSandbox.get_sandbox_id ─────────────────────────────────────────────────

class TestGetSandboxId:
    def test_returns_none_without_remote_pid(self, tmp_path):
        sandbox = _sandbox(tmp_path)
        assert sandbox.get_sandbox_id() is None

    def test_returns_pid_as_string(self, tmp_path):
        sandbox = _sandbox(tmp_path)
        sandbox._remote_pid = 12345
        assert sandbox.get_sandbox_id() == "12345"


# ── SSHSandbox.tail_new_output ────────────────────────────────────────────────

class TestTailNewOutput:
    def test_first_call_reads_from_start(self, tmp_path):
        sandbox = _sandbox(tmp_path)
        with patch("backend.sandbox.ssh_sandbox.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(stdout="Pass rate: 50.0% (10/20)\n")
            output = sandbox.tail_new_output()

        assert output == "Pass rate: 50.0% (10/20)\n"
        tail_call = mock_run.call_args_list[0].args[0]
        assert tail_call[0] == "ssh"
        assert "tail -c +1" in tail_call[2]

    def test_second_call_advances_offset(self, tmp_path):
        sandbox = _sandbox(tmp_path)
        with patch("backend.sandbox.ssh_sandbox.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(stdout="0123456789")
            sandbox.tail_new_output()

        with patch("backend.sandbox.ssh_sandbox.subprocess.run") as mock_run2:
            mock_run2.return_value = MagicMock(stdout="new stuff")
            sandbox.tail_new_output()

        tail_call = mock_run2.call_args_list[0].args[0]
        assert "tail -c +11" in tail_call[2]   # 10 bytes consumed, 1-indexed next offset

    def test_returns_empty_string_when_nothing_new(self, tmp_path):
        sandbox = _sandbox(tmp_path)
        with patch("backend.sandbox.ssh_sandbox.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(stdout="")
            output = sandbox.tail_new_output()
        assert output == ""


# ── SSHSandbox._sync_back ─────────────────────────────────────────────────────

class TestSyncBack:
    def test_sync_back_fetches_log_and_rsyncs_checkpoints(self, tmp_path):
        sandbox = _sandbox(tmp_path)
        with patch("backend.sandbox.ssh_sandbox.subprocess.run") as mock_run:
            sandbox._sync_back()

        scp_call = mock_run.call_args_list[0].args[0]
        assert scp_call[0] == "scp"
        rsync_call = mock_run.call_args_list[1].args[0]
        assert rsync_call[0] == "rsync"
        assert rsync_call[1] == "-az"
        assert "test-mission/checkpoints/" in rsync_call[2]

    def test_sync_back_uses_remote_checkpoint_dir_override(self, tmp_path):
        """dpo/grpo missions save under finetune_dir/adapters/, not the default
        {remote_mission_dir}/checkpoints — sync-back must pull from there instead."""
        config = SandboxConfig(
            mission_id="test-mission",
            script_path="/tmp/fake_train.py",
            data_dir=str(tmp_path),
            remote_checkpoint_dir="/Users/kewang/finetune/adapters/astra_test-mis",
        )
        sandbox = SSHSandbox(config, host="mac-mini.local", remote_data_root="/tmp/astra")

        with patch("backend.sandbox.ssh_sandbox.subprocess.run") as mock_run:
            sandbox._sync_back()

        rsync_call = mock_run.call_args_list[1].args[0]
        assert "/Users/kewang/finetune/adapters/astra_test-mis/" in rsync_call[2]
        assert "checkpoints" not in rsync_call[2]
