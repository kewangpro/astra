"""
SSHSandbox — remote training execution on a separate machine.

Transfers the training script via scp, launches it in the background via ssh,
polls liveness with kill -0, and rsyncs logs + checkpoints back on exit.

Requires passwordless SSH access to the remote host.

All ssh/scp/rsync calls force IPv4 (-4). mDNS ".local" hostnames often
resolve to both an IPv4 address and IPv6 addresses (including a link-local
fe80::... one) — if a client picks the IPv6 address without the right
interface scope, it fails with "No route to host" even though the same
hostname is reachable fine over IPv4 (e.g. via `ping`, which typically
prefers IPv4). Confirmed as the suspected cause of repeated, consistent
mkdir-over-ssh failures during otherwise-stable connectivity.

All calls also carry an explicit timeout. Forcing IPv4 above removes one
source of hangs, but a connection can still stall indefinitely if a SYN
packet is silently dropped rather than actively refused (no route-to-host
response at all) — confirmed via a real incident where exactly this froze
one of these calls, and because these run synchronously inside the async
event loop (not in a thread), it froze the entire backend, every mission,
every endpoint, not just the one SSH call. A bounded timeout turns a
service-wide indefinite hang into a single call failing after a few
seconds, handled the same way each site already handles a fast connection
failure.
"""
from __future__ import annotations

import os
import subprocess
from typing import Optional

from backend.sandbox.base import BaseSandbox, SandboxConfig, SandboxStatus
from backend.logging_config import get_logger
from backend.config import settings

logger = get_logger(__name__)

# Quick remote commands (mkdir, kill -0, wc -c, tail) — should return almost
# instantly if the host is reachable at all.
_QUICK_TIMEOUT = 15
# terminate()'s remote command has its own up-to-10s retry loop server-side;
# give it headroom beyond that on top of normal SSH overhead.
_TERMINATE_TIMEOUT = 20
# scp/rsync of scripts/checkpoints — larger payloads (adapter files can be
# tens of MB), needs more room than a quick one-line remote command.
_TRANSFER_TIMEOUT = 60


class SSHSandbox(BaseSandbox):
    def __init__(self, config: SandboxConfig, host: str, remote_data_root: str = "/tmp/astra") -> None:
        super().__init__(config)
        self._host = host
        self._remote_root = remote_data_root
        self._remote_mission_dir = os.path.join(remote_data_root, "missions", config.mission_id)
        self._remote_script = os.path.join(self._remote_mission_dir, "train.py")
        self._remote_log = os.path.join(self._remote_mission_dir, "sandbox.log")
        self._remote_pid: Optional[int] = None
        self._local_log = os.path.join(config.data_dir, "sandbox.log")
        self._tail_offset = 0

    def launch(self) -> None:
        # Create remote directories
        subprocess.run(
            ["ssh", "-4", self._host, f"mkdir -p {self._remote_mission_dir}/checkpoints"],
            check=True, capture_output=True, timeout=_QUICK_TIMEOUT,
        )

        # Transfer training script
        subprocess.run(
            ["scp", "-4", self.config.script_path, f"{self._host}:{self._remote_script}"],
            check=True, capture_output=True, timeout=_TRANSFER_TIMEOUT,
        )

        # Build env string
        env_vars = {
            **self.config.env_vars,
            "ASTRA_CHECKPOINT_DIR": os.path.join(self._remote_mission_dir, "checkpoints"),
        }
        if self.config.gpu_index is not None:
            env_vars["CUDA_VISIBLE_DEVICES"] = str(self.config.gpu_index)
            env_vars["MPS_DEVICE_INDEX"] = str(self.config.gpu_index)
        env_str = " ".join(f"{k}={v}" for k, v in env_vars.items())

        # Launch nohup in background; echo PID so we can track it
        python = settings.sandbox_python or "python3"
        cmd = f"nohup env {env_str} {python} {self._remote_script} > {self._remote_log} 2>&1 & echo $!"
        result = subprocess.run(
            ["ssh", "-4", self._host, cmd],
            capture_output=True, text=True, check=True, timeout=_QUICK_TIMEOUT,
        )
        self._remote_pid = int(result.stdout.strip())
        self.status = SandboxStatus.RUNNING
        logger.info(
            "SSHSandbox launched: mission=%s host=%s remote_pid=%d",
            self.config.mission_id, self._host, self._remote_pid,
        )

    def is_alive(self) -> bool:
        if self._remote_pid is None:
            return False
        try:
            result = subprocess.run(
                ["ssh", "-4", self._host,
                 f"kill -0 {self._remote_pid} 2>/dev/null && echo alive || echo dead"],
                capture_output=True, text=True, timeout=_QUICK_TIMEOUT,
            )
        except subprocess.TimeoutExpired:
            logger.warning(
                "SSHSandbox: is_alive check timed out for mission=%s host=%s — assuming still alive",
                self.config.mission_id, self._host,
            )
            return True
        output = result.stdout.strip()
        if output == "alive":
            return True
        if output == "dead":
            self._sync_back()
            return False
        # SSH itself failed to connect/run (e.g. transient network blip) — we
        # can't confirm the remote process is actually gone. Fail safe and
        # report alive rather than treating an unreachable host as job
        # completion; a false "dead" here previously caused a still-running
        # remote training process to be silently dropped from tracking.
        logger.warning(
            "SSHSandbox: is_alive check inconclusive for mission=%s host=%s "
            "(stderr=%s) — assuming still alive",
            self.config.mission_id, self._host, result.stderr.strip(),
        )
        return True

    def terminate(self) -> None:
        if self._remote_pid:
            try:
                result = subprocess.run(
                    ["ssh", "-4", self._host,
                     f"kill -TERM {self._remote_pid} 2>/dev/null; "
                     f"for i in $(seq 1 10); do "
                     f"kill -0 {self._remote_pid} 2>/dev/null || exit 0; sleep 1; done; "
                     f"kill -9 {self._remote_pid} 2>/dev/null"],
                    capture_output=True, timeout=_TERMINATE_TIMEOUT,
                )
            except subprocess.TimeoutExpired:
                logger.warning(
                    "SSHSandbox: terminate timed out reaching host=%s for mission=%s "
                    "(remote_pid=%d may still be running)",
                    self._host, self.config.mission_id, self._remote_pid,
                )
                return
            if result.returncode != 0:
                # SSH itself failed (e.g. host unreachable) — the remote
                # process was likely never signaled. Don't claim STOPPED for
                # a kill we couldn't deliver; leave remote_pid set so it can
                # be reconciled later instead of being silently orphaned.
                logger.warning(
                    "SSHSandbox: terminate could not reach host=%s for mission=%s "
                    "(remote_pid=%d may still be running)",
                    self._host, self.config.mission_id, self._remote_pid,
                )
                return
        self._sync_back()
        self.status = SandboxStatus.STOPPED
        logger.info("SSHSandbox terminated: mission=%s host=%s", self.config.mission_id, self._host)

    def get_sandbox_id(self) -> Optional[str]:
        return str(self._remote_pid) if self._remote_pid else None

    def sync_tail_offset_to_current(self) -> None:
        """Set the tail offset to the remote log's current size. Used when
        reattaching to an already-running process (e.g. after a backend
        restart) so the next tail_new_output() call returns only output
        written from this point on, instead of re-emitting the entire
        historical log as if it were new."""
        try:
            result = subprocess.run(
                ["ssh", "-4", self._host, f"wc -c < {self._remote_log} 2>/dev/null"],
                capture_output=True, text=True, timeout=_QUICK_TIMEOUT,
            )
            self._tail_offset = int(result.stdout.strip())
        except (ValueError, TypeError, subprocess.TimeoutExpired):
            self._tail_offset = 0

    @property
    def log_path(self) -> str:
        return self._local_log

    def tail_new_output(self) -> str:
        """Return remote log bytes written since the last call, without waiting
        for the process to exit. Used for live progress on long fine-tune runs,
        where _sync_back() only happens on death — no network calls or extra
        deps needed on the remote host, this reads the log ssh already redirects
        the training subprocess's stdout/stderr into."""
        try:
            result = subprocess.run(
                ["ssh", "-4", self._host, f"tail -c +{self._tail_offset + 1} {self._remote_log} 2>/dev/null"],
                capture_output=True, text=True, timeout=_QUICK_TIMEOUT,
            )
        except subprocess.TimeoutExpired:
            return ""
        new_output = result.stdout
        self._tail_offset += len(new_output.encode("utf-8"))
        return new_output

    def _sync_back(self) -> None:
        """Fetch sandbox log and rsync checkpoints from mac-mini to local."""
        os.makedirs(self.config.data_dir, exist_ok=True)

        # Fetch log
        try:
            subprocess.run(
                ["scp", "-4", f"{self._host}:{self._remote_log}", self._local_log],
                capture_output=True, timeout=_TRANSFER_TIMEOUT,
            )
        except subprocess.TimeoutExpired:
            logger.warning("SSHSandbox: log sync timed out for mission=%s", self.config.mission_id)
            return

        # Rsync checkpoints — from remote_checkpoint_dir if set (e.g. dpo/grpo save
        # under finetune_dir/adapters/, not the generic mission checkpoints path),
        # otherwise the default {remote_mission_dir}/checkpoints/.
        remote_source = self.config.remote_checkpoint_dir or os.path.join(self._remote_mission_dir, "checkpoints")
        local_ckpt = os.path.join(self.config.data_dir, "checkpoints")
        os.makedirs(local_ckpt, exist_ok=True)
        try:
            subprocess.run(
                ["rsync", "-az", "-4",
                 f"{self._host}:{remote_source}/",
                 f"{local_ckpt}/"],
                capture_output=True, timeout=_TRANSFER_TIMEOUT,
            )
        except subprocess.TimeoutExpired:
            logger.warning("SSHSandbox: checkpoint sync timed out for mission=%s", self.config.mission_id)
            return
        logger.info("SSHSandbox: synced logs+checkpoints from %s for mission=%s",
                    self._host, self.config.mission_id)
