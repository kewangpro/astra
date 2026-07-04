"""
SSHSandbox — remote training execution on a separate machine.

Transfers the training script via scp, launches it in the background via ssh,
polls liveness with kill -0, and rsyncs logs + checkpoints back on exit.

Requires passwordless SSH access to the remote host.
"""
from __future__ import annotations

import os
import subprocess
from typing import Optional

from backend.sandbox.base import BaseSandbox, SandboxConfig, SandboxStatus
from backend.logging_config import get_logger
from backend.config import settings

logger = get_logger(__name__)


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
            ["ssh", self._host, f"mkdir -p {self._remote_mission_dir}/checkpoints"],
            check=True, capture_output=True,
        )

        # Transfer training script
        subprocess.run(
            ["scp", self.config.script_path, f"{self._host}:{self._remote_script}"],
            check=True, capture_output=True,
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
            ["ssh", self._host, cmd],
            capture_output=True, text=True, check=True,
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
        result = subprocess.run(
            ["ssh", self._host,
             f"kill -0 {self._remote_pid} 2>/dev/null && echo alive || echo dead"],
            capture_output=True, text=True,
        )
        alive = result.stdout.strip() == "alive"
        if not alive:
            self._sync_back()
        return alive

    def terminate(self) -> None:
        if self._remote_pid:
            subprocess.run(
                ["ssh", self._host,
                 f"kill -TERM {self._remote_pid} 2>/dev/null; "
                 f"for i in $(seq 1 10); do "
                 f"kill -0 {self._remote_pid} 2>/dev/null || exit 0; sleep 1; done; "
                 f"kill -9 {self._remote_pid} 2>/dev/null"],
                capture_output=True,
            )
        self._sync_back()
        self.status = SandboxStatus.STOPPED
        logger.info("SSHSandbox terminated: mission=%s host=%s", self.config.mission_id, self._host)

    def get_sandbox_id(self) -> Optional[str]:
        return str(self._remote_pid) if self._remote_pid else None

    @property
    def log_path(self) -> str:
        return self._local_log

    def tail_new_output(self) -> str:
        """Return remote log bytes written since the last call, without waiting
        for the process to exit. Used for live progress on long fine-tune runs,
        where _sync_back() only happens on death — no network calls or extra
        deps needed on the remote host, this reads the log ssh already redirects
        the training subprocess's stdout/stderr into."""
        result = subprocess.run(
            ["ssh", self._host, f"tail -c +{self._tail_offset + 1} {self._remote_log} 2>/dev/null"],
            capture_output=True, text=True,
        )
        new_output = result.stdout
        self._tail_offset += len(new_output.encode("utf-8"))
        return new_output

    def _sync_back(self) -> None:
        """Fetch sandbox log and rsync checkpoints from mac-mini to local."""
        os.makedirs(self.config.data_dir, exist_ok=True)

        # Fetch log
        subprocess.run(
            ["scp", f"{self._host}:{self._remote_log}", self._local_log],
            capture_output=True,
        )

        # Rsync checkpoints — from remote_checkpoint_dir if set (e.g. dpo/grpo save
        # under finetune_dir/adapters/, not the generic mission checkpoints path),
        # otherwise the default {remote_mission_dir}/checkpoints/.
        remote_source = self.config.remote_checkpoint_dir or os.path.join(self._remote_mission_dir, "checkpoints")
        local_ckpt = os.path.join(self.config.data_dir, "checkpoints")
        os.makedirs(local_ckpt, exist_ok=True)
        subprocess.run(
            ["rsync", "-az",
             f"{self._host}:{remote_source}/",
             f"{local_ckpt}/"],
            capture_output=True,
        )
        logger.info("SSHSandbox: synced logs+checkpoints from %s for mission=%s",
                    self._host, self.config.mission_id)
