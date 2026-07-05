"""
SubprocessSandbox — Apple Silicon backend.

Spawns a restricted host subprocess (needed because Docker cannot access the
Metal GPU on Apple Silicon). Memory is capped via the POSIX resource module;
CPU affinity is controlled via psutil where available.
"""
from __future__ import annotations

import os
import resource
import subprocess
import sys
from pathlib import Path
from typing import Optional

import psutil

from backend.sandbox.base import BaseSandbox, SandboxConfig, SandboxStatus
from backend.logging_config import get_logger

logger = get_logger(__name__)


def _apply_resource_limits(memory_limit_gb: float, cpu_count: int) -> None:
    """Called as preexec_fn inside the child process."""
    mem_bytes = int(memory_limit_gb * 1024 ** 3)
    try:
        resource.setrlimit(resource.RLIMIT_AS, (mem_bytes, mem_bytes))
    except ValueError:
        # Hard limit may be lower than requested — clamp to current hard limit
        _, hard = resource.getrlimit(resource.RLIMIT_AS)
        if hard != resource.RLIM_INFINITY:
            resource.setrlimit(resource.RLIMIT_AS, (hard, hard))


class SubprocessSandbox(BaseSandbox):
    def __init__(self, config: SandboxConfig) -> None:
        super().__init__(config)
        self._process: Optional[subprocess.Popen] = None
        self._reattach_pid: Optional[int] = None  # set when recovering a live process by pid

    def launch(self) -> None:
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
        log_file = open(self.log_path, "a")

        env = {**os.environ, **self.config.env_vars}
        if self.config.gpu_index is not None:
            env["CUDA_VISIBLE_DEVICES"] = str(self.config.gpu_index)
            env["MPS_DEVICE_INDEX"] = str(self.config.gpu_index)

        # Prefer the .venv interpreter so sandbox scripts have access to
        # project dependencies regardless of how uvicorn was invoked.
        _project_root = Path(__file__).resolve().parents[2]
        _venv_python = _project_root / ".venv" / "bin" / "python"
        python = str(_venv_python) if _venv_python.exists() else sys.executable

        self._process = subprocess.Popen(
            [python, self.config.script_path],
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            preexec_fn=lambda: _apply_resource_limits(
                self.config.memory_limit_gb, self.config.cpu_count
            ),
        )

        # Apply CPU affinity if psutil supports it on this platform
        try:
            proc = psutil.Process(self._process.pid)
            cpu_ids = list(range(self.config.cpu_count))
            proc.cpu_affinity(cpu_ids)
        except (AttributeError, psutil.AccessDenied, NotImplementedError):
            pass  # cpu_affinity not available on all platforms (e.g. macOS)

        self.status = SandboxStatus.RUNNING
        logger.info(
            "SubprocessSandbox launched: mission=%s pid=%d",
            self.config.mission_id,
            self._process.pid,
        )

    def terminate(self) -> None:
        if self._process and self._process.poll() is None:
            self._process.terminate()
            try:
                self._process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self._process.kill()
        elif self._reattach_pid is not None:
            # Reattached after service restart — no Popen handle, kill by stored pid.
            try:
                proc = psutil.Process(self._reattach_pid)
                proc.terminate()
                try:
                    proc.wait(timeout=10)
                except psutil.TimeoutExpired:
                    proc.kill()
                logger.info(
                    "SubprocessSandbox terminated reattached pid=%d: mission=%s",
                    self._reattach_pid, self.config.mission_id,
                )
            except psutil.NoSuchProcess:
                pass
            self._reattach_pid = None
        self.status = SandboxStatus.STOPPED
        logger.info("SubprocessSandbox terminated: mission=%s", self.config.mission_id)

    def is_alive(self) -> bool:
        if self._process is not None:
            return self._process.poll() is None
        # Re-attach path: no Popen handle after a restart — check by PID instead.
        if self._reattach_pid is not None:
            return psutil.pid_exists(self._reattach_pid)
        return False

    def is_pid_alive(self, pid: int) -> bool:
        return psutil.pid_exists(pid)

    def get_sandbox_id(self) -> Optional[str]:
        if self._process:
            return str(self._process.pid)
        return None
