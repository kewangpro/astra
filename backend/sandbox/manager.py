"""
SandboxManager — unified factory and lifecycle coordinator.

Auto-selects SubprocessSandbox on Apple Silicon (Metal GPU access requires
running on the host) and ContainerSandbox on cloud/CUDA targets.
"""
from __future__ import annotations

import os
import platform
from typing import Optional

import psutil

from backend.sandbox.base import SandboxConfig, SandboxStatus
from backend.sandbox.subprocess_sandbox import SubprocessSandbox
from backend.sandbox.container_sandbox import ContainerSandbox
from backend.sandbox.ssh_sandbox import SSHSandbox
from backend.config import settings
from backend.logging_config import get_logger

logger = get_logger(__name__)


def _detect_backend() -> str:
    if settings.sandbox_host:
        return "ssh"
    if platform.system() == "Darwin" and platform.machine() == "arm64":
        return "subprocess"
    return "container"


class GPUPool:
    """
    Tracks GPU assignment per mission.
    Picks the least-loaded GPU index for each new launch.
    gpu_count=0 means GPU pinning is disabled (single-GPU or CPU-only).
    """

    def __init__(self, gpu_count: int = 0) -> None:
        self._gpu_count = gpu_count
        self._assignments: dict[str, int] = {}   # mission_id → gpu_index

    def acquire(self, mission_id: str) -> Optional[int]:
        if self._gpu_count == 0:
            return None
        counts = [0] * self._gpu_count
        for idx in self._assignments.values():
            counts[idx] += 1
        chosen = counts.index(min(counts))
        self._assignments[mission_id] = chosen
        return chosen

    def release(self, mission_id: str) -> None:
        self._assignments.pop(mission_id, None)


class SandboxManager:
    """
    Manages the full lifecycle of training sandboxes:
    launch → monitor → terminate → recover.
    """

    def __init__(self) -> None:
        self.default_backend = _detect_backend()
        self._sandboxes: dict[str, SubprocessSandbox | ContainerSandbox] = {}
        self._gpu_pool = GPUPool(gpu_count=int(os.environ.get("ASTRA_GPU_COUNT", "0")))
        logger.info("SandboxManager initialized (default backend: %s)", self.default_backend)

    def _mission_data_dir(self, mission_id: str) -> str:
        path = os.path.join(settings.data_path, "missions", mission_id)
        os.makedirs(path, exist_ok=True)
        return path

    def launch(
        self,
        mission_id: str,
        script_path: str,
        *,
        env_vars: Optional[dict] = None,
        memory_limit_gb: float = 8.0,
        cpu_count: int = 4,
        gpu: bool = False,
        gpu_index: Optional[int] = None,
        image: str = ContainerSandbox.DEFAULT_IMAGE,
        backend: Optional[str] = None,
    ) -> tuple[Optional[int], Optional[str]]:
        """
        Launch a sandbox for the given mission.

        Returns (subprocess_pid, container_id) — one will be None depending
        on which backend is used. Store both in the Mission Store.
        """
        backend = backend or self.default_backend
        data_dir = self._mission_data_dir(mission_id)
        assigned_gpu = gpu_index if gpu_index is not None else self._gpu_pool.acquire(mission_id)

        config = SandboxConfig(
            mission_id=mission_id,
            script_path=script_path,
            data_dir=data_dir,
            env_vars=env_vars or {},
            memory_limit_gb=memory_limit_gb,
            cpu_count=cpu_count,
            gpu=gpu,
            gpu_index=assigned_gpu,
        )

        if backend == "ssh":
            sandbox = SSHSandbox(
                config,
                host=settings.sandbox_host,
                remote_data_root=settings.sandbox_data_path,
            )
        elif backend == "subprocess":
            sandbox = SubprocessSandbox(config)
        else:
            sandbox = ContainerSandbox(config, image=image)

        sandbox.launch()
        self._sandboxes[mission_id] = sandbox

        pid = int(sandbox.get_sandbox_id()) if backend == "subprocess" else None
        container_id = sandbox.get_sandbox_id() if backend == "container" else None
        return pid, container_id

    def terminate(self, mission_id: str) -> None:
        sandbox = self._sandboxes.get(mission_id)
        if sandbox:
            sandbox.terminate()
            del self._sandboxes[mission_id]
        self._gpu_pool.release(mission_id)

    def is_alive(self, mission_id: str) -> bool:
        sandbox = self._sandboxes.get(mission_id)
        return sandbox.is_alive() if sandbox else False

    def recover(
        self,
        mission_id: str,
        subprocess_pid: Optional[int],
        container_id: Optional[str],
    ) -> str:
        """
        Called by the State Recovery Manager on boot for each interrupted mission.

        Returns:
            "reattached" — sandbox is still running; telemetry will back-fill.
            "dead"       — sandbox is gone; caller should re-launch from checkpoint.
        """
        if subprocess_pid is not None:
            if psutil.pid_exists(subprocess_pid):
                logger.info(
                    "Recovery: subprocess pid=%d still alive for mission=%s — reattaching",
                    subprocess_pid, mission_id,
                )
                # Reconstruct a minimal sandbox object so is_alive() works
                data_dir = self._mission_data_dir(mission_id)
                config = SandboxConfig(
                    mission_id=mission_id,
                    script_path="",  # not needed for reattach
                    data_dir=data_dir,
                )
                sandbox = SubprocessSandbox(config)
                sandbox.status = SandboxStatus.RUNNING
                self._sandboxes[mission_id] = sandbox
                return "reattached"
            else:
                logger.warning(
                    "Recovery: subprocess pid=%d gone for mission=%s",
                    subprocess_pid, mission_id,
                )
                return "dead"

        if container_id is not None:
            probe = ContainerSandbox(
                SandboxConfig(mission_id=mission_id, script_path="", data_dir="")
            )
            if probe.is_container_alive(container_id):
                logger.info(
                    "Recovery: container %s still running for mission=%s — reattaching",
                    container_id[:12], mission_id,
                )
                return "reattached"
            else:
                logger.warning(
                    "Recovery: container %s gone for mission=%s",
                    container_id[:12], mission_id,
                )
                return "dead"

        return "dead"

    def get_log_path(self, mission_id: str) -> str:
        return os.path.join(self._mission_data_dir(mission_id), "sandbox.log")


# Singleton — shared across the application
sandbox_manager = SandboxManager()
