"""
SandboxManager — unified factory and lifecycle coordinator.

Auto-selects SubprocessSandbox on Apple Silicon (Metal GPU access requires
running on the host) and ContainerSandbox on cloud/CUDA targets.
"""
from __future__ import annotations

import os
import platform
import subprocess
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
    """Default backend for missions that don't force a specific one.

    Deliberately does NOT check settings.sandbox_host — that setting now only
    controls where _FINETUNE_REMOTE_TASK_TYPES missions go (forced in launch()
    below). Making it the general default here would silently reroute RL/ml
    missions to the Mac Mini the moment sandbox_host is configured for
    fine-tuning, which is not what a non-empty sandbox_host should mean.
    """
    if platform.system() == "Darwin" and platform.machine() == "arm64":
        return "subprocess"
    return "container"


# Fine-tune task types that wrap ensemble/finetune scripts living only on the
# Mac Mini — these always dispatch via SSH to settings.sandbox_host, never
# falling back to a local backend if it isn't configured.
_FINETUNE_REMOTE_TASK_TYPES = {"dpo", "grpo"}


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
        path = os.path.abspath(os.path.join(settings.data_path, "missions", mission_id))
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
        task_type: Optional[str] = None,
        remote_checkpoint_dir: Optional[str] = None,
    ) -> tuple[Optional[int], Optional[str]]:
        """
        Launch a sandbox for the given mission.

        Returns (subprocess_pid, container_id) — one will be None depending
        on which backend is used. Store both in the Mission Store.

        remote_checkpoint_dir: for SSHSandbox, where to rsync checkpoints/adapters
        FROM on sync-back, if it differs from the default {remote_mission_dir}/checkpoints
        (e.g. dpo/grpo missions save under finetune_dir/adapters/ instead — see
        backend.agent.code_generator.finetune_checkpoint_dir).
        """
        if task_type in _FINETUNE_REMOTE_TASK_TYPES:
            if not settings.sandbox_host:
                raise RuntimeError(
                    f"{task_type} missions require settings.sandbox_host (Mac Mini) "
                    "to be configured — no local fallback for fine-tune task types"
                )
            backend = "ssh"
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
            remote_checkpoint_dir=remote_checkpoint_dir,
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

        # Kill any existing sandbox for this mission before registering the new one.
        # Guards against leaking processes when retrying after errors.
        existing = self._sandboxes.pop(mission_id, None)
        if existing and existing.is_alive():
            logger.warning("SandboxManager: terminating stale sandbox for mission=%s before new launch", mission_id)
            existing.terminate()

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

    def get_sandbox_id(self, mission_id: str) -> Optional[str]:
        """The sandbox's own id (local pid, container id, or remote pid string) —
        for display/logging only. Distinct from the Mission Store's
        subprocess_pid, which recovery treats as a LOCAL pid via psutil; do not
        use this to populate that field for non-subprocess backends."""
        sandbox = self._sandboxes.get(mission_id)
        return sandbox.get_sandbox_id() if sandbox else None

    def tail_new_output(self, mission_id: str) -> Optional[str]:
        """New log output since the last call, for backends that support live
        tailing without waiting for the sandbox to exit (currently SSHSandbox
        only). Returns None if there's no active sandbox or it doesn't support this."""
        sandbox = self._sandboxes.get(mission_id)
        if sandbox is None or not hasattr(sandbox, "tail_new_output"):
            return None
        return sandbox.tail_new_output()

    def recover(
        self,
        mission_id: str,
        subprocess_pid: Optional[int],
        container_id: Optional[str],
        remote_pid: Optional[int] = None,
    ) -> str:
        """
        Called by the State Recovery Manager on boot for each interrupted mission.

        Returns:
            "reattached" — sandbox is still running; telemetry will back-fill.
            "dead"       — sandbox is gone; caller should re-launch from checkpoint.

        remote_pid: for SSH-dispatched (dpo/grpo) missions — checked via SSH
        kill -0 against settings.sandbox_host, the same liveness semantics as
        the local subprocess_pid check below via psutil. Requires the wrapper
        script to os.execv into the training process (not subprocess.run/fork)
        so this pid IS the actual training process, not an orphan-prone parent.
        """
        if remote_pid is not None:
            if not settings.sandbox_host:
                logger.warning(
                    "Recovery: mission=%s has remote_pid=%d but sandbox_host is "
                    "unconfigured — cannot verify, treating as dead",
                    mission_id, remote_pid,
                )
                return "dead"
            result = subprocess.run(
                ["ssh", settings.sandbox_host,
                 f"kill -0 {remote_pid} 2>/dev/null && echo alive || echo dead"],
                capture_output=True, text=True,
            )
            if result.stdout.strip() == "alive":
                logger.info(
                    "Recovery: remote pid=%d still alive on %s for mission=%s — reattaching",
                    remote_pid, settings.sandbox_host, mission_id,
                )
                data_dir = self._mission_data_dir(mission_id)
                config = SandboxConfig(mission_id=mission_id, script_path="", data_dir=data_dir)
                sandbox = SSHSandbox(config, host=settings.sandbox_host, remote_data_root=settings.sandbox_data_path)
                sandbox._remote_pid = remote_pid
                sandbox.status = SandboxStatus.RUNNING
                sandbox.sync_tail_offset_to_current()
                self._sandboxes[mission_id] = sandbox
                return "reattached"
            else:
                logger.warning(
                    "Recovery: remote pid=%d gone on %s for mission=%s",
                    remote_pid, settings.sandbox_host, mission_id,
                )
                return "dead"

        if subprocess_pid is not None:
            if psutil.pid_exists(subprocess_pid):
                logger.info(
                    "Recovery: subprocess pid=%d still alive for mission=%s — reattaching",
                    subprocess_pid, mission_id,
                )
                # Reconstruct a minimal sandbox so terminate() works.
                # Store the pid so terminate() can kill it by pid (no Popen handle).
                data_dir = self._mission_data_dir(mission_id)
                config = SandboxConfig(
                    mission_id=mission_id,
                    script_path="",  # not needed for reattach
                    data_dir=data_dir,
                )
                sandbox = SubprocessSandbox(config)
                sandbox._reattach_pid = subprocess_pid
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
                probe._container_id = container_id
                probe.status = SandboxStatus.RUNNING
                self._sandboxes[mission_id] = probe
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
