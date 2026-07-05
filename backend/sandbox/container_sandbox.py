"""
ContainerSandbox — Cloud / CUDA backend.

Uses the Docker SDK to run training code in an isolated container with optional
nvidia-container-toolkit GPU passthrough. Not used on Apple Silicon (Metal GPU
is not accessible inside Docker on macOS).
"""
from __future__ import annotations

import os
from typing import Optional

from backend.sandbox.base import BaseSandbox, SandboxConfig, SandboxStatus
from backend.logging_config import get_logger

logger = get_logger(__name__)

_DOCKER_AVAILABLE = False
try:
    import docker
    _DOCKER_AVAILABLE = True
except ImportError:
    pass


class ContainerSandbox(BaseSandbox):
    DEFAULT_IMAGE = "python:3.11-slim"

    def __init__(self, config: SandboxConfig, image: str = DEFAULT_IMAGE) -> None:
        super().__init__(config)
        self.image = image
        self._container = None
        self._container_id: Optional[str] = None

        if not _DOCKER_AVAILABLE:
            raise RuntimeError(
                "docker package is not installed. Run: pip install docker"
            )

    def _get_client(self):
        return docker.from_env()

    def launch(self) -> None:
        client = self._get_client()
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)

        device_requests = []
        if self.config.gpu_index is not None:
            device_requests = [
                docker.types.DeviceRequest(device_ids=[str(self.config.gpu_index)], capabilities=[["gpu"]])
            ]
        elif self.config.gpu:
            device_requests = [
                docker.types.DeviceRequest(count=-1, capabilities=[["gpu"]])
            ]

        self._container = client.containers.run(
            self.image,
            command=["python", self.config.script_path],
            environment=self.config.env_vars,
            mem_limit=f"{int(self.config.memory_limit_gb)}g",
            nano_cpus=int(self.config.cpu_count * 1e9),
            device_requests=device_requests,
            volumes={
                os.path.abspath(self.config.data_dir): {
                    "bind": "/data",
                    "mode": "rw",
                }
            },
            detach=True,
            remove=False,
        )
        self._container_id = self._container.id
        self.status = SandboxStatus.RUNNING
        logger.info(
            "ContainerSandbox launched: mission=%s container=%s",
            self.config.mission_id,
            self._container_id[:12],
        )

    def terminate(self) -> None:
        if self._container:
            try:
                self._container.stop(timeout=10)
            except Exception:
                self._container.kill()
        self.status = SandboxStatus.STOPPED
        logger.info("ContainerSandbox stopped: mission=%s", self.config.mission_id)

    def is_alive(self) -> bool:
        if self._container:
            try:
                self._container.reload()
                return self._container.status == "running"
            except Exception:
                return False
        # Re-attach path: no live container handle after a restart — check by ID instead.
        if self._container_id is not None:
            return self.is_container_alive(self._container_id)
        return False

    def is_container_alive(self, container_id: str) -> bool:
        """Check a container by ID (used during recovery)."""
        if not _DOCKER_AVAILABLE:
            return False
        try:
            client = self._get_client()
            container = client.containers.get(container_id)
            return container.status == "running"
        except Exception:
            return False

    def get_sandbox_id(self) -> Optional[str]:
        return self._container_id
