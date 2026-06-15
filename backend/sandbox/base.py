from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class SandboxStatus(str, Enum):
    IDLE = "idle"
    RUNNING = "running"
    STOPPED = "stopped"
    FAILED = "failed"


@dataclass
class SandboxConfig:
    mission_id: str
    script_path: str
    data_dir: str                        # path to mission's data/ subdirectory
    env_vars: dict = field(default_factory=dict)
    memory_limit_gb: float = 8.0
    cpu_count: int = 4
    gpu: bool = False                    # only meaningful for ContainerSandbox
    gpu_index: Optional[int] = None      # None = no pinning; int = specific GPU device index


class BaseSandbox(ABC):
    def __init__(self, config: SandboxConfig) -> None:
        self.config = config
        self.status = SandboxStatus.IDLE

    @abstractmethod
    def launch(self) -> None:
        """Start the sandbox and begin execution."""

    @abstractmethod
    def terminate(self) -> None:
        """Stop the sandbox immediately."""

    @abstractmethod
    def is_alive(self) -> bool:
        """Return True if the sandbox process/container is still running."""

    @abstractmethod
    def get_sandbox_id(self) -> Optional[str]:
        """Return PID (subprocess) or container ID (docker) for Mission Store."""

    @property
    def log_path(self) -> str:
        import os
        return os.path.join(self.config.data_dir, "sandbox.log")
