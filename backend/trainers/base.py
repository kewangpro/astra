"""
BaseTrainer — runs inside the sandbox (subprocess or container).

Responsibilities:
- Background checkpoint thread: saves weights every CHECKPOINT_INTERVAL_SEC.
- Metric logging: writes to telemetry.jsonl AND POSTs to FastAPI for live HUD.
- Checkpoint registration: PATCHes the Model Registry with the latest path.
"""
from __future__ import annotations

import json
import os
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

import httpx

from backend.logging_config import get_logger

logger = get_logger(__name__)

CHECKPOINT_INTERVAL_SEC = 180   # target: every 3 minutes (within 2-5 min window)


@dataclass
class TrainerConfig:
    mission_id: str
    model_record_id: str            # ID in the Model Registry to update
    data_dir: str                   # mission's data/ subdirectory
    hyperparameters: dict = field(default_factory=dict)
    target_metric: dict = field(default_factory=dict)  # {"mean_reward": 150}
    api_base_url: str = "http://127.0.0.1:8200"
    checkpoint_interval_sec: int = CHECKPOINT_INTERVAL_SEC


class BaseTrainer(ABC):
    def __init__(self, config: TrainerConfig) -> None:
        self.config = config
        self._stop_event = threading.Event()
        self._checkpoint_thread: Optional[threading.Thread] = None
        self._iteration = 0
        self.telemetry_path = os.path.join(config.data_dir, "telemetry.jsonl")
        self.checkpoint_dir = os.path.join(config.data_dir, "checkpoints")
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    # ── Public interface ───────────────────────────────────────────────────────

    def run(self) -> None:
        """Entry point called by the sandbox script."""
        logger.info("Trainer starting: mission=%s", self.config.mission_id)
        self._start_checkpoint_thread()
        try:
            self._run_training()
        finally:
            self._stop_checkpoint_thread()
            self.save_checkpoint()   # final checkpoint on exit
            logger.info("Trainer finished: mission=%s", self.config.mission_id)

    def log_metric(self, name: str, value: float, step: Optional[int] = None) -> None:
        """Write a metric to the JSONL log and push to FastAPI telemetry endpoint."""
        event = {
            "mission_id": self.config.mission_id,
            "name": name,
            "value": value,
            "step": step,
            "iteration": self._iteration,
            "recorded_at": datetime.now(timezone.utc).isoformat(),
        }
        self._write_telemetry(event)
        self._push_metric(event)

    # ── Abstract methods — implemented by each specialist trainer ──────────────

    @abstractmethod
    def _run_training(self) -> None:
        """Execute the domain-specific training loop."""

    @abstractmethod
    def save_checkpoint(self) -> str:
        """
        Persist weights + optimizer state to self.checkpoint_dir.
        Returns the path to the saved checkpoint file.
        """

    @abstractmethod
    def load_checkpoint(self, path: str) -> None:
        """Resume training from a checkpoint path."""

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _start_checkpoint_thread(self) -> None:
        self._checkpoint_thread = threading.Thread(
            target=self._checkpoint_loop, daemon=True
        )
        self._checkpoint_thread.start()

    def _stop_checkpoint_thread(self) -> None:
        self._stop_event.set()
        if self._checkpoint_thread:
            self._checkpoint_thread.join(timeout=30)

    def _checkpoint_loop(self) -> None:
        while not self._stop_event.wait(timeout=self.config.checkpoint_interval_sec):
            try:
                path = self.save_checkpoint()
                self._register_checkpoint(path)
            except Exception as e:
                logger.error("Checkpoint failed: %s", e)

    def _write_telemetry(self, event: dict) -> None:
        try:
            with open(self.telemetry_path, "a") as f:
                f.write(json.dumps(event) + "\n")
        except Exception as e:
            logger.error("Failed to write telemetry: %s", e)

    def _push_metric(self, event: dict) -> None:
        url = f"{self.config.api_base_url}/telemetry/missions/{self.config.mission_id}/metrics"
        try:
            httpx.post(url, json=event, timeout=2.0)
        except Exception:
            pass  # telemetry is best-effort; training must not block

    def _register_checkpoint(self, path: str) -> None:
        url = f"{self.config.api_base_url}/registry/models/{self.config.model_record_id}"
        try:
            httpx.patch(url, json={"checkpoint_path": path}, timeout=2.0)
        except Exception:
            pass
