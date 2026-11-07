"""
RLTrainer — legacy host-side stub.

Epoch 9: real Environment RL runs in the sandbox via codegen using
``astragym.make`` + SB3 (optionally ``astrarl.make``). This class is not used
by the mission loop; kept for BaseTrainer registry compatibility.
"""
from __future__ import annotations

import os
from typing import Optional

from backend.trainers.base import BaseTrainer, TrainerConfig
from backend.logging_config import get_logger

logger = get_logger(__name__)


class RLTrainer(BaseTrainer):
    """Deprecated host trainer — sandbox codegen + AstraRL is the real path."""

    def _run_training(self) -> None:
        raise NotImplementedError(
            "RLTrainer is unused: Environment RL trains in the sandbox via "
            "astragym + Stable-Baselines3 (see astrarl.sb3_learner / codegen templates)"
        )

    def save_checkpoint(self) -> str:
        path = os.path.join(self.checkpoint_dir, f"rl_checkpoint_{self._iteration}.zip")
        logger.info("RLTrainer checkpoint: %s", path)
        return path

    def load_checkpoint(self, path: str) -> None:
        logger.info("RLTrainer loading checkpoint: %s", path)
