"""
RLTrainer — wraps Stable-Baselines3 / PyTorch.

The _run_training() body is a stub; Phase 3 (Lead Agent) will inject the
environment setup, algorithm selection, and reward shaping via code generation.
SB3 and PyTorch are installed inside the sandbox, not the host environment.
"""
from __future__ import annotations

import os
from typing import Optional

from backend.trainers.base import BaseTrainer, TrainerConfig
from backend.logging_config import get_logger

logger = get_logger(__name__)


class RLTrainer(BaseTrainer):
    """
    Reinforcement Learning trainer.

    Expected hyperparameters:
        algorithm       : "PPO" | "DQN" | "A2C"
        env_id          : str   (e.g. "Snake-v0")
        total_timesteps : int
        learning_rate   : float
        gamma           : float
        batch_size      : int
    """

    def _run_training(self) -> None:
        # Phase 3: Lead Agent generates and injects the training loop here.
        # The structure below shows the expected call pattern for SB3.
        #
        # from stable_baselines3 import PPO
        # from stable_baselines3.common.callbacks import CheckpointCallback
        #
        # env = make_env(self.config.hyperparameters["env_id"])
        # model = PPO(
        #     "MlpPolicy", env,
        #     learning_rate=self.config.hyperparameters.get("learning_rate", 3e-4),
        #     gamma=self.config.hyperparameters.get("gamma", 0.99),
        #     batch_size=self.config.hyperparameters.get("batch_size", 64),
        #     verbose=0,
        # )
        # for step in range(0, total_timesteps, eval_interval):
        #     model.learn(eval_interval, reset_num_timesteps=False)
        #     mean_reward = evaluate(model, env)
        #     self.log_metric("mean_reward", mean_reward, step=step)
        #     self._iteration += 1
        raise NotImplementedError("RLTrainer._run_training() is injected by Phase 3 Lead Agent")

    def save_checkpoint(self) -> str:
        path = os.path.join(self.checkpoint_dir, f"rl_checkpoint_{self._iteration}.zip")
        # model.save(path)  — called by Phase 3 generated code
        logger.info("RLTrainer checkpoint: %s", path)
        return path

    def load_checkpoint(self, path: str) -> None:
        # model = PPO.load(path)
        logger.info("RLTrainer loading checkpoint: %s", path)
