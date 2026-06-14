"""
MLTrainer — wraps Scikit-learn / PyTorch Lightning.

Classical ML and Lightning models use joblib / trainer.save_checkpoint()
for persistence. Checkpoint cadence is handled by the base class thread.
"""
from __future__ import annotations

import os

from backend.trainers.base import BaseTrainer, TrainerConfig
from backend.logging_config import get_logger

logger = get_logger(__name__)


class MLTrainer(BaseTrainer):
    """
    Classical ML / Lightning trainer.

    Expected hyperparameters:
        framework       : "sklearn" | "lightning"
        model_class     : str  (e.g. "RandomForestClassifier", "LightningModule subclass")
        dataset_path    : str
        target_column   : str
        model_params    : dict
        max_epochs      : int  (Lightning only)
    """

    def _run_training(self) -> None:
        # Phase 3: Lead Agent generates and injects the training loop here.
        #
        # Scikit-learn pattern:
        # from sklearn.ensemble import RandomForestClassifier
        # model = RandomForestClassifier(**self.config.hyperparameters["model_params"])
        # model.fit(X_train, y_train)
        # self.log_metric("accuracy", model.score(X_val, y_val))
        #
        # Lightning pattern:
        # trainer = pl.Trainer(max_epochs=..., default_root_dir=self.checkpoint_dir)
        # trainer.fit(model, datamodule)
        # for metric in trainer.callback_metrics:
        #     self.log_metric(metric, trainer.callback_metrics[metric])
        raise NotImplementedError("MLTrainer._run_training() is injected by Phase 3 Lead Agent")

    def save_checkpoint(self) -> str:
        path = os.path.join(self.checkpoint_dir, f"ml_checkpoint_{self._iteration}.pkl")
        # joblib.dump(model, path)  — called by Phase 3 generated code
        logger.info("MLTrainer checkpoint: %s", path)
        return path

    def load_checkpoint(self, path: str) -> None:
        # model = joblib.load(path)
        logger.info("MLTrainer loading checkpoint: %s", path)
