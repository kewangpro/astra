"""
SFTTrainer — wraps HuggingFace Transformers + PEFT (LoRA / QLoRA).

save_strategy is forced to "steps" with save_steps tuned so checkpoints land
within the 2-5 minute window regardless of dataset size.
"""
from __future__ import annotations

import os

from backend.trainers.base import BaseTrainer, TrainerConfig, CHECKPOINT_INTERVAL_SEC
from backend.logging_config import get_logger

logger = get_logger(__name__)

# Rough estimate: force a save every N steps so wall-clock time ≈ CHECKPOINT_INTERVAL_SEC.
# Phase 3 Lead Agent refines this based on observed step duration.
DEFAULT_SAVE_STEPS = 200


class SFTTrainer(BaseTrainer):
    """
    Supervised Fine-Tuning trainer.

    Expected hyperparameters:
        base_model                  : str  (e.g. "meta-llama/Llama-3.1-8B")
        dataset_path                : str
        lora_r                      : int
        lora_alpha                  : int
        lora_dropout                : float
        per_device_train_batch_size : int
        gradient_accumulation_steps : int
        learning_rate               : float
        num_train_epochs            : int
        save_steps                  : int  (defaults to DEFAULT_SAVE_STEPS)
    """

    def _run_training(self) -> None:
        # Phase 3: Lead Agent generates and injects the training loop here.
        # Expected call pattern:
        #
        # from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
        # from peft import get_peft_model, LoraConfig
        # from trl import SFTTrainer as HFSFTTrainer
        #
        # model = AutoModelForCausalLM.from_pretrained(base_model, load_in_4bit=True)
        # model = get_peft_model(model, LoraConfig(...))
        # trainer = HFSFTTrainer(
        #     model=model,
        #     args=TrainingArguments(
        #         save_strategy="steps",
        #         save_steps=self.config.hyperparameters.get("save_steps", DEFAULT_SAVE_STEPS),
        #         output_dir=self.checkpoint_dir,
        #         ...
        #     ),
        #     ...
        # )
        # for epoch_metrics in trainer.train_generator():
        #     self.log_metric("train_loss", epoch_metrics["loss"], step=epoch_metrics["step"])
        #     self._iteration += 1
        raise NotImplementedError("SFTTrainer._run_training() is injected by Phase 3 Lead Agent")

    def save_checkpoint(self) -> str:
        path = os.path.join(self.checkpoint_dir, f"sft_checkpoint_{self._iteration}")
        # model.save_pretrained(path) — called by Phase 3 generated code
        logger.info("SFTTrainer checkpoint: %s", path)
        return path

    def load_checkpoint(self, path: str) -> None:
        # model = AutoModelForCausalLM.from_pretrained(path)
        logger.info("SFTTrainer loading checkpoint: %s", path)
