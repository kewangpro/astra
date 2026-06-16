"""
CodeGenerator — Step 3.2.

Produces runnable Python training scripts from a plan dict.
The script is written to data/missions/{id}/train.py and executed
by the SandboxManager.
"""
from __future__ import annotations

import json
import os
from typing import Optional

from backend.agent.inference.base import InferenceProvider, Message, GenerationConfig
from backend.config import settings
from backend.logging_config import get_logger

logger = get_logger(__name__)

# ── System prompts ─────────────────────────────────────────────────────────────

_BASE_SYSTEM = """\
You are ASTRA's Code Generator — an expert ML engineer.
Generate complete, runnable Python training scripts.
Requirements:
- The script must run standalone inside a sandboxed Python environment.
- Import all dependencies at the top.
- For RL scripts: ALWAYS use `import gymnasium as gym` — NOT `import gym`.
  The `gym` package is NOT installed; only `gymnasium` is available.
- NEVER import or call `stable_baselines3.common.logger.configure()` — it is
  not needed. Use Python's standard `logging` module for any logging.
- Log metrics by POSTing to the ASTRA telemetry endpoint:
    POST {api_url}/telemetry/missions/{mission_id}/metrics
    Body: {{"mission_id": "...", "name": "...", "value": 0.0, "step": 0}}
- Save checkpoints to: {checkpoint_dir}
- On error, print the full traceback and exit with code 1.
- DO NOT use markdown code blocks (```python ... ```).
- Return ONLY the raw Python script, no explanation, no preamble, no stop tokens."""

# Injected verbatim at the top of Snake training scripts
_SNAKE_SETUP = """\
import sys as _sys
_sys.path.insert(0, "{project_root}")
from envs.snake_env import register as _register_snake
_register_snake()
"""

_RL_TEMPLATE = """\
Generate a complete RL training script using Stable-Baselines3.
{snake_setup}
Mission ID: {mission_id}
Algorithm: {algorithm}
Environment: {env_id}
Hyperparameters: {hyperparameters}
Target metric: {target_metric}
Checkpoint directory: {checkpoint_dir}
Telemetry URL: {api_url}/telemetry/missions/{mission_id}/metrics

The script MUST start with these exact imports (do not omit any):
    import gymnasium as gym
    import numpy as np
    import requests
    import logging

The script must:
1. Create the environment with EXACTLY these two lines — copy verbatim:
       import gymnasium as gym
       env = gym.make("{env_id}")
   The env_id is "{env_id}". Do NOT use "CartPole-v1" or any other env name.
   Do NOT read it from hyperparameters. Hard-code "{env_id}" in gym.make().
2. Instantiate the {algorithm} model passing ONLY these valid SB3 PPO kwargs
   (filter out anything else from the hyperparameters dict before passing):
   learning_rate, n_steps, batch_size, n_epochs, gamma, gae_lambda,
   clip_range, clip_range_vf, ent_coef, vf_coef, max_grad_norm, target_kl.
   DO NOT pass: actor_lr, critic_lr, entropy_coef, entropy_coeff,
   clip_range_value, or any other key not in the list above.
3. Implement a custom BaseCallback. Inside _on_step use EXACTLY this pattern —
   do NOT post telemetry on every step, only every 2048 steps:

       def _on_step(self) -> bool:
           if self.n_calls % 2048 == 0 and len(self.model.ep_info_buffer) > 0:
               mean_reward = float(np.mean([ep["r"] for ep in self.model.ep_info_buffer]))
               try:
                   response = requests.post(
                       "{api_url}/telemetry/missions/{mission_id}/metrics",
                       json={{"mission_id": "{mission_id}", "name": "mean_reward",
                              "value": mean_reward, "step": self.n_calls}},
                       timeout=2,
                   )
                   if not response.ok:
                       logger.warning("Telemetry failed: %s", response.status_code)
               except Exception as exc:
                   logger.warning("Telemetry error: %s", exc)
               if mean_reward >= {target_reward}:
                   return False  # stop training — target reached
           return True

   The `self.n_calls % 2048 == 0` guard is MANDATORY. Never remove it.
4. Call model.learn(total_timesteps=500000, callback=callback) — use at least
   500000 timesteps. Do NOT use 10000 or any small number.
5. Save a checkpoint when mean_reward improves.
6. Exit cleanly when target mean_reward is reached."""

_SFT_TEMPLATE = """\
Generate a complete SFT (QLoRA) fine-tuning script using HuggingFace + PEFT.

Mission ID: {mission_id}
Base model: {base_model}
Dataset path: {dataset_path}
LoRA config: r={lora_r}, alpha={lora_alpha}, dropout={lora_dropout}
Training args: batch={batch_size}, lr={learning_rate}, epochs={num_epochs}
save_strategy: steps, save_steps: {save_steps}
Checkpoint directory: {checkpoint_dir}
Telemetry URL: {api_url}/telemetry/missions/{mission_id}/metrics

The script must:
1. Load the base model in 4-bit (BitsAndBytes).
2. Apply LoRA via peft.get_peft_model.
3. Load the dataset from {dataset_path}.
4. Train using trl.SFTTrainer.
5. POST eval_loss to the telemetry endpoint after each save step.
6. Exit cleanly when eval_loss ≤ target."""

_ML_TEMPLATE = """\
Generate a complete ML training script.

Mission ID: {mission_id}
Framework: {framework}
Model class: {model_class}
Dataset path: {dataset_path}
Target column: {target_column}
Model params: {model_params}
Checkpoint directory: {checkpoint_dir}
Telemetry URL: {api_url}/telemetry/missions/{mission_id}/metrics

The script must:
1. Load the dataset:
   - If {dataset_path} is a well-known sklearn dataset name ("iris", "digits",
     "breast_cancer", "wine"), call sklearn.datasets.load_{dataset_path}() and use
     the returned .data and .target arrays directly as X and y. Do NOT convert to a
     DataFrame and do NOT import pandas or numpy for this step.
   - Otherwise load from {dataset_path} using pandas (CSV) or json.
2. Split into train/val sets using train_test_split(X, y, ...).
3. Train the model.
4. POST accuracy to the telemetry endpoint with metric name "accuracy".
   Use response.ok (2xx) to check success; log a warning on failure but do NOT exit — telemetry is non-critical.
5. Save the model with joblib.
6. Exit cleanly (exit(0) on success, exit(1) on error with traceback)."""


class CodeGenerator:
    def __init__(self, provider: InferenceProvider) -> None:
        self._provider = provider

    async def generate_training_script(
        self,
        mission_id: str,
        plan: dict,
    ) -> str:
        """
        Generate a training script and write it to data/missions/{id}/train.py.
        Returns the path to the written script.
        """
        task_type = plan.get("task_type", "rl")
        if settings.sandbox_host:
            checkpoint_dir = os.path.join(
                settings.sandbox_data_path, "missions", mission_id, "checkpoints"
            )
        else:
            checkpoint_dir = os.path.join(settings.data_path, "missions", mission_id, "checkpoints")
            os.makedirs(checkpoint_dir, exist_ok=True)

        system_prompt = _BASE_SYSTEM.format(
            api_url=f"http://{settings.telemetry_host}:{settings.api_port}",
            mission_id=mission_id,
            checkpoint_dir=checkpoint_dir,
        )

        # Inject lessons learned from prior code generation failures
        lessons = self._query_lessons(plan)
        if lessons:
            lesson_block = "\n".join(f"- {l}" for l in lessons)
            system_prompt += f"\n\nLessons learned from prior failures (avoid repeating these):\n{lesson_block}"

        user_prompt = self._build_user_prompt(task_type, mission_id, plan, checkpoint_dir)

        messages = [
            Message(role="system", content=system_prompt),
            Message(role="user", content=user_prompt),
        ]
        code = await self._provider.generate(messages, GenerationConfig(max_tokens=4096, temperature=0.1))
        code = self._strip_fences(code)
        if task_type == "rl":
            code = self._patch_rl_imports(code)

        script_path = os.path.abspath(os.path.join(settings.data_path, "missions", mission_id, "train.py"))
        os.makedirs(os.path.dirname(script_path), exist_ok=True)
        with open(script_path, "w") as f:
            f.write(code)

        logger.info("Generated training script: %s (%d chars)", script_path, len(code))
        return script_path

    def _build_user_prompt(self, task_type: str, mission_id: str, plan: dict, checkpoint_dir: str) -> str:
        hp = plan.get("hyperparameters", {})
        api_url = f"http://127.0.0.1:{settings.api_port}"
        base = {
            "mission_id": mission_id,
            "checkpoint_dir": checkpoint_dir,
            "api_url": api_url,
            "target_metric": json.dumps(plan.get("target_metric", {})),
        }
        if task_type == "rl":
            tm = plan.get("target_metric", {})
            target_reward = next(iter(tm.values()), 200) if tm else 200
            env_id = plan.get("env_id", "CartPole-v1")
            # Inject Snake registration preamble when the env is Snake-v0
            snake_setup = (
                _SNAKE_SETUP.format(project_root=os.path.abspath(
                    os.path.join(settings.data_path, "..")
                ))
                if env_id == "Snake-v0" else ""
            )
            ctx = {
                "algorithm": plan.get("algorithm", "PPO"),
                "env_id": env_id,
                "hyperparameters": json.dumps(hp, indent=2),
                "target_reward": target_reward,
                "snake_setup": snake_setup,
                **hp,
                **base,
            }
            return _RL_TEMPLATE.format(**ctx)
        if task_type == "sft":
            ctx = {
                "base_model": "meta-llama/Llama-3.1-8B",
                "dataset_path": "dataset.jsonl",
                "lora_r": 16,
                "lora_alpha": 32,
                "lora_dropout": 0.05,
                "batch_size": 4,
                "learning_rate": 2e-4,
                "num_epochs": 3,
                "save_steps": 200,
                **hp,
                **base,
            }
            return _SFT_TEMPLATE.format(**ctx)
        # ml
        ctx = {
            "framework": "sklearn",
            "model_class": "RandomForestClassifier",
            "dataset_path": "dataset.csv",
            "target_column": "label",
            "model_params": json.dumps(hp.get("model_params", {}), indent=2),
            **hp,
            **base,
        }
        return _ML_TEMPLATE.format(**ctx)

    @staticmethod
    def _patch_rl_imports(code: str) -> str:
        """Inject any SB3 imports the LLM forgot to include."""
        lines = code.splitlines()
        last_import_idx = 0
        for i, line in enumerate(lines):
            s = line.strip()
            if s.startswith("import ") or s.startswith("from "):
                last_import_idx = i

        to_inject = []
        for algo in ("PPO", "SAC", "A2C", "DQN", "TD3"):
            if algo in code and f"from stable_baselines3 import {algo}" not in code:
                to_inject.append(f"from stable_baselines3 import {algo}")
        if "BaseCallback" in code and "from stable_baselines3.common.callbacks import BaseCallback" not in code:
            to_inject.append("from stable_baselines3.common.callbacks import BaseCallback")

        import re
        if to_inject:
            insert_at = last_import_idx + 1
            for i, imp in enumerate(to_inject):
                lines.insert(insert_at + i, imp)
            code = "\n".join(lines)
        # Fix callback constructor calls with unsupported kwargs
        code = re.sub(
            r'(\w*[Cc]allback\w*)\s*\(\s*(?:check_freq|checkpoint_freq|save_path)[^)]*\)',
            r'\1()',
            code,
        )
        return code

    @staticmethod
    def _query_lessons(plan: dict) -> list[str]:
        """Retrieve past code generation failure lessons from vector memory."""
        try:
            from backend.services import vector_memory
            domain = plan.get("domain", "code_generation")
            results = vector_memory.query_lessons(
                f"{plan.get('algorithm', '')} {plan.get('task_type', '')} training script",
                domain=domain,
                n_results=3,
            )
            return [r["text"] for r in results if r.get("text")]
        except Exception:
            return []

    @staticmethod
    def _strip_fences(text: str) -> str:
        import re
        # Remove common LLM artifacts
        text = re.sub(r"<\|im_end\|>|<\|endoftext\|>", "", text)
        # Remove lines that are purely markdown fences or artifacts
        lines = []
        for line in text.splitlines():
            if line.strip().startswith("```"):
                continue
            lines.append(line)
        return "\n".join(lines).strip()
