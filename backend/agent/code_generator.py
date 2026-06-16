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
- Log metrics by POSTing to the ASTRA telemetry endpoint:
    POST {api_url}/telemetry/missions/{mission_id}/metrics
    Body: {{"mission_id": "...", "name": "...", "value": 0.0, "step": 0}}
- Save checkpoints to: {checkpoint_dir}
- On error, print the full traceback and exit with code 1.
- DO NOT use markdown code blocks (```python ... ```).
- Return ONLY the raw Python script, no explanation, no preamble, no stop tokens."""

_RL_TEMPLATE = """\
Generate a complete RL training script using Stable-Baselines3.

Mission ID: {mission_id}
Algorithm: {algorithm}
Environment: {env_id}
Hyperparameters: {hyperparameters}
Target metric: {target_metric}
Checkpoint directory: {checkpoint_dir}
Telemetry URL: {api_url}/telemetry/missions/{mission_id}/metrics

The script must:
1. Create the environment with EXACTLY: env = gym.make("{env_id}")
   Do NOT read the environment name from hyperparameters or any variable —
   hard-code the string "{env_id}" directly in the gym.make() call.
2. Instantiate the {algorithm} model passing ONLY these valid SB3 PPO kwargs
   (filter out anything else from the hyperparameters dict before passing):
   learning_rate, n_steps, batch_size, n_epochs, gamma, gae_lambda,
   clip_range, clip_range_vf, ent_coef, vf_coef, max_grad_norm, target_kl.
   DO NOT pass: actor_lr, critic_lr, entropy_coef, entropy_coeff,
   clip_range_value, or any other key not in the list above.
3. Implement a custom BaseCallback that computes mean_reward from
   self.model.ep_info_buffer — NOT from self.locals (which does not contain
   mean_reward). Use this pattern inside _on_step:
       if len(self.model.ep_info_buffer) > 0:
           mean_reward = float(np.mean([ep["r"] for ep in self.model.ep_info_buffer]))
4. Every 2048 steps (check with self.n_calls % 2048 == 0), POST mean_reward
   to the telemetry endpoint. Use response.ok to check success; log a warning
   on failure but do NOT exit — telemetry is non-critical.
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
            ctx = {
                "algorithm": plan.get("algorithm", "PPO"),
                "env_id": "CartPole-v1",
                "hyperparameters": json.dumps(hp, indent=2),
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
