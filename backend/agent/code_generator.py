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

# Injected verbatim at the top of custom-env training scripts
_SNAKE_SETUP = """\
import sys as _sys
_sys.path.insert(0, "{project_root}")
from envs.snake_env import register as _register_snake
_register_snake()
"""

_TETRIS_SETUP = """\
import sys as _sys
_sys.path.insert(0, "{project_root}")
import gymnasium as gym
from envs.tetris_env import register as _register_tetris
_register_tetris()
"""

_RL_TEMPLATE = """\
Generate a complete RL training script using Stable-Baselines3.
{env_setup}
Mission ID: {mission_id}
Algorithm: {algorithm}
Environment: {env_id}
Hyperparameters: {hyperparameters}
Policy kwargs (network architecture): {policy_kwargs}
Target metric: {target_metric}
Checkpoint directory: {checkpoint_dir}
Telemetry URL: {api_url}/telemetry/missions/{mission_id}/metrics

The script MUST start with these exact imports (do not omit any):
    import os
    import gymnasium as gym
    import numpy as np
    import requests
    import logging
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback

The script must:
1. Create the environment with EXACTLY these lines — copy verbatim:
       import gymnasium as gym
       env = gym.make("{env_id}"{env_kwargs_str})
   The env_id is "{env_id}". Do NOT use "CartPole-v1" or any other env name.
   Do NOT read it from hyperparameters. Hard-code "{env_id}" in gym.make().
2. The script MUST use these EXACT lines to construct the model — copy verbatim, do NOT change any values:

       _VALID_PPO_KEYS = {{"learning_rate", "n_steps", "batch_size", "n_epochs", "gamma",
                           "gae_lambda", "clip_range", "clip_range_vf", "ent_coef",
                           "vf_coef", "max_grad_norm", "target_kl"}}
       _hp = {hyperparameters}
       _filtered = {{k: v for k, v in _hp.items() if k in _VALID_PPO_KEYS}}
       _policy_kwargs = {policy_kwargs}
       model = PPO("MlpPolicy", env, **_filtered,
                   **(dict(policy_kwargs=_policy_kwargs) if _policy_kwargs else {{}}))

   These values are set by the optimizer — do NOT substitute your own hyperparameter values.
3. Immediately after constructing the model, copy this warm-start block EXACTLY — do not modify:

       _best_ckpt = "{checkpoint_dir}/best_model.zip"
       if os.path.exists(_best_ckpt):
           try:
               _warm = PPO.load(_best_ckpt, env=env)
               model.policy.load_state_dict(_warm.policy.state_dict())
               del _warm
           except Exception as _e:
               logging.warning("Warm-start skipped (architecture mismatch or load error): %s", _e)

   This resumes training from the best previously saved weights while keeping the new hyperparameters.
   If the checkpoint architecture differs (e.g. after a net_arch pivot), the except branch silently
   falls back to random weights. `os` and `logging` are already imported.
   The block is MANDATORY — do not remove or skip it.
4. Implement a custom BaseCallback. Copy this ENTIRE class EXACTLY — do not add, remove, or
   modify any line:

       class CustomCallback(BaseCallback):
           def __init__(self, verbose=0):
               super().__init__(verbose=verbose)
               try:
                   self._best_reward = float(open("{checkpoint_dir}/best_score.txt").read().strip())
               except Exception:
                   self._best_reward = float("-inf")

           def _on_step(self) -> bool:
               if self.n_calls % {telemetry_interval} == 0 and len(self.model.ep_info_buffer) > 0:
                   mean_reward = float(np.mean([ep["r"] for ep in self.model.ep_info_buffer]))
                   try:
                       response = requests.post(
                           "{api_url}/telemetry/missions/{mission_id}/metrics",
                           json={{"mission_id": "{mission_id}", "name": "mean_reward",
                                  "value": mean_reward, "step": self.n_calls,
                                  "iteration": {current_iteration}}},
                           timeout=2,
                       )
                       if not response.ok:
                           logging.warning("Telemetry failed: %s", response.status_code)
                   except Exception as exc:
                       logging.warning("Telemetry error: %s", exc)
                   if mean_reward > self._best_reward:
                       self._best_reward = mean_reward
                       self.model.save("{checkpoint_dir}/best_model")
                       with open("{checkpoint_dir}/best_score.txt", "w") as _f:
                           _f.write(str(mean_reward))
                       with open("{checkpoint_dir}/best_model_algo.txt", "w") as _f:
                           _f.write(self.model.__class__.__name__)
                   if mean_reward >= {target_reward}:
                       return False  # stop training — target reached
               return True

   The `self.n_calls % {telemetry_interval} == 0` guard is MANDATORY. Never remove it.
   The best_model save block is MANDATORY — it ensures the peak model is preserved.
   The __init__ loading from best_score.txt is MANDATORY — it preserves peak weights across restarts.
5. Call model.learn(total_timesteps={total_timesteps}, callback=callback).
6. After training, save the final model: model.save("{checkpoint_dir}/last_model")
7. Exit cleanly when target mean_reward is reached."""

_ACTOR_CRITIC_CONTRACT = """\
Generate a complete custom Actor-Critic training script for Tetris using PyTorch.
Do NOT use Stable-Baselines3. Implement everything with PyTorch and plain Python.

{env_setup}
Mission ID: {mission_id}
Environment: {env_id}
Checkpoint directory: {checkpoint_dir}
Telemetry URL: {api_url}/telemetry/missions/{mission_id}/metrics
Target: lines_cleared >= {target_lines}
Hyperparameters: {hyperparameters}

== Architecture ==
Import the canonical class — do NOT redefine it inline:
    from envs.actor_critic_net import ActorCriticNet
This ensures torch.load can resolve the class in any process (play, benchmark, eval).
model = ActorCriticNet()  # shared MLP [4→64→64] + critic head Linear(64,1)

== Training skeleton (follow exactly — do NOT deviate from these API calls) ==
BUFFER = collections.deque(maxlen={replay_buffer_size})
ep_rewards, ep_lines = [], []
epsilon = 1.0
best_reward = float("-inf")

for episode in range({episodes}):
    obs, _ = env.reset()          # gymnasium reset → (obs, info); unpack both
    ep_reward = 0.0
    ep_lines_cleared = 0
    done = False

    while not done:
        next_states = env.unwrapped.get_next_states()  # dict{{action: np.array(4,)}}
        if not next_states:
            action = 0
        elif random.random() < epsilon:
            action = random.choice(list(next_states.keys()))
        else:
            with torch.no_grad():
                action = max(
                    next_states,
                    key=lambda a: model(
                        torch.tensor(next_states[a], dtype=torch.float32).unsqueeze(0)
                    ).item()
                )

        next_obs, reward, terminated, truncated, info = env.step(action)  # 5-tuple
        done = terminated or truncated
        BUFFER.append((obs, action, reward, next_obs, float(done)))
        obs = next_obs
        ep_reward += reward
        if done:
            ep_lines_cleared = info.get("lines_cleared", 0)

        if len(BUFFER) >= {batch_size}:
            batch = random.sample(BUFFER, {batch_size})
            s, _, r, ns, d = zip(*batch)
            s  = torch.tensor(s,  dtype=torch.float32)
            ns = torch.tensor(ns, dtype=torch.float32)
            r  = torch.tensor(r,  dtype=torch.float32).unsqueeze(1)
            d  = torch.tensor(d,  dtype=torch.float32).unsqueeze(1)
            with torch.no_grad():
                td_target = r + {gamma} * model(ns) * (1 - d)
            loss = nn.MSELoss()(model(s), td_target)
            optimizer.zero_grad(); loss.backward(); optimizer.step()

    epsilon = max({epsilon_min}, epsilon * {epsilon_decay})
    ep_rewards.append(ep_reward)
    ep_lines.append(ep_lines_cleared)

    if len(ep_rewards) >= {ac_telemetry_interval} and (episode + 1) % {ac_telemetry_interval} == 0:
        mean_reward_50 = float(np.mean(ep_rewards[-{ac_telemetry_interval}:]))
        mean_lines_50  = float(np.mean(ep_lines[-{ac_telemetry_interval}:]))
        # telemetry POSTs here (catch all exceptions)
        if mean_reward_50 > best_reward:
            best_reward = mean_reward_50
            # save best model here

== ASTRA integration contract (ALL items MANDATORY) ==
1. Write this file once at startup (so play/eval endpoints detect the trainer):
     open("{checkpoint_dir}/trainer_type.txt", "w").write("actor_critic")
2. Warm-start: if "{checkpoint_dir}/best_model.pth" exists load with:
     model = torch.load("{checkpoint_dir}/best_model.pth", weights_only=False)
3. Track rolling mean_reward over last {ac_telemetry_interval} episodes.
   Every {ac_telemetry_interval} episodes POST to telemetry:
     POST {api_url}/telemetry/missions/{mission_id}/metrics
     json={{"mission_id": "{mission_id}", "name": "mean_reward",
            "value": mean_reward_50, "step": episode, "iteration": {current_iteration}}}
     AND post lines_cleared (mean over last {ac_telemetry_interval} episodes) with name "lines_cleared".
   Use timeout=2, catch all exceptions (telemetry is non-critical).
4. When rolling mean_reward improves over best seen so far:
     torch.save(model, "{checkpoint_dir}/best_model.pth")
     open("{checkpoint_dir}/best_score.txt", "w").write(str(best_reward))
     open("{checkpoint_dir}/best_model_algo.txt", "w").write("ActorCritic")
5. After all episodes: torch.save(model, "{checkpoint_dir}/last_model.pth")

Return ONLY the raw Python script. No markdown fences, no explanation."""

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
5. Save the model using this EXACT line — do not change the path:
       import joblib; joblib.dump(model, "{checkpoint_dir}/model.joblib")
6. Exit cleanly (exit(0) on success, exit(1) on error with traceback)."""


def _resolve_env_kwargs(env_id: str, plan_env_kwargs: Optional[dict]) -> dict:
    """Merge plan env_kwargs with per-env hardcoded defaults."""
    kw = dict(plan_env_kwargs or {})
    if env_id == "Snake-v0":
        # Always use compact feature obs — the flat 256D grid is a poor inductive
        # bias for MLP policies; 25D feature obs learns faster and generalises better.
        kw.setdefault("obs_type", "features")
        kw.setdefault("max_steps", 2000)
    return kw


class CodeGenerator:
    def __init__(self, provider: InferenceProvider) -> None:
        self._provider = provider

    async def generate_training_script(
        self,
        mission_id: str,
        plan: dict,
        current_iteration: int = 0,
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

        user_prompt = self._build_user_prompt(task_type, mission_id, plan, checkpoint_dir, current_iteration)

        messages = [
            Message(role="system", content=system_prompt),
            Message(role="user", content=user_prompt),
        ]
        code = await self._provider.generate(messages, GenerationConfig(max_tokens=4096, temperature=0.1))
        code = self._strip_fences(code)
        if task_type == "rl":
            code = self._patch_rl_imports(code)
            code = self._patch_undefined_logger(code)
            env_id = plan.get("env_id", "")
            _proj_root = os.path.abspath(os.path.join(settings.data_path, ".."))
            if env_id == "Snake-v0" and "register" not in code:
                code = _SNAKE_SETUP.format(project_root=_proj_root) + "\n" + code
                logger.info("CodeGenerator: injected Snake-v0 registration preamble")
            elif env_id == "Tetris-v0" and "register" not in code:
                code = _TETRIS_SETUP.format(project_root=_proj_root) + "\n" + code
                logger.info("CodeGenerator: injected Tetris-v0 registration preamble")
        # Fix any relative checkpoint paths the LLM may have substituted for the absolute checkpoint_dir
        code = self._fix_checkpoint_paths(code, checkpoint_dir)

        script_path = os.path.abspath(os.path.join(settings.data_path, "missions", mission_id, "train.py"))
        os.makedirs(os.path.dirname(script_path), exist_ok=True)
        with open(script_path, "w") as f:
            f.write(code)

        # Write train_config.json so the play endpoint knows the algorithm and env_kwargs
        if task_type == "rl":
            import json as _json
            config_path = os.path.join(checkpoint_dir, "train_config.json")
            os.makedirs(checkpoint_dir, exist_ok=True)
            _env_id = plan.get("env_id", "")
            train_cfg = {
                "algorithm": plan.get("algorithm", "PPO"),
                "env_id": _env_id,
                "env_kwargs": _resolve_env_kwargs(_env_id, plan.get("env_kwargs")),
                "trainer_type": plan.get("trainer_type", ""),
            }
            with open(config_path, "w") as f:
                _json.dump(train_cfg, f)
            logger.info("CodeGenerator: wrote train_config.json algorithm=%s env_kwargs=%s trainer_type=%s",
                        train_cfg["algorithm"], train_cfg["env_kwargs"], train_cfg["trainer_type"])

        logger.info("Generated training script: %s (%d chars)", script_path, len(code))
        return script_path

    def _build_user_prompt(self, task_type: str, mission_id: str, plan: dict, checkpoint_dir: str, current_iteration: int = 0) -> str:
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
            tm_name = next(iter(tm), None) if tm else None
            tm_value = next(iter(tm.values()), 200) if tm else 200
            # Only use the target value as the mean_reward threshold when the
            # target metric IS mean_reward; for custom goals (food_eaten, etc.)
            # mean_reward scale is unrelated to the target — use a sentinel that
            # never triggers so the full timestep budget always runs.
            target_reward = tm_value if tm_name in (None, "mean_reward") else 9999
            env_id = plan.get("env_id", "CartPole-v1")
            # Inject custom env registration preamble when needed
            _project_root = os.path.abspath(os.path.join(settings.data_path, ".."))
            if env_id == "Snake-v0":
                env_setup = _SNAKE_SETUP.format(project_root=_project_root)
            elif env_id == "Tetris-v0":
                env_setup = _TETRIS_SETUP.format(project_root=_project_root)
            else:
                env_setup = ""
            hp = dict(hp)  # copy so we don't mutate the plan's hyperparameters dict
            policy_kwargs = hp.pop("policy_kwargs", None)
            # Build env_kwargs_str: ", key=value, ..." for gym.make() call
            env_kwargs = _resolve_env_kwargs(env_id, plan.get("env_kwargs"))
            if env_kwargs:
                env_kwargs_str = ", " + ", ".join(f"{k}={v!r}" for k, v in env_kwargs.items())
            else:
                env_kwargs_str = ""
            trainer_type = plan.get("trainer_type", "")
            if trainer_type == "actor_critic":
                tm_lines = next(iter(tm.values()), 20) if tm else 20
                lr = hp.get("learning_rate", 0.0001)
                episodes = hp.get("episodes", 10000)
                ctx = {
                    "env_id": env_id,
                    "env_setup": env_setup,
                    "target_lines": tm_lines,
                    "hyperparameters": json.dumps(hp, indent=2),
                    "lr": lr,
                    "episodes": episodes,
                    "replay_buffer_size": hp.get("replay_buffer_size", 10000),
                    "batch_size": hp.get("batch_size", 64),
                    "gamma": hp.get("gamma", 0.99),
                    "epsilon_min": hp.get("epsilon_min", 0.01),
                    "epsilon_decay": hp.get("epsilon_decay", 0.9995),
                    "ac_telemetry_interval": hp.get("ac_telemetry_interval", 50),
                    "current_iteration": current_iteration,
                    **base,
                }
                return _ACTOR_CRITIC_CONTRACT.format(**ctx)
            ctx = {
                "algorithm": plan.get("algorithm", "PPO"),
                "env_id": env_id,
                "hyperparameters": json.dumps(hp, indent=2),
                "policy_kwargs": json.dumps(policy_kwargs) if policy_kwargs else "None",
                "target_reward": target_reward,
                "env_setup": env_setup,
                "env_kwargs_str": env_kwargs_str,
                "current_iteration": current_iteration,
                "total_timesteps": hp.get("total_timesteps", 2000000),
                "telemetry_interval": hp.get("telemetry_interval", 2048),
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
        import re
        # Replace `import stable_baselines3 as sb3` + `sb3.XXX` with direct imports
        if "import stable_baselines3 as sb3" in code:
            code = re.sub(r'\bsb3\.(PPO|SAC|A2C|DQN|TD3)\b', r'\1', code)
            code = re.sub(r'^\s*import stable_baselines3 as sb3\s*\n?', '', code, flags=re.MULTILINE)

        # Re-split after any string-level substitution above
        lines = code.splitlines()
        last_import_idx = 0
        for i, line in enumerate(lines):
            s = line.strip()
            if s.startswith("import ") or s.startswith("from "):
                last_import_idx = i

        import_lines = "\n".join(l for l in lines if l.strip().startswith(("import ", "from ")))
        to_inject = []
        if "import os" not in import_lines:
            to_inject.append("import os")
        for algo in ("PPO", "SAC", "A2C", "DQN", "TD3"):
            if algo in code and f"from stable_baselines3 import {algo}" not in import_lines:
                to_inject.append(f"from stable_baselines3 import {algo}")
        if "BaseCallback" in code and "from stable_baselines3.common.callbacks import BaseCallback" not in import_lines:
            to_inject.append("from stable_baselines3.common.callbacks import BaseCallback")
        if "CheckpointCallback" in code and "CheckpointCallback" not in import_lines:
            to_inject.append("from stable_baselines3.common.callbacks import CheckpointCallback")

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
        # Fix callback class that has _on_step but doesn't extend BaseCallback
        code = re.sub(
            r'(class\s+\w*[Cc]allback\w*)\s*:',
            r'\1(BaseCallback):',
            code,
        )
        # Ensure __init__ accepts **kwargs to tolerate extra args
        if "class " in code and "(BaseCallback)" in code:
            if "def __init__(self):" in code:
                code = code.replace(
                    "def __init__(self):",
                    "def __init__(self, verbose=0, **kwargs):\n        super().__init__(verbose=verbose)",
                )
        return code

    @staticmethod
    def _patch_undefined_logger(code: str) -> str:
        """Replace bare logger.warning/error/info calls with logging.* when no logger is defined.
        LLMs frequently emit `logger.warning(...)` without defining `logger = logging.getLogger(...)`.
        """
        import re
        if re.search(r'\blogger\s*=', code):
            return code  # logger is explicitly defined — leave as-is
        # Replace logger.<level>(...) → logging.<level>(...)
        return re.sub(r'\blogger\.(warning|error|info|debug|critical)\b', r'logging.\1', code)

    @staticmethod
    def _fix_checkpoint_paths(code: str, checkpoint_dir: str) -> str:
        """Replace relative ./data/missions/.../checkpoints paths with the absolute checkpoint_dir."""
        import re
        # Match patterns like ./data/missions/<uuid>/checkpoints or data/missions/<uuid>/checkpoints
        pattern = r'\.?/?\bdata/missions/[0-9a-f-]{36}/checkpoints\b'
        fixed = re.sub(pattern, checkpoint_dir, code)
        if fixed != code:
            logger.info("CodeGenerator: replaced relative checkpoint paths with absolute: %s", checkpoint_dir)
        return fixed

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
