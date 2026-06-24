"""
LeadAgent — Step 3.1 orchestrator.

Wraps the inference provider with:
- System prompts for strategic goal decomposition
- Structured output parsing (JSON schema retry on parse error)
- Smart KV cache for context management
- Prefix caching for real-time log analysis
"""
from __future__ import annotations

import json
import platform
from typing import Optional

from backend.agent.inference.base import InferenceProvider, Message, GenerationConfig
from backend.agent.kv_cache import SmartKVCache
from backend.logging_config import get_logger

logger = get_logger(__name__)

_PLANNING_SYSTEM = """\
You are ASTRA's Lead Agent — an autonomous ML training strategist.
Your job is to decompose a high-level training goal into a concrete plan.

Always respond with valid JSON. Think step by step before committing to a plan.
Consider: task type (rl/sft/ml/mlx_lora), algorithm selection, hyperparameters, curriculum phases,
and how you will measure success against the target metric.

For ml tasks, always include "dataset_path" in hyperparameters. Use the sklearn dataset name
(e.g. "iris", "digits", "breast_cancer", "wine") for built-in datasets, or a file path for
custom datasets.

For mlx_lora tasks (Apple Silicon MLX fine-tuning), include "dataset" as a top-level field
with "train" and "valid" JSONL paths. Hyperparameters: base_model, lora_rank, lora_scale,
lora_dropout, num_layers, batch_size, learning_rate, iters, mask_prompt.

For rl tasks, always include "env_id" as a top-level field in the plan (NOT in hyperparameters).
Available environments:
  - Standard gymnasium: "CartPole-v1", "LunarLander-v3", "Acrobot-v1", "MountainCar-v0"
  - Custom ASTRA env: "Snake-v0" (16×16 grid, discrete 4-action, food reward +10, death -10)
    Use "Snake-v0" when the goal mentions Snake or a grid-based game.
Valid SB3 PPO hyperparameter keys: learning_rate, n_steps, batch_size, n_epochs, gamma,
gae_lambda, clip_range, clip_range_vf, ent_coef, vf_coef, max_grad_norm, target_kl.
Do NOT include env_id, dataset_path, entropy_coeff, actor_lr, or any non-SB3 key in hyperparameters."""

_PIVOT_SYSTEM = """\
You are ASTRA's Lead Agent analyzing a training run that has stalled or plateaued.
Given the current metrics, training history, and escalation level, propose a strategic pivot.
Respond with valid JSON.

Escalation levels — follow the level provided in the user message:
  Level 0 (first plateau): tune hyperparameters only. Small changes — adjust learning_rate,
    batch_size, gamma, ent_coef, etc. Keep the same algorithm and architecture.
  Level 1 (repeated plateau): change the policy network architecture in addition to HPs.
    Use "policy_kwargs" with a "net_arch" from: [256, 256], [400, 300], or [256, 256, 128].
    IMPORTANT: if a "Best performing architecture" is listed below, reuse it — do NOT switch
    to a different architecture. Only deviate if the best arch is identical to the current one.
  Level 2 (stuck for many pivots): switch to a fundamentally different algorithm if the
    current one is not working. Set "algorithm" to "PPO", "SAC", "A2C", or "DQN".
    For Snake-v0, PPO with [256, 256] net_arch is strongly recommended over DQN.
    Also update hyperparameters to suit the new algorithm.
  Level 3 (deeply stuck): reshape the reward function via "env_kwargs". For Snake-v0:
    - Disable distance shaping (set distance_weight=0) to prevent greedy body collision
    - Increase food_reward (e.g. 20.0) to make food-seeking the dominant signal
    - Adjust survival_bonus (e.g. 0.05) and death_penalty (e.g. -5.0)
    env_kwargs example: {"food_reward": 20.0, "death_penalty": -5.0, "distance_weight": 0.0, "survival_bonus": 0.05}

PPO hyperparameter ranges:
  learning_rate: 1e-5 to 1e-2 | n_steps: 1024–4096 | batch_size: 64–512
  n_epochs: 3–20 | gamma: 0.90–0.999 | gae_lambda: 0.80–0.99
  clip_range: 0.1–0.4 | ent_coef: 0.0–0.1 | vf_coef: 0.1–1.0 | max_grad_norm: 0.3–1.0

DQN hyperparameter ranges:
  learning_rate: 1e-5 to 1e-3 | batch_size: 32–256 | gamma: 0.90–0.999
  exploration_fraction: 0.1–0.5 | exploration_final_eps: 0.01–0.1
  target_update_interval: 500–10000 | learning_starts: 1000–50000

Do NOT set n_epochs > 20 or n_steps < 512 for PPO — these destabilize training."""

_PLAN_SCHEMA = {
    "type": "object",
    "properties": {
        "plan": {
            "type": "object",
            "properties": {
                "task_type": {"type": "string", "enum": ["rl", "sft", "ml", "mlx_lora"]},
                "algorithm": {"type": "string"},
                "env_id": {"type": "string"},
                "hyperparameters": {"type": "object"},
                "curriculum_phases": {"type": "array"},
                "estimated_iterations": {"type": "integer"},
                "reasoning": {"type": "string"},
            },
            "required": ["task_type", "algorithm", "hyperparameters"],
        }
    },
    "required": ["plan"],
}

_PIVOT_SCHEMA = {
    "type": "object",
    "properties": {
        "pivot": {
            "type": "object",
            "properties": {
                "reason": {"type": "string"},
                "adjustments": {"type": "object"},
                "policy_kwargs": {"type": "object"},
                "algorithm": {"type": "string"},
                "env_kwargs": {"type": "object"},
                "reasoning": {"type": "string"},
            },
            "required": ["reason", "adjustments"],
        }
    },
    "required": ["pivot"],
}


class LeadAgent:
    def __init__(self, provider: InferenceProvider) -> None:
        self._provider = provider
        self._cache = SmartKVCache()
        self._cache.set_system_prompt(_PLANNING_SYSTEM)

    # ── Planning ──────────────────────────────────────────────────────────────

    async def plan(self, goal: str, task_type: str, target_metric: dict) -> dict:
        """
        Decompose a user goal into a structured training plan.
        Checks the recipe library for a warm-start hint before planning.
        Returns the plan dict (JSON-parsed).
        """
        warm_hint = self._get_warm_start_hint(goal, task_type)
        query = (
            f"Goal: {goal}\n"
            f"Task type: {task_type}\n"
            f"Target metric: {json.dumps(target_metric)}\n"
            + (f"Warm-start hint (best matching past recipe): {json.dumps(warm_hint)}\n" if warm_hint else "")
            + "\nDesign the optimal training plan. Return JSON."
        )
        messages = self._cache.get_messages(query)
        response = await self._generate_structured(messages, _PLAN_SCHEMA)
        self._cache.add_turn("user", query)
        self._cache.add_turn("assistant", json.dumps(response))
        logger.info("LeadAgent plan generated: task=%s algorithm=%s",
                    response.get("plan", {}).get("task_type"),
                    response.get("plan", {}).get("algorithm"))
        return response.get("plan", response)

    # ── Pivot ─────────────────────────────────────────────────────────────────

    async def propose_pivot(
        self,
        current_metrics: dict,
        history: list[dict],
        escalation_level: int = 0,
        current_algorithm: str = "PPO",
        algorithm_locked: bool = False,
        current_policy_kwargs: Optional[dict] = None,
        current_hyperparameters: Optional[dict] = None,
        current_env_kwargs: Optional[dict] = None,
        best_policy_kwargs: Optional[dict] = None,
        best_metric_value: Optional[float] = None,
        best_metric_iteration: Optional[int] = None,
    ) -> dict:
        """
        Analyze a stalled run and propose a strategic pivot.
        escalation_level: 0=tune HPs, 1=change arch, 2=allow algorithm switch (or reward shaping
        when algorithm_locked=True), 3=reshape rewards.
        algorithm_locked: when True (goal explicitly names an algorithm), skip level 2 algo switch
        and use reward shaping instead — levels map as 0=HPs, 1=arch, 2+=env_kwargs.
        Returns the pivot dict.
        """
        self._cache.set_system_prompt(_PIVOT_SYSTEM)
        if algorithm_locked:
            escalation_desc = {
                0: "Level 0 — tune hyperparameters only, keep algorithm and architecture.",
                1: "Level 1 — try a larger network architecture in addition to HP tuning.",
                2: f"Level 2 — the algorithm ({current_algorithm}) is fixed by the user. "
                   "Reshape the reward function via env_kwargs instead. For Snake-v0, try disabling "
                   "distance shaping (distance_weight=0) and increasing food_reward to 20+.",
                3: f"Level 3 — prior reward shaping did not help. Try more aggressive env_kwargs: "
                   "higher food_reward (25–30), lower death_penalty (−3 to −5), survival_bonus 0.02–0.1. "
                   "Also consider tuning architecture and HPs together.",
            }.get(escalation_level, "Level 0 — tune hyperparameters only.")
        else:
            escalation_desc = {
                0: "Level 0 — tune hyperparameters only, keep algorithm and architecture.",
                1: "Level 1 — try a larger network architecture in addition to HP tuning.",
                2: f"Level 2 — the current algorithm ({current_algorithm}) is not working. "
                   "Consider switching to a different algorithm entirely (e.g. PPO if currently DQN).",
                3: f"Level 3 — algorithm and architecture changes have not worked. "
                   "Reshape the reward function via env_kwargs. For Snake-v0, try disabling "
                   "distance shaping (distance_weight=0) and increasing food_reward.",
            }.get(escalation_level, "Level 0 — tune hyperparameters only.")
        current_state_lines = [f"Current algorithm: {current_algorithm}"]
        if current_policy_kwargs:
            current_state_lines.append(f"Current policy_kwargs (net_arch etc): {json.dumps(current_policy_kwargs)}")
        if current_hyperparameters:
            current_state_lines.append(f"Current hyperparameters: {json.dumps(current_hyperparameters)}")
        if current_env_kwargs:
            current_state_lines.append(f"Current env_kwargs: {json.dumps(current_env_kwargs)}")
        if best_policy_kwargs is not None:
            best_arch_desc = json.dumps(best_policy_kwargs) if best_policy_kwargs else "default MLP (no net_arch override)"
            _metric_label = self._metric_name_from_history(history)
            if best_metric_value is not None and best_metric_iteration is not None:
                best_context = f" (best {_metric_label}={best_metric_value:.4f} at iteration {best_metric_iteration})"
            elif best_metric_value is not None:
                best_context = f" (best {_metric_label}={best_metric_value:.4f})"
            else:
                best_context = ""
            current_state_lines.append(
                f"Best performing architecture so far: {best_arch_desc}{best_context} — prefer this at Level 1"
            )
        query = (
            "\n".join(current_state_lines) + "\n"
            f"Current metrics: {json.dumps(current_metrics)}\n"
            f"Recent history (last {min(len(history), 5)} iterations): {json.dumps(history[-5:])}\n"
            f"Escalation: {escalation_desc}\n\n"
            "Propose a strategic pivot. Avoid repeating changes that are already applied above. Return JSON."
        )
        messages = self._cache.get_messages(query)
        response = await self._generate_structured(messages, _PIVOT_SCHEMA)
        # Restore planning system prompt for next call
        self._cache.set_system_prompt(_PLANNING_SYSTEM)
        pivot = response.get("pivot", response)
        logger.info(
            "LeadAgent pivot proposed: reason=%s algorithm=%s escalation=%d",
            pivot.get("reason"), pivot.get("algorithm", current_algorithm), escalation_level,
        )
        return pivot

    # ── Plan revision (critic feedback) ──────────────────────────────────────

    async def revise_plan(self, plan: dict, feedback: str) -> dict:
        """
        Revise a training plan in response to critic feedback.
        Returns an updated plan dict.
        """
        query = (
            f"The following training plan was rejected by the Safety Critic:\n"
            f"{json.dumps(plan, indent=2)}\n\n"
            f"Critic feedback:\n{feedback}\n\n"
            "Revise the plan to address the concerns. Return JSON with the same schema."
        )
        messages = self._cache.get_messages(query)
        response = await self._generate_structured(messages, _PLAN_SCHEMA)
        self._cache.add_turn("user", query)
        self._cache.add_turn("assistant", json.dumps(response))
        revised = response.get("plan", response)
        logger.info("LeadAgent: plan revised in response to critic feedback")
        return revised

    # ── Log analysis (prefix cache) ───────────────────────────────────────────

    async def analyze_logs(self, log_snippet: str) -> str:
        """
        Inject recent sandbox logs into code context and generate a brief analysis.
        The log is pinned in the KV cache so repeated calls are efficient.
        """
        self._cache.add_code_context(f"[SANDBOX LOG]\n{log_snippet[:4000]}")
        query = "Briefly summarize the most critical events in the sandbox log above."
        messages = self._cache.get_messages(query)
        return await self._provider.generate(messages, GenerationConfig(max_tokens=512, temperature=0.3))

    # ── Structured output with retry ─────────────────────────────────────────

    @staticmethod
    def _extract_json(raw: str) -> str:
        """Strip markdown fences and LLM stop tokens, then extract the first JSON object/array."""
        import re
        text = re.sub(r"<\|im_end\|>|<\|endoftext\|>", "", raw)
        # Remove markdown code fences
        text = re.sub(r"```(?:json)?\s*", "", text).replace("```", "").strip()
        # Extract first {...} or [...] block in case of surrounding prose
        m = re.search(r"(\{.*\}|\[.*\])", text, re.DOTALL)
        return m.group(1) if m else text

    async def _generate_structured(self, messages: list[Message], schema: dict, retries: int = 2) -> dict:
        config = GenerationConfig(max_tokens=2048, temperature=0.2, json_schema=schema)
        for attempt in range(retries + 1):
            raw = await self._provider.generate(messages, config)
            try:
                return json.loads(self._extract_json(raw))
            except json.JSONDecodeError as e:
                if attempt < retries:
                    logger.warning("LeadAgent: JSON parse failed (attempt %d): %s", attempt + 1, e)
                    messages = messages + [
                        Message(role="assistant", content=raw),
                        Message(role="user", content=f"Invalid JSON: {e}. Please return valid JSON only, no markdown."),
                    ]
                else:
                    logger.error("LeadAgent: JSON parse failed after %d retries", retries)
                    raise

    def flush_iteration_context(self) -> None:
        """Call between training iterations to clear stale code context."""
        self._cache.flush_code_context()

    @staticmethod
    def _metric_name_from_history(history: list[dict]) -> str:
        """Infer the goal metric name from history entries (any non-iteration key)."""
        for entry in history:
            for k in entry:
                if k != "iteration":
                    return k
        return "metric"

    # ── Warm-start ────────────────────────────────────────────────────────────

    def _get_warm_start_hint(self, goal: str, task_type: str) -> Optional[dict]:
        """
        Query the recipe library for the best matching past recipe.
        Returns a hint dict for the planning prompt, or None if none found.
        """
        try:
            from backend.services.recipe_library import get_warm_start_hint
            # Derive domain from task_type as a rough signal
            hint = get_warm_start_hint(goal, task_type)
            if hint:
                logger.info("LeadAgent: warm-start hint from recipe '%s'", hint.get("best_matching_recipe"))
            return hint
        except Exception as exc:
            logger.debug("LeadAgent: warm-start lookup failed (non-fatal): %s", exc)
            return None
