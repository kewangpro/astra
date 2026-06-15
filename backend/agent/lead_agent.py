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
Consider: task type (rl/sft/ml), algorithm selection, hyperparameters, curriculum phases,
and how you will measure success against the target metric.

For ml tasks, always include "dataset_path" in hyperparameters. Use the sklearn dataset name
(e.g. "iris", "digits", "breast_cancer", "wine") for built-in datasets, or a file path for
custom datasets."""

_PIVOT_SYSTEM = """\
You are ASTRA's Lead Agent analyzing a training run that has stalled or plateaued.
Given the current metrics and training history, propose a strategic pivot.
Respond with valid JSON describing the adjustments to make."""

_PLAN_SCHEMA = {
    "type": "object",
    "properties": {
        "plan": {
            "type": "object",
            "properties": {
                "task_type": {"type": "string", "enum": ["rl", "sft", "ml"]},
                "algorithm": {"type": "string"},
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

    async def propose_pivot(self, current_metrics: dict, history: list[dict]) -> dict:
        """
        Analyze a stalled run and propose hyperparameter adjustments.
        Returns the pivot dict.
        """
        self._cache.set_system_prompt(_PIVOT_SYSTEM)
        query = (
            f"Current metrics: {json.dumps(current_metrics)}\n"
            f"Recent history (last {len(history)} iterations): {json.dumps(history[-5:])}\n\n"
            "Propose a strategic pivot. Return JSON."
        )
        messages = self._cache.get_messages(query)
        response = await self._generate_structured(messages, _PIVOT_SCHEMA)
        # Restore planning system prompt for next call
        self._cache.set_system_prompt(_PLANNING_SYSTEM)
        pivot = response.get("pivot", response)
        logger.info("LeadAgent pivot proposed: reason=%s", pivot.get("reason"))
        return pivot

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
