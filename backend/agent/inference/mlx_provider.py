"""
Native MLX inference provider (Apple Silicon).

Uses mlx-lm for lowest memory footprint on 24GB unified memory.
mlx-lm is only installable on Apple Silicon — import is guarded.

Install: pip install mlx-lm  (Apple Silicon only)
Recommended models (quantized to fit alongside training sandboxes):
  - mlx-community/Meta-Llama-3.1-8B-Instruct-4bit   (planning/reasoning)
  - mlx-community/Qwen2.5-Coder-7B-Instruct-4bit    (code generation)
"""
from __future__ import annotations

import asyncio
import gc
import json
import re
from typing import Optional

from backend.agent.inference.base import InferenceProvider, Message, GenerationConfig
from backend.logging_config import get_logger

logger = get_logger(__name__)

_MLX_AVAILABLE = False
try:
    import mlx.core as mx
    import mlx_lm
    _MLX_AVAILABLE = True
except ImportError:
    pass


class MLXProvider(InferenceProvider):
    def __init__(self, model_id: str = "mlx-community/Meta-Llama-3.1-8B-Instruct-4bit") -> None:
        self._model_id = model_id
        self._model = None
        self._tokenizer = None

        if not _MLX_AVAILABLE:
            raise RuntimeError(
                "mlx-lm is not installed or this is not Apple Silicon. "
                "Install with: pip install mlx-lm"
            )

    def load(self) -> None:
        if self._model is not None:
            return
        logger.info("Loading MLX model: %s", self._model_id)
        self._model, self._tokenizer = mlx_lm.load(self._model_id)
        logger.info("MLX model loaded: %s", self._model_id)

    def unload(self) -> None:
        self._model = None
        self._tokenizer = None
        gc.collect()
        if _MLX_AVAILABLE:
            mx.metal.clear_cache()
        logger.info("MLX model unloaded and cache cleared: %s", self._model_id)

    def is_loaded(self) -> bool:
        return self._model is not None

    @property
    def model_id(self) -> str:
        return self._model_id

    async def generate(self, messages: list[Message], config: Optional[GenerationConfig] = None) -> str:
        if not self.is_loaded():
            self.load()

        cfg = config or GenerationConfig()

        # Build prompt using the tokenizer's chat template
        chat = [{"role": m.role, "content": m.content} for m in messages]
        prompt = self._tokenizer.apply_chat_template(
            chat, tokenize=False, add_generation_prompt=True
        )

        # If structured output requested, append schema hint to the last user message
        if cfg.json_schema:
            schema_hint = f"\nRespond ONLY with valid JSON matching this schema:\n{json.dumps(cfg.json_schema, indent=2)}"
            prompt = prompt.rstrip() + schema_hint

        # mlx_lm.generate is synchronous — run in thread pool to avoid blocking
        response = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: mlx_lm.generate(
                self._model,
                self._tokenizer,
                prompt=prompt,
                max_tokens=cfg.max_tokens,
                temperature=cfg.temperature,
                top_p=cfg.top_p,
                verbose=False,
            ),
        )

        if cfg.json_schema:
            return self._extract_json(response)
        return response

    @staticmethod
    def _extract_json(text: str) -> str:
        """Extract the first JSON object from a response (handles markdown fences)."""
        # Strip markdown code fences
        clean = re.sub(r"```(?:json)?\s*(.*?)\s*```", r"\1", text, flags=re.DOTALL).strip()
        # Find first { ... } block
        match = re.search(r"\{.*\}", clean, re.DOTALL)
        return match.group(0) if match else clean
