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

import psutil

from backend.agent.inference.base import InferenceProvider, Message, GenerationConfig
from backend.logging_config import get_logger

logger = get_logger(__name__)

# Below this much real free memory, proactively GC + clear the Metal cache
# before attempting to load a model — a real incident showed a backend crash
# (uncatchable libc++abi/Metal command-buffer OOM, same failure class as
# ModelManager.before_sandbox_launch()'s fix) happening during mlx_lm.load()
# itself while real memory was tight from concurrently-running missions.
# ModelManager's real-memory-aware guard only covers the sandbox-launch path;
# this covers the other in-process Metal entry point — loading a planning/
# coding model — which has no relationship to ModelManager and can't reuse
# its guard directly.
_LOW_MEMORY_THRESHOLD_GB = 2.0

# Serializes all MLX inference calls — concurrent Metal GPU access causes SIGABRT
_MLX_LOCK: Optional[asyncio.Lock] = None

def _get_mlx_lock() -> asyncio.Lock:
    global _MLX_LOCK
    if _MLX_LOCK is None:
        _MLX_LOCK = asyncio.Lock()
    return _MLX_LOCK

_MLX_AVAILABLE = False
try:
    import mlx.core as mx
    import mlx_lm
    from mlx_lm.sample_utils import make_sampler
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
        try:
            available_gb = psutil.virtual_memory().available / (1024 ** 3)
        except Exception:
            available_gb = None
        if available_gb is not None and available_gb < _LOW_MEMORY_THRESHOLD_GB:
            logger.warning(
                "MLXProvider: real memory low (%.1f GB free) before loading %s — "
                "running gc + Metal cache clear first",
                available_gb, self._model_id,
            )
            gc.collect()
            mx.metal.clear_cache()
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
        async with _get_mlx_lock():
            return await self._generate_locked(messages, config)

    async def _generate_locked(self, messages: list[Message], config: Optional[GenerationConfig] = None) -> str:
        if not self.is_loaded():
            self.load()

        cfg = config or GenerationConfig()

        # Build prompt using the tokenizer's chat template
        chat = [{"role": m.role, "content": m.content} for m in messages]

        # Inject schema instruction into the last user message so it appears
        # inside the user turn, not after the assistant start token.
        if cfg.json_schema:
            schema_hint = (
                f"\n\nRespond ONLY with valid JSON matching this schema:\n"
                f"{json.dumps(cfg.json_schema, indent=2)}"
            )
            if chat and chat[-1]["role"] == "user":
                chat[-1] = {**chat[-1], "content": chat[-1]["content"] + schema_hint}
            else:
                chat.append({"role": "user", "content": schema_hint})

        prompt = self._tokenizer.apply_chat_template(
            chat, tokenize=False, add_generation_prompt=True
        )

        # mlx_lm 0.29+: temperature/top_p go through make_sampler, not generate() kwargs
        sampler = make_sampler(temp=cfg.temperature, top_p=cfg.top_p)

        # mlx_lm.generate is synchronous — run in thread pool to avoid blocking.
        # asyncio.shield prevents task cancellation from interrupting mid-flight Metal
        # command buffers (which causes _MTLCommandBuffer assertion crashes on macOS).
        response = await asyncio.shield(
            asyncio.get_event_loop().run_in_executor(
                None,
                lambda: mlx_lm.generate(
                    self._model,
                    self._tokenizer,
                    prompt=prompt,
                    max_tokens=cfg.max_tokens,
                    sampler=sampler,
                    verbose=False,
                ),
            )
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
