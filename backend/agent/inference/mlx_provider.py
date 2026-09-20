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

import ctypes
import ctypes.util
import importlib.util
import platform

from backend.agent.inference.base import InferenceProvider, Message, GenerationConfig
from backend.agent.inference.metal_lock import get_metal_lock
from backend.agent.model_manager import MODEL_FOOTPRINTS
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
#
# A flat 2.0 GB was too low: a second real incident crashed the backend with
# 2-4 GB free (above this flat floor, so no GC/cache-clear ran) while loading
# a ~4.5 GB model — free memory was fine by this threshold's standard but
# nowhere near enough for the model actually being loaded. The threshold now
# scales with the target model's own footprint plus a fixed safety margin for
# the allocation/copy overhead mlx_lm.load() needs beyond the model's steady-
# state size, so a big model triggers the defensive GC at a correspondingly
# higher free-memory bar than a small one.
_LOW_MEMORY_SAFETY_MARGIN_GB = 2.0


def is_metal_available() -> bool:
    """Check whether Metal GPU devices are accessible in this process context.

    Under restricted environments (e.g. sandboxed IDE runners, test subshells, or
    containers), macOS blocks access to Metal devices. Direct import of mlx.core
    in that state triggers mlx::core::metal::Device::Device() which throws an
    uncaught Objective-C NSRangeException (-[__NSArray0 objectAtIndex:]),
    aborting Python with SIGABRT (Abort trap: 6).
    """
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        return False
    try:
        metal_path = ctypes.util.find_library("Metal")
        if not metal_path:
            return False
        metal = ctypes.cdll.LoadLibrary(metal_path)
        metal.MTLCopyAllDevices.restype = ctypes.c_void_p
        devices = metal.MTLCopyAllDevices()
        if not devices:
            return False
        objc = ctypes.cdll.LoadLibrary(ctypes.util.find_library("objc"))
        objc.sel_registerName.restype = ctypes.c_void_p
        objc.sel_registerName.argtypes = [ctypes.c_char_p]
        objc.objc_msgSend.restype = ctypes.c_ulong
        objc.objc_msgSend.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        count = objc.objc_msgSend(devices, objc.sel_registerName(b"count"))
        return count > 0
    except Exception:
        return False


def _check_mlx_installed() -> bool:
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        return False
    return (
        importlib.util.find_spec("mlx") is not None
        and importlib.util.find_spec("mlx_lm") is not None
    )


_MLX_AVAILABLE = _check_mlx_installed()

_real_mx = None
_real_mlx_lm = None
_real_make_sampler = None


def _get_real_mx():
    global _real_mx
    if _real_mx is None and _MLX_AVAILABLE and is_metal_available():
        try:
            import mlx.core as mx_mod
            _real_mx = mx_mod
        except Exception as exc:
            logger.warning("MLXProvider: failed to import mlx.core: %s", exc)
    return _real_mx


def _get_real_mlx_lm():
    global _real_mlx_lm
    if _real_mlx_lm is None and _MLX_AVAILABLE and is_metal_available():
        try:
            import mlx_lm as mlx_lm_mod
            _real_mlx_lm = mlx_lm_mod
        except Exception as exc:
            logger.warning("MLXProvider: failed to import mlx_lm: %s", exc)
    return _real_mlx_lm


def _get_real_make_sampler():
    global _real_make_sampler
    if _real_make_sampler is None and _MLX_AVAILABLE and is_metal_available():
        try:
            from mlx_lm.sample_utils import make_sampler as sampler_fn
            _real_make_sampler = sampler_fn
        except Exception as exc:
            logger.warning("MLXProvider: failed to import make_sampler: %s", exc)
    return _real_make_sampler


class _MetalProxy:
    def clear_cache(self):
        real = _get_real_mx()
        if real is not None and hasattr(real, "metal"):
            real.metal.clear_cache()

    def __getattr__(self, item):
        real = _get_real_mx()
        if real is not None and hasattr(real, "metal"):
            return getattr(real.metal, item)
        raise AttributeError(f"Metal has no attribute '{item}'")


class _MxProxy:
    def __init__(self):
        self.metal = _MetalProxy()

    def __getattr__(self, item):
        real = _get_real_mx()
        if real is not None:
            return getattr(real, item)
        raise AttributeError(f"mlx.core has no attribute '{item}'")


class _MlxLmProxy:
    def load(self, *args, **kwargs):
        real = _get_real_mlx_lm()
        if real is not None:
            return real.load(*args, **kwargs)
        raise RuntimeError("mlx_lm is not available")

    def generate(self, *args, **kwargs):
        real = _get_real_mlx_lm()
        if real is not None:
            return real.generate(*args, **kwargs)
        raise RuntimeError("mlx_lm is not available")

    def __getattr__(self, item):
        real = _get_real_mlx_lm()
        if real is not None:
            return getattr(real, item)
        raise AttributeError(f"mlx_lm has no attribute '{item}'")


def make_sampler(*args, **kwargs):
    real_fn = _get_real_make_sampler()
    if real_fn is not None:
        return real_fn(*args, **kwargs)
    return None


mx = _MxProxy()
mlx_lm = _MlxLmProxy()


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
        model_footprint_gb = MODEL_FOOTPRINTS.get(self._model_id, 4.5)
        low_memory_threshold_gb = model_footprint_gb + _LOW_MEMORY_SAFETY_MARGIN_GB
        if available_gb is not None and available_gb < low_memory_threshold_gb:
            logger.warning(
                "MLXProvider: real memory low (%.1f GB free, need ~%.1f GB) before "
                "loading %s — running gc + Metal cache clear first",
                available_gb, low_memory_threshold_gb, self._model_id,
            )
            gc.collect()
            mx.metal.clear_cache()
        logger.info("Loading MLX model: %s", self._model_id)
        self._model, self._tokenizer = mlx_lm.load(self._model_id)
        logger.info("MLX model loaded: %s", self._model_id)

    async def unload(self) -> None:
        self._model = None
        self._tokenizer = None
        gc.collect()
        if _MLX_AVAILABLE:
            # Must hold the same lock generate()/load() use — a real incident
            # showed this call racing against an in-flight, lock-held
            # generate() call (running in a background thread) and crashing
            # the whole backend with an uncatchable Metal assertion. See
            # metal_lock.py's docstring for the full incident writeup.
            async with get_metal_lock():
                mx.metal.clear_cache()
        logger.info("MLX model unloaded and cache cleared: %s", self._model_id)

    def is_loaded(self) -> bool:
        return self._model is not None

    @property
    def model_id(self) -> str:
        return self._model_id

    async def generate(self, messages: list[Message], config: Optional[GenerationConfig] = None) -> str:
        async with get_metal_lock():
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
