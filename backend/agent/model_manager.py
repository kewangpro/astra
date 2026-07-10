"""
ModelManager — coordinates LLM memory on 24GB unified memory.

Responsibilities:
- Tracks which inference providers are loaded and how much memory they consume.
- Evicts the speculative drafter before a training sandbox is launched.
- Triggers GC / Metal cache clear when the sandbox needs headroom.
- Restores the drafter when the sandbox goes idle.
"""
from __future__ import annotations

import gc
import platform
from typing import Optional

import psutil

from backend.agent.inference.base import InferenceProvider
from backend.logging_config import get_logger

logger = get_logger(__name__)

# Estimated VRAM footprints (GB) for common quantizations
MODEL_FOOTPRINTS: dict[str, float] = {
    # Local MLX models (MacBook)
    "mlx-community/Meta-Llama-3.1-8B-Instruct-4bit": 4.5,
    "mlx-community/Qwen2.5-Coder-7B-Instruct-4bit": 4.0,
    "mlx-community/Mistral-Nemo-Instruct-2407-4bit": 8.0,
    # Ollama models run on mac-mini — no local memory cost
    "llama3.1:8b": 0.0,
    "gemma3:12b": 0.0,
    # Speculative drafter models (local, loaded only when sandbox is idle)
    "mlx-community/Llama-3.2-1B-Instruct-4bit": 0.7,
    "mlx-community/Llama-3.2-3B-Instruct-4bit": 2.5,
}


class ModelManager:
    """
    Manages LLM memory allocation across inference providers.
    All providers must be registered before use.
    """

    def __init__(self, total_memory_gb: float = 24.0) -> None:
        self.total_memory_gb = total_memory_gb
        self._providers: dict[str, InferenceProvider] = {}
        self._drafter: Optional[InferenceProvider] = None
        self._sandbox_active: bool = False

    # ── Provider registration ─────────────────────────────────────────────────

    def register(self, name: str, provider: InferenceProvider) -> None:
        self._providers[name] = provider
        logger.info("ModelManager: registered provider '%s' (%s)", name, provider.model_id)

    def register_drafter(self, provider: InferenceProvider) -> None:
        self._drafter = provider
        logger.info("ModelManager: registered speculative drafter (%s)", provider.model_id)

    # ── Memory estimation ────────────────────────────────────────────────────

    def estimated_usage_gb(self) -> float:
        total = 0.0
        for provider in self._providers.values():
            if provider.is_loaded():
                total += MODEL_FOOTPRINTS.get(provider.model_id, 8.0)
        if self._drafter and self._drafter.is_loaded():
            total += MODEL_FOOTPRINTS.get(self._drafter.model_id, 1.5)
        return total

    def available_gb(self) -> float:
        return self.total_memory_gb - self.estimated_usage_gb()

    def real_available_gb(self) -> float:
        """Real system memory headroom via psutil, reflecting actual pressure
        from EVERY consumer (registered LLM providers, a locally-running
        training subprocess, anything else) — not just what this class
        happens to be tracking. estimated_usage_gb()/available_gb() only
        know about registered InferenceProviders; a local RL training
        subprocess (SubprocessSandbox) is invisible to that accounting even
        though it can consume several GB of real (often GPU-shared, on
        Apple Silicon unified memory) memory. Falls back to the tracked
        estimate if psutil is unavailable for any reason.
        """
        try:
            return psutil.virtual_memory().available / (1024 ** 3)
        except Exception:
            return self.available_gb()

    # ── Sandbox lifecycle hooks ───────────────────────────────────────────────

    def before_sandbox_launch(self, sandbox_memory_gb: float) -> None:
        """
        Called by SandboxManager before spawning a training process.
        Evicts the speculative drafter and optionally the coding model
        to free enough memory for the sandbox.

        Gates the GC/cache-clear decision on the more conservative of the
        tracked-provider estimate and real system memory (see
        real_available_gb()) — a real incident showed the tracked estimate
        alone reporting healthy headroom while a concurrently-running local
        training subprocess had actually driven real memory pressure high
        enough that the in-process mx.metal.clear_cache() call below (the
        only in-process, unisolated Metal call in this whole path) itself
        hit a fatal "Insufficient Memory" Metal command buffer failure —
        an uncaught C++ exception that crashed the entire backend process,
        not just the mission being launched.
        """
        self._sandbox_active = True
        self._evict_drafter()

        tracked_available = self.available_gb()
        real_available = self.real_available_gb()
        effective_available = min(tracked_available, real_available)

        if effective_available < sandbox_memory_gb:
            logger.warning(
                "ModelManager: insufficient headroom (tracked=%.1f GB, real=%.1f GB "
                "free, need %.1f GB) — triggering GC",
                tracked_available, real_available, sandbox_memory_gb,
            )
            self._gc()

        logger.info(
            "ModelManager: pre-sandbox state — %.1f GB used, %.1f GB free (real=%.1f GB)",
            self.estimated_usage_gb(), tracked_available, real_available,
        )

    def after_sandbox_exit(self) -> None:
        """Called when the sandbox terminates; reload the speculative drafter."""
        self._sandbox_active = False
        self._restore_drafter()
        logger.info("ModelManager: post-sandbox — drafter restored")

    # ── Speculative drafter ──────────────────────────────────────────────────

    def _evict_drafter(self) -> None:
        if self._drafter and self._drafter.is_loaded():
            self._drafter.unload()
            logger.info("ModelManager: speculative drafter evicted")

    def _restore_drafter(self) -> None:
        if self._drafter and not self._sandbox_active:
            # Lazy-load on next generate() call — don't pre-load here
            logger.info("ModelManager: speculative drafter ready to reload on next use")

    # ── GC ───────────────────────────────────────────────────────────────────

    def _gc(self) -> None:
        gc.collect()
        if platform.system() == "Darwin":
            try:
                import mlx.core as mx
                mx.metal.clear_cache()
                logger.info("ModelManager: Metal cache cleared")
            except ImportError:
                pass
