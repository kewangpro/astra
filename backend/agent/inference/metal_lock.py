"""Process-wide lock serializing every Metal GPU touch point.

Concurrent Metal command-buffer encoding from different call sites causes an
uncatchable native assertion crash (libc++abi / _MTLCommandBuffer) that takes
down the entire backend process, not just one mission. Confirmed via a real
incident: MLXProvider.generate()/load() already serialized themselves via a
lock, but ModelManager._gc() and MLXProvider.unload() both called
mx.metal.clear_cache() directly, unprotected — a cache-clear racing against
an in-flight, lock-held generate() call (running in a background thread via
run_in_executor, so genuinely concurrent with the main event-loop thread)
produced exactly this crash signature:
  "A command encoder is already encoding to this command buffer"
  "Completed handler provided after commit call"

Every call site anywhere in the backend that touches Metal (mx.* calls,
mlx_lm.load/generate) must acquire this same lock — not a local, per-module
one — since Metal/the GPU is a single process-wide shared resource.
"""
from __future__ import annotations

import asyncio
from typing import Optional

_METAL_LOCK: Optional[asyncio.Lock] = None


def get_metal_lock() -> asyncio.Lock:
    global _METAL_LOCK
    if _METAL_LOCK is None:
        _METAL_LOCK = asyncio.Lock()
    return _METAL_LOCK
