"""Unit tests for MLXProvider's real-memory-aware guard in mlx_provider.py."""
from __future__ import annotations

import asyncio
from unittest.mock import MagicMock, patch

import pytest

from backend.agent.inference.mlx_provider import MLXProvider


def _provider() -> MLXProvider:
    return MLXProvider(model_id="mlx-community/fake-model-4bit")


class TestLoadMemoryGuard:
    def test_load_skips_gc_when_memory_healthy(self):
        provider = _provider()
        fake_vm = MagicMock(available=8 * (1024 ** 3))
        with patch("backend.agent.inference.mlx_provider.psutil.virtual_memory", return_value=fake_vm), \
             patch("backend.agent.inference.mlx_provider.gc.collect") as mock_gc, \
             patch("backend.agent.inference.mlx_provider.mx.metal.clear_cache") as mock_clear, \
             patch("backend.agent.inference.mlx_provider.mlx_lm.load", return_value=(MagicMock(), MagicMock())):
            provider.load()
        mock_gc.assert_not_called()
        mock_clear.assert_not_called()

    def test_load_runs_gc_and_clear_cache_when_memory_low(self):
        """Real incident: the backend crashed with an uncatchable Metal OOM
        during mlx_lm.load() itself while real memory was tight from
        concurrently-running missions. Proactively freeing memory first
        reduces (does not eliminate) that risk."""
        provider = _provider()
        fake_vm = MagicMock(available=1 * (1024 ** 3))
        with patch("backend.agent.inference.mlx_provider.psutil.virtual_memory", return_value=fake_vm), \
             patch("backend.agent.inference.mlx_provider.gc.collect") as mock_gc, \
             patch("backend.agent.inference.mlx_provider.mx.metal.clear_cache") as mock_clear, \
             patch("backend.agent.inference.mlx_provider.mlx_lm.load", return_value=(MagicMock(), MagicMock())):
            provider.load()
        mock_gc.assert_called_once()
        mock_clear.assert_called_once()

    def test_load_proceeds_even_if_memory_check_fails(self):
        """psutil failing must not block loading the model — fail open, same
        as ModelManager.real_available_gb()'s own fallback behavior."""
        provider = _provider()
        with patch("backend.agent.inference.mlx_provider.psutil.virtual_memory", side_effect=OSError("boom")), \
             patch("backend.agent.inference.mlx_provider.mlx_lm.load", return_value=(MagicMock(), MagicMock())) as mock_load:
            provider.load()
        mock_load.assert_called_once()
        assert provider.is_loaded()

    def test_load_is_noop_when_already_loaded(self):
        provider = _provider()
        provider._model = MagicMock()
        provider._tokenizer = MagicMock()
        with patch("backend.agent.inference.mlx_provider.mlx_lm.load") as mock_load:
            provider.load()
        mock_load.assert_not_called()


class TestMetalLockUsage:
    """Real incident: MLXProvider.unload() and ModelManager._gc() both called
    mx.metal.clear_cache() directly, unprotected by the same lock generate()/
    load() use — a cache-clear racing against an in-flight, lock-held
    generate() call (running in a background thread, genuinely concurrent
    with the main event-loop thread) crashed the whole backend with an
    uncatchable Metal assertion ("A command encoder is already encoding to
    this command buffer")."""

    def setup_method(self):
        # get_metal_lock()'s singleton binds to whatever event loop is active
        # the first time it's created; pytest-asyncio gives each test its own
        # loop, so a lock created in an earlier test is stale here. Reset it
        # so this test creates a fresh one on its own loop — production only
        # ever has the one long-lived loop, so this is purely a test-isolation
        # concern, not a real behavior difference.
        import backend.agent.inference.metal_lock as metal_lock_module
        metal_lock_module._METAL_LOCK = None

    async def test_unload_acquires_metal_lock(self):
        from backend.agent.inference.metal_lock import get_metal_lock

        provider = _provider()
        provider._model = MagicMock()
        provider._tokenizer = MagicMock()
        lock = get_metal_lock()
        with patch("backend.agent.inference.mlx_provider.mx.metal.clear_cache"):
            async with lock:
                # Lock is already held — unload() must wait for it rather
                # than racing past and touching Metal concurrently.
                task = asyncio.ensure_future(provider.unload())
                await asyncio.sleep(0.01)
                assert not task.done()  # blocked on the lock
            await task  # released — unload() can now proceed and complete
        assert not provider.is_loaded()

    async def test_generate_and_gc_share_the_same_lock_instance(self):
        """mlx_provider.py and model_manager.py must import the identical
        lock object — two separate locks would defeat the whole point."""
        from backend.agent.inference import mlx_provider
        from backend.agent import model_manager
        from backend.agent.inference.metal_lock import get_metal_lock

        assert mlx_provider.get_metal_lock is get_metal_lock
        assert model_manager.get_metal_lock is get_metal_lock
        assert mlx_provider.get_metal_lock() is model_manager.get_metal_lock()
