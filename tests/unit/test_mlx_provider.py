"""Unit tests for MLXProvider's real-memory-aware guard in mlx_provider.py."""
from __future__ import annotations

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
