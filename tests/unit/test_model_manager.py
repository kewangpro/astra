"""Unit tests for ModelManager in agent/model_manager.py."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from backend.agent.model_manager import ModelManager, MODEL_FOOTPRINTS


def _mock_provider(model_id: str, loaded: bool = True) -> MagicMock:
    p = MagicMock()
    p.model_id = model_id
    p.is_loaded.return_value = loaded
    return p


class TestModelManagerMemory:
    def setup_method(self):
        self.mm = ModelManager(total_memory_gb=24.0)

    def test_available_gb_full_when_nothing_loaded(self):
        assert self.mm.available_gb() == pytest.approx(24.0)

    def test_estimated_usage_zero_when_nothing_registered(self):
        assert self.mm.estimated_usage_gb() == pytest.approx(0.0)

    def test_estimated_usage_accounts_known_model(self):
        model_id = "mlx-community/Meta-Llama-3.1-8B-Instruct-4bit"
        provider = _mock_provider(model_id, loaded=True)
        self.mm.register("llama", provider)
        assert self.mm.estimated_usage_gb() == pytest.approx(MODEL_FOOTPRINTS[model_id])

    def test_available_gb_reduces_when_model_loaded(self):
        model_id = "mlx-community/Qwen2.5-Coder-7B-Instruct-4bit"
        provider = _mock_provider(model_id, loaded=True)
        self.mm.register("coder", provider)
        expected = 24.0 - MODEL_FOOTPRINTS[model_id]
        assert self.mm.available_gb() == pytest.approx(expected)

    def test_unloaded_provider_not_counted(self):
        model_id = "mlx-community/Meta-Llama-3.1-8B-Instruct-4bit"
        provider = _mock_provider(model_id, loaded=False)
        self.mm.register("llama", provider)
        assert self.mm.estimated_usage_gb() == pytest.approx(0.0)

    def test_unknown_model_uses_default_footprint(self):
        provider = _mock_provider("some-unknown-model", loaded=True)
        self.mm.register("mystery", provider)
        assert self.mm.estimated_usage_gb() == pytest.approx(8.0)

    def test_drafter_counted_when_loaded(self):
        drafter_id = "mlx-community/Llama-3.2-1B-Instruct-4bit"
        drafter = _mock_provider(drafter_id, loaded=True)
        self.mm.register_drafter(drafter)
        assert self.mm.estimated_usage_gb() == pytest.approx(MODEL_FOOTPRINTS[drafter_id])

    def test_drafter_not_counted_when_unloaded(self):
        drafter_id = "mlx-community/Llama-3.2-1B-Instruct-4bit"
        drafter = _mock_provider(drafter_id, loaded=False)
        self.mm.register_drafter(drafter)
        assert self.mm.estimated_usage_gb() == pytest.approx(0.0)

    def test_multiple_providers_summed(self):
        id1 = "mlx-community/Meta-Llama-3.1-8B-Instruct-4bit"
        id2 = "mlx-community/Qwen2.5-Coder-7B-Instruct-4bit"
        self.mm.register("p1", _mock_provider(id1, loaded=True))
        self.mm.register("p2", _mock_provider(id2, loaded=True))
        expected = MODEL_FOOTPRINTS[id1] + MODEL_FOOTPRINTS[id2]
        assert self.mm.estimated_usage_gb() == pytest.approx(expected)

    def test_ollama_model_has_zero_footprint(self):
        provider = _mock_provider("llama3.1:8b", loaded=True)
        self.mm.register("ollama", provider)
        assert self.mm.estimated_usage_gb() == pytest.approx(0.0)


class TestModelManagerSandboxLifecycle:
    def setup_method(self):
        self.mm = ModelManager(total_memory_gb=24.0)

    def test_before_sandbox_launch_sets_sandbox_active(self):
        self.mm.before_sandbox_launch(sandbox_memory_gb=4.0)
        assert self.mm._sandbox_active is True

    def test_after_sandbox_exit_clears_sandbox_active(self):
        self.mm._sandbox_active = True
        self.mm.after_sandbox_exit()
        assert self.mm._sandbox_active is False

    def test_before_sandbox_evicts_drafter(self):
        drafter = _mock_provider("mlx-community/Llama-3.2-1B-Instruct-4bit", loaded=True)
        self.mm.register_drafter(drafter)
        self.mm.before_sandbox_launch(sandbox_memory_gb=1.0)
        drafter.unload.assert_called_once()

    def test_before_sandbox_skips_eviction_if_drafter_not_loaded(self):
        drafter = _mock_provider("mlx-community/Llama-3.2-1B-Instruct-4bit", loaded=False)
        self.mm.register_drafter(drafter)
        self.mm.before_sandbox_launch(sandbox_memory_gb=1.0)
        drafter.unload.assert_not_called()

    def test_before_sandbox_no_drafter_registered(self):
        # Should not raise
        self.mm.before_sandbox_launch(sandbox_memory_gb=1.0)

    def test_gc_triggered_when_insufficient_headroom(self):
        # Load providers until available memory < requested sandbox memory
        model_id = "mlx-community/Meta-Llama-3.1-8B-Instruct-4bit"
        for i in range(3):
            self.mm.register(f"p{i}", _mock_provider(model_id, loaded=True))
        # 3 × 4.5 GB = 13.5 GB used, 10.5 GB free
        # Request 12 GB → should trigger GC
        with patch("backend.agent.model_manager.gc.collect") as mock_gc:
            self.mm.before_sandbox_launch(sandbox_memory_gb=12.0)
            mock_gc.assert_called_once()

    def test_gc_not_triggered_when_sufficient_headroom(self):
        with patch("backend.agent.model_manager.gc.collect") as mock_gc:
            self.mm.before_sandbox_launch(sandbox_memory_gb=1.0)
            mock_gc.assert_not_called()

    def test_restore_drafter_logs_readiness(self):
        drafter = _mock_provider("mlx-community/Llama-3.2-1B-Instruct-4bit", loaded=False)
        self.mm.register_drafter(drafter)
        self.mm._sandbox_active = False
        self.mm._restore_drafter()   # should not raise

    def test_restore_drafter_skipped_while_sandbox_active(self):
        drafter = _mock_provider("mlx-community/Llama-3.2-1B-Instruct-4bit", loaded=False)
        self.mm.register_drafter(drafter)
        self.mm._sandbox_active = True
        self.mm._restore_drafter()   # no-op because sandbox is active
        drafter.unload.assert_not_called()
