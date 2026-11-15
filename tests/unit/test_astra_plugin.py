"""Unit tests for astra_plugin loader + hook validation."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from astra_plugin import load_use_case, validate_hooks
from astra_plugin.hooks import GRPO_HOOK_ATTRS
from astra_plugin.validate import HookContractError


def test_validate_hooks_passes_complete_bag():
    hooks = SimpleNamespace(**{a: object() for a in GRPO_HOOK_ATTRS})
    assert validate_hooks("grpo", hooks) is hooks


def test_validate_hooks_fails_on_missing_attr():
    hooks = SimpleNamespace(**{a: object() for a in GRPO_HOOK_ATTRS if a != "mlx_load"})
    with pytest.raises(HookContractError, match="mlx_load"):
        validate_hooks("grpo", hooks)


def test_validate_hooks_unknown_command_passthrough():
    hooks = SimpleNamespace(x=1)
    assert validate_hooks("distill", hooks) is hooks


def test_load_use_case_requires_name_or_env(monkeypatch):
    monkeypatch.delenv("ASTRA_USE_CASE", raising=False)
    with pytest.raises(ValueError, match="use-case plugin not specified"):
        load_use_case(None)


def test_load_use_case_from_env(monkeypatch):
    """ASTRA_USE_CASE supplies the importable module when CLI name is empty."""
    import sys
    from types import ModuleType

    mod = ModuleType("fake_usecase_pkg")
    mod.make_hooks = lambda *a, **k: object()  # type: ignore[attr-defined]
    mod.parse_args = lambda *a, **k: None  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "fake_usecase_pkg", mod)
    monkeypatch.setenv("ASTRA_USE_CASE", "fake_usecase_pkg")
    plugin = load_use_case(None)
    assert plugin is mod


def test_load_use_case_missing_raises():
    with pytest.raises(ModuleNotFoundError, match="not found"):
        load_use_case("definitely_not_a_real_use_case_xyz")
