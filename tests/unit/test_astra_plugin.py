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


def test_load_use_case_ensemble_alias(monkeypatch):
    """ensemble → finetune when Ensemble is on PYTHONPATH."""
    import sys
    from pathlib import Path

    root = Path("/Users/kewang/PyProjects/ensemble")
    if not root.is_dir():
        pytest.skip("Ensemble checkout not present")
    monkeypatch.syspath_prepend(str(root))
    # Drop cached modules so alias resolution is fresh
    for key in list(sys.modules):
        if key == "finetune" or key.startswith("finetune."):
            del sys.modules[key]
    plugin = load_use_case("ensemble")
    assert callable(plugin.make_hooks)
    assert callable(plugin.parse_args)


def test_load_use_case_missing_raises():
    with pytest.raises(ModuleNotFoundError, match="not found"):
        load_use_case("definitely_not_a_real_use_case_xyz")
