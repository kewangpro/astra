"""Astra use-case plugin contract — shared by astrarl / astratune train CLIs.

Products implement :class:`UseCasePlugin` and are loaded by module path via
``--use-case`` / ``ASTRA_USE_CASE`` (no built-in product aliases).
"""
from __future__ import annotations

from astra_plugin.api import UseCasePlugin
from astra_plugin.hooks import (
    AGRPO_HOOK_ATTRS,
    DPO_HOOK_ATTRS,
    GRPO_HOOK_ATTRS,
    HOOK_ATTRS_BY_COMMAND,
    OPD_HOOK_ATTRS,
    RLCD_HOOK_ATTRS,
)
from astra_plugin.loader import ensure_workspace_on_path, load_use_case
from astra_plugin.validate import validate_hooks

__all__ = [
    "UseCasePlugin",
    "load_use_case",
    "ensure_workspace_on_path",
    "validate_hooks",
    "HOOK_ATTRS_BY_COMMAND",
    "GRPO_HOOK_ATTRS",
    "AGRPO_HOOK_ATTRS",
    "OPD_HOOK_ATTRS",
    "DPO_HOOK_ATTRS",
    "RLCD_HOOK_ATTRS",
]
