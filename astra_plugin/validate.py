"""Validate hook bags against per-command attribute contracts."""
from __future__ import annotations

from typing import Any

from astra_plugin.hooks import HOOK_ATTRS_BY_COMMAND


class HookContractError(TypeError):
    """Hooks object missing required attributes for a learner command."""


def validate_hooks(command: str, hooks: Any) -> Any:
    """Ensure ``hooks`` exposes every attr required by ``command``'s ``run_*``.

    Returns ``hooks`` unchanged on success (convenient for call chaining).
    Commands without a registered contract (e.g. CE runners) pass through.
    """
    cmd = (command or "").lower()
    required = HOOK_ATTRS_BY_COMMAND.get(cmd)
    if not required:
        return hooks
    if hooks is None:
        raise HookContractError(f"hooks is None for command {cmd!r}")
    missing = sorted(a for a in required if not hasattr(hooks, a))
    if missing:
        raise HookContractError(
            f"use-case hooks for {cmd!r} missing required attributes: "
            f"{', '.join(missing)}"
        )
    return hooks
