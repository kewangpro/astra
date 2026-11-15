"""Discover and import use-case plugins."""
from __future__ import annotations

import importlib
import os
import sys
from typing import Any

# Built-in product aliases → candidate module names (tried in order).
# CLIs should pass the alias (e.g. ``ensemble``); hardcoding product imports
# belongs here, not in astrarl/astratune CLI modules.
_USE_CASE_ALIASES: dict[str, tuple[str, ...]] = {
    "ensemble": ("finetune", "usecase"),
    "finetune": ("finetune", "usecase"),
    # Legacy name from the transitional ensemble_usecase package.
    "ensemble_usecase": ("finetune", "usecase", "ensemble_usecase"),
}


def ensure_workspace_on_path(workspace: str | None) -> None:
    """Put workspace (+ parent / nested finetune) on ``sys.path`` for plugin imports."""
    if not workspace:
        return
    ws = os.path.abspath(workspace)
    parent = os.path.dirname(ws)
    for p in (parent, ws, os.path.join(ws, "finetune")):
        if os.path.isdir(p) and p not in sys.path:
            sys.path.insert(0, p)


def _looks_like_plugin(mod: Any) -> bool:
    return callable(getattr(mod, "make_hooks", None)) and callable(
        getattr(mod, "parse_args", None)
    )


def load_use_case(name: str | None = None) -> Any:
    """Import a use-case plugin module.

    Resolution order
    ----------------
    1. ``name`` or ``ASTRA_USE_CASE`` env (default ``ensemble``)
    2. If ``name`` is an alias in ``_USE_CASE_ALIASES``, try each candidate module
    3. Else ``importlib.import_module(name)`` (dotted path allowed)

    Raises
    ------
    ModuleNotFoundError
        When no candidate imports, with a short install hint.
    TypeError
        When the module imports but lacks ``make_hooks`` / ``parse_args``.
    """
    raw = (name or os.environ.get("ASTRA_USE_CASE") or "ensemble").strip()
    key = raw.lower().replace("-", "_")
    candidates = list(_USE_CASE_ALIASES.get(key, (raw,)))

    errors: list[str] = []
    for mod_name in candidates:
        try:
            mod = importlib.import_module(mod_name)
        except ImportError as exc:
            errors.append(f"{mod_name}: {exc}")
            continue
        if not _looks_like_plugin(mod):
            # Package root may re-export; try ``.usecase`` submodule.
            try:
                sub = importlib.import_module(f"{mod_name}.usecase")
            except ImportError:
                errors.append(
                    f"{mod_name}: imported but missing make_hooks/parse_args"
                )
                continue
            if _looks_like_plugin(sub):
                return sub
            errors.append(
                f"{mod_name}.usecase: missing make_hooks/parse_args"
            )
            continue
        return mod

    hint = (
        "Install the product package on PYTHONPATH "
        "(Ensemble: repo root so `import finetune` works; "
        "see finetune/VENDOR.md)."
    )
    raise ModuleNotFoundError(
        f"use-case {raw!r} not found (tried {candidates}). {hint} "
        f"Details: {'; '.join(errors)}"
    )
