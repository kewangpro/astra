"""Discover and import use-case plugins (product-agnostic)."""
from __future__ import annotations

import importlib
import os
import sys
from typing import Any


def ensure_workspace_on_path(workspace: str | None) -> None:
    """Put workspace (+ parent + optional ``.astra_plugins``) on ``sys.path``.

    Products may ship branded plugin packages under ``<repo>/.astra_plugins/``
    (prepended first) so ``--use-case <brand>`` imports cleanly without
    colliding with other top-level modules in the product repo.
    """
    if not workspace:
        return
    ws = os.path.abspath(workspace)
    parent = os.path.dirname(ws)
    # .astra_plugins first so branded names win over other top-level modules.
    for p in (
        os.path.join(parent, ".astra_plugins"),
        os.path.join(ws, ".astra_plugins"),
        parent,
        ws,
    ):
        if os.path.isdir(p) and p not in sys.path:
            sys.path.insert(0, p)


def _looks_like_plugin(mod: Any) -> bool:
    return callable(getattr(mod, "make_hooks", None)) and callable(
        getattr(mod, "parse_args", None)
    )


def load_use_case(name: str | None = None) -> Any:
    """Import a use-case plugin by **module path**.

    Resolution
    ----------
    1. Explicit ``name`` (CLI ``--use-case``)
    2. Else ``ASTRA_USE_CASE`` environment variable
    3. Import that dotted/top-level module name
    4. If the module lacks ``make_hooks``, try ``{name}.usecase``

    No product aliases — pass an importable module that implements
    :class:`~astra_plugin.api.UseCasePlugin`. Products that want a short
    branded name (e.g. Ensemble's ``--use-case ensemble``) ship that module
    themselves on ``PYTHONPATH``.

    Raises
    ------
    ValueError
        When neither ``name`` nor ``ASTRA_USE_CASE`` is set.
    ModuleNotFoundError
        When the module cannot be imported.
    TypeError
        When the module lacks ``make_hooks`` / ``parse_args``.
    """
    raw = (name or os.environ.get("ASTRA_USE_CASE") or "").strip()
    if not raw:
        raise ValueError(
            "use-case plugin not specified: pass --use-case <module> "
            "or set ASTRA_USE_CASE to an importable UseCasePlugin module"
        )

    try:
        mod = importlib.import_module(raw)
    except ImportError as exc:
        raise ModuleNotFoundError(
            f"use-case module {raw!r} not found. Put the product package on "
            f"PYTHONPATH. Original: {exc}"
        ) from exc

    if _looks_like_plugin(mod):
        return mod

    try:
        sub = importlib.import_module(f"{raw}.usecase")
    except ImportError as exc:
        raise TypeError(
            f"{raw!r} imported but has no make_hooks/parse_args, and "
            f"{raw}.usecase is missing ({exc})"
        ) from exc
    if _looks_like_plugin(sub):
        return sub
    raise TypeError(
        f"{raw!r} / {raw}.usecase must define make_hooks and parse_args "
        f"(astra_plugin.UseCasePlugin)"
    )
