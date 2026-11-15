"""Recipe disk search paths — platform recipes/ plus optional product packs.

Product YAML lives in product repos (e.g. Ensemble ``astra_recipes/``). Configure
via ``ASTRA_PRODUCT_RECIPES`` (comma-separated ``use_case:/abs/or/rel/path``).
Platform ``settings.recipes_path`` is always searched first so local overrides win
during the duplication / cutover window.
"""
from __future__ import annotations

import os
from typing import Iterable, Optional

from backend.config import settings


def parse_product_recipes(raw: Optional[str] = None) -> dict[str, str]:
    """Parse ``ASTRA_PRODUCT_RECIPES`` into ``{use_case: abs_dir}``.

    Example: ``ensemble:/Users/kewang/PyProjects/ensemble/astra_recipes``.
    Entries with an empty path or missing directory are skipped.
    """
    text = (raw if raw is not None else getattr(settings, "product_recipes", "") or "").strip()
    out: dict[str, str] = {}
    if not text:
        return out
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if ":" not in part:
            continue
        use_case, path = part.split(":", 1)
        use_case = use_case.strip()
        path = path.strip()
        if not use_case or not path:
            continue
        abs_path = os.path.abspath(os.path.expanduser(path))
        if os.path.isdir(abs_path):
            out[use_case] = abs_path
    return out


def recipe_search_dirs(use_case: Optional[str] = None) -> list[str]:
    """Ordered dirs to search for recipe YAML (platform first, then packs).

    When ``use_case`` is set, only that product pack is appended (if configured).
    When unset, all configured product packs are appended (for listing / bare stem load).
    """
    dirs: list[str] = []
    plat = os.path.abspath(settings.recipes_path)
    if os.path.isdir(plat):
        dirs.append(plat)
    packs = parse_product_recipes()
    if use_case:
        uc = use_case.strip()
        path = packs.get(uc)
        if path and path not in dirs:
            dirs.append(path)
    else:
        for path in packs.values():
            if path not in dirs:
                dirs.append(path)
    return dirs


def find_recipe_file(stem: str, use_case: Optional[str] = None) -> Optional[str]:
    """Return absolute path to ``stem.yaml`` / ``stem.yml`` on the search path, or None."""
    clean = (stem or "").removesuffix(".yaml").removesuffix(".yml").strip()
    if not clean:
        return None
    # Absolute / already-a-file passthrough
    if os.path.isfile(stem):
        return os.path.abspath(stem)
    for directory in recipe_search_dirs(use_case):
        for ext in (".yaml", ".yml"):
            path = os.path.join(directory, f"{clean}{ext}")
            if os.path.isfile(path):
                return path
    return None


def iter_disk_recipe_filenames(use_case: Optional[str] = None) -> Iterable[tuple[str, str]]:
    """Yield ``(directory, filename)`` for every ``*.yaml``/``*.yml`` on the search path.

    Later directories do not override earlier names when the caller builds a dict
    with platform-first insertion order.
    """
    seen: set[str] = set()
    for directory in recipe_search_dirs(use_case):
        try:
            names = sorted(os.listdir(directory))
        except OSError:
            continue
        for fname in names:
            if not fname.endswith((".yaml", ".yml")):
                continue
            stem = os.path.splitext(fname)[0]
            if stem in seen:
                continue
            seen.add(stem)
            yield directory, fname


def recipe_origin_for_dir(directory: str) -> tuple[str, Optional[str]]:
    """Classify a recipe directory as platform or product pack.

    Returns ``(source, product)`` where ``source`` is ``\"platform\"`` or
    ``\"product\"``, and ``product`` is the ``ASTRA_PRODUCT_RECIPES`` use_case
    key when ``source == \"product\"`` (never hardcodes a brand name).
    """
    abs_dir = os.path.abspath(directory)
    plat = os.path.abspath(settings.recipes_path)
    if abs_dir == plat or abs_dir.startswith(plat + os.sep):
        return "platform", None
    for uc, path in parse_product_recipes().items():
        if abs_dir == path or abs_dir.startswith(path + os.sep):
            return "product", uc
    # Unknown dir (e.g. deleted pack) — treat as platform so UI still lists it
    return "platform", None


def configured_product_use_cases() -> list[str]:
    """Sorted use_case keys from ``ASTRA_PRODUCT_RECIPES`` (may be empty)."""
    return sorted(parse_product_recipes().keys())
