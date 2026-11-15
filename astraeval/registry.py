"""Suite registry: ``suite_id@rev`` → Suite instance + tournament seed rule."""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple, Type

from astraeval.protocol import Suite
from astraeval.suites.hstu_hr10 import HstuHr10Suite
from astraeval.suites.routing_bare import RoutingBareEvalSuite

_SUITE_TYPES: Dict[str, Type] = {
    "hstu_hr10": HstuHr10Suite,
    "routing_bare": RoutingBareEvalSuite,
}


def parse_suite_ref(ref: str) -> Tuple[str, int]:
    text = (ref or "").strip()
    if not text:
        raise ValueError("empty suite ref")
    if "@" in text:
        name, rev_s = text.rsplit("@", 1)
        return name.strip(), int(rev_s)
    return text, 1


def get_suite(suite_id: str, suite_rev: int = 1, **kwargs: Any) -> Suite:
    cls = _SUITE_TYPES.get(suite_id)
    if cls is None:
        raise KeyError(f"unknown suite_id: {suite_id!r}")
    return cls(suite_rev=suite_rev, **kwargs)  # type: ignore[call-arg]


def ensure_suite(suite_id: str, suite_rev: int = 1, **kwargs: Any) -> Suite:
    """Alias of get_suite — suites are code-defined, not downloaded."""
    return get_suite(suite_id, suite_rev, **kwargs)


def tournament_seed(episode: int, suite_id: str = "rl_tournament", suite_rev: int = 1) -> int:
    """Canonical tournament seed — suite identity owns the 2000+ep rule.

    ``suite_id`` / ``suite_rev`` are part of the contract so champions can
    record which suite population produced the ranking.
    """
    _ = (suite_id, suite_rev)  # reserved for future suite-specific offsets
    return 2000 + int(episode)
