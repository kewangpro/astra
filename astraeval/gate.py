"""Create-time / runtime suite ceiling helpers."""
from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Tuple

from astraeval.registry import get_suite, parse_suite_ref

# Same margin as backend.routers.missions._CEILING_TARGET_MARGIN
_CEILING_TARGET_MARGIN = 0.005


def suite_ceiling(
    suite_ref: str,
    *,
    override: Optional[Mapping[str, float]] = None,
) -> Dict[str, float]:
    """Return ceiling map for ``suite_id@rev`` (override wins when provided)."""
    if override:
        return {str(k): float(v) for k, v in override.items()}
    suite_id, rev = parse_suite_ref(suite_ref)
    suite = get_suite(suite_id, rev)
    return {str(k): float(v) for k, v in dict(getattr(suite, "ceiling", {}) or {}).items()}


def reject_over_ceiling(
    target_metric: Mapping[str, Any],
    suite_ref: str,
    *,
    override: Optional[Mapping[str, float]] = None,
    margin: float = _CEILING_TARGET_MARGIN,
) -> Optional[Tuple[str, float, float]]:
    """If any target exceeds suite ceiling, return ``(metric, target, cap)``.

    Callers raise HTTP 422 / ValueError. Returns None when ok.
    """
    if not target_metric or not suite_ref:
        return None
    ceiling = suite_ceiling(suite_ref, override=override)
    for name, tgt in target_metric.items():
        cap = ceiling.get(name)
        if cap is None:
            continue
        try:
            if float(tgt) > float(cap) + margin:
                return str(name), float(tgt), float(cap)
        except (TypeError, ValueError):
            continue
    return None


def should_unlock_hard_phase(
    *,
    plateau_iters: int,
    threshold: int = 3,
) -> bool:
    """Eval plateau → Gym hard-phase unlock hint (optional Epoch 11 hook)."""
    return int(plateau_iters) >= int(threshold)
