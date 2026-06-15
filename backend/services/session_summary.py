"""
SessionSummary — Step 7.3.

Writes a SESSION_SUMMARY.md to data/missions/{id}/ at the end of every
loop iteration. The file captures:
  1. Last successful action
  2. Current blocker (if any)
  3. Exact next step

This artifact acts as "Primary Working Memory" for the next agent instance,
ensuring continuity after context resets or service restarts.
"""
from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Optional

from backend.config import settings
from backend.logging_config import get_logger

logger = get_logger(__name__)


def write_session_summary(
    mission_id: str,
    iteration: int,
    goal: str,
    algorithm: str,
    current_metrics: dict,
    manifest_summary: dict,
    blocker: Optional[str] = None,
    pivot_applied: Optional[str] = None,
) -> None:
    """Write SESSION_SUMMARY.md for the completed iteration."""
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %Human:%M:%S UTC").replace("Human", "H")
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    metrics_str = ", ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                            for k, v in current_metrics.items()) or "none recorded"

    passed = manifest_summary.get("passed", 0)
    total = manifest_summary.get("total", 0)
    complete = manifest_summary.get("complete", False)

    last_action = (
        f"Completed iteration {iteration} — trained {algorithm} — "
        f"metrics: [{metrics_str}] — "
        f"manifest: {passed}/{total} requirements passed"
    )

    if complete:
        next_step = "All requirements satisfied. Mission will complete."
    elif blocker:
        next_step = f"Fix blocker and retry: {blocker}"
    elif pivot_applied:
        next_step = f"Apply pivot '{pivot_applied}' and run iteration {iteration + 1}"
    else:
        next_step = f"Run iteration {iteration + 1} with current hyperparameters"

    lines = [
        f"# Session Summary — Mission {mission_id}",
        f"_Generated: {now}_",
        "",
        f"## Iteration {iteration}",
        "",
        "### Last Successful Action",
        last_action,
        "",
        "### Current Blocker",
        blocker if blocker else "None",
        "",
        "### Next Step",
        next_step,
        "",
        "### Manifest Status",
        f"{passed}/{total} requirements passed — {'COMPLETE' if complete else 'IN PROGRESS'}",
        "",
        "### Current Metrics",
        metrics_str,
    ]
    if pivot_applied:
        lines += ["", "### Pivot Applied", pivot_applied]

    path = os.path.join(settings.data_path, "missions", mission_id, "SESSION_SUMMARY.md")
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            f.write("\n".join(lines) + "\n")
        logger.info("SessionSummary: wrote iteration=%d for mission=%s", iteration, mission_id)
    except Exception as exc:
        logger.warning("SessionSummary: failed to write for mission=%s: %s", mission_id, exc)
