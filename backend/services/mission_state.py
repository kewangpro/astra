"""
MissionState — Step 7.5.

Maintains MISSION_MANIFEST.json at data/missions/{id}/ as the
"Current Source of Truth" for an in-progress mission.

Stores: best hyperparameters, best score, best algorithm, and a
compressed per-iteration history. The LeadAgent reads this on warm-start
to avoid repeating failed approaches.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Optional

from backend.config import settings
from backend.logging_config import get_logger

logger = get_logger(__name__)

_VERSION = "1.0"
_MAX_HISTORY = 20   # keep last N iterations to bound file size


def _path(mission_id: str) -> str:
    return os.path.join(settings.data_path, "missions", mission_id, "MISSION_MANIFEST.json")


def load(mission_id: str) -> dict:
    """Load MISSION_MANIFEST.json, returning an empty structure if not present."""
    p = _path(mission_id)
    if os.path.isfile(p):
        try:
            with open(p) as f:
                return json.load(f)
        except Exception as exc:
            logger.warning("MissionState: failed to load manifest for %s: %s", mission_id, exc)
    return {
        "version": _VERSION,
        "mission_id": mission_id,
        "best_hyperparameters": {},
        "best_score": None,
        "best_algorithm": None,
        "iteration_history": [],
        "lessons_learned": [],
        "last_updated": None,
    }


def update(
    mission_id: str,
    iteration: int,
    plan: dict,
    metrics: dict,
    lessons: Optional[list] = None,
) -> dict:
    """
    Update MISSION_MANIFEST.json with results from the latest iteration.
    Returns the updated state dict.
    """
    state = load(mission_id)

    # Extract primary metric value for tracking best
    hp = plan.get("hyperparameters", {})
    algorithm = plan.get("algorithm", "unknown")
    score = _primary_score(metrics)

    # Update best if improved
    current_best = state.get("best_score")
    if score is not None and (current_best is None or score > current_best):
        state["best_score"] = score
        state["best_hyperparameters"] = hp
        state["best_algorithm"] = algorithm

    # Append iteration record (cap history length)
    record = {
        "iteration": iteration,
        "algorithm": algorithm,
        "hyperparameters": hp,
        "metrics": metrics,
        "score": score,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    history = state.get("iteration_history", [])
    history.append(record)
    state["iteration_history"] = history[-_MAX_HISTORY:]

    # Merge lessons
    if lessons:
        existing = state.get("lessons_learned", [])
        state["lessons_learned"] = list(dict.fromkeys(existing + lessons))[-_MAX_HISTORY:]

    state["last_updated"] = datetime.now(timezone.utc).isoformat()

    _save(mission_id, state)
    logger.info(
        "MissionState: updated iteration=%d score=%s best=%s mission=%s",
        iteration, score, state.get("best_score"), mission_id,
    )
    return state


def _primary_score(metrics: dict) -> Optional[float]:
    """Extract a single numeric score from a metrics dict (first numeric value)."""
    for v in metrics.values():
        try:
            return float(v)
        except (TypeError, ValueError):
            continue
    return None


def _save(mission_id: str, state: dict) -> None:
    p = _path(mission_id)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w") as f:
        json.dump(state, f, indent=2)
