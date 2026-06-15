"""
Utility for emitting structured status events to the telemetry WebSocket stream.

Broadcasts to connected HUD clients and appends to the mission's telemetry.jsonl
so events survive page reloads via back-fill.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Optional

from backend.config import settings
from backend.services.connection_manager import manager
from backend.logging_config import get_logger

logger = get_logger(__name__)


def _telemetry_path(mission_id: str) -> str:
    return os.path.join(settings.data_path, "missions", mission_id, "telemetry.jsonl")


async def emit_status(
    mission_id: str,
    name: str,
    *,
    event_type: str = "info",
    value: Optional[str] = None,
    iteration: Optional[int] = None,
) -> None:
    """
    Broadcast a status event and persist it to the mission telemetry log.

    Args:
        mission_id: The mission UUID.
        name: Human-readable description shown in the event stream.
        event_type: "info" | "success" | "warn" | "error" | "pivot"
        value: Optional detail string (shown after name in the log row).
        iteration: Current training iteration, if applicable.
    """
    payload: dict = {
        "type": event_type,
        "mission_id": mission_id,
        "name": name,
        "recorded_at": datetime.now(timezone.utc).isoformat(),
    }
    if value is not None:
        payload["value"] = value
    if iteration is not None:
        payload["iteration"] = iteration

    path = _telemetry_path(mission_id)
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "a") as f:
            f.write(json.dumps(payload) + "\n")
    except Exception as exc:
        logger.warning("telemetry_emitter: failed to persist event for %s: %s", mission_id, exc)

    await manager.broadcast(mission_id, payload)
