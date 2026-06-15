"""
Telemetry router — Step 2.3.

Two surfaces:
  POST /telemetry/missions/{id}/metrics   — sandbox pushes metrics here (HTTP)
  WS   /ws/missions/{id}/telemetry        — HUD connects here for live stream

On WebSocket connect, back-fills missed entries from the mission's
telemetry.jsonl log so the HUD covers any outage window.
"""
from __future__ import annotations

import json
import os
from typing import Optional

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, HTTPException
from pydantic import BaseModel

from backend.config import settings
from backend.services.connection_manager import manager
from backend.logging_config import get_logger

logger = get_logger(__name__)

router = APIRouter(tags=["telemetry"])


# ── Schema ─────────────────────────────────────────────────────────────────────

class MetricEvent(BaseModel):
    mission_id: str
    name: str
    value: float
    step: Optional[int] = None
    iteration: Optional[int] = None
    recorded_at: Optional[str] = None


# ── Helpers ────────────────────────────────────────────────────────────────────

def _telemetry_path(mission_id: str) -> str:
    return os.path.join(settings.data_path, "missions", mission_id, "telemetry.jsonl")


async def _backfill(mission_id: str, ws: WebSocket) -> int:
    """Replay all historical metric events from the JSONL log to a newly connected client."""
    path = _telemetry_path(mission_id)
    if not os.path.isfile(path):
        return 0
    count = 0
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                await ws.send_json({"type": "backfill", **json.loads(line)})
                count += 1
            except Exception:
                break
    return count


# ── HTTP: sandbox → FastAPI ────────────────────────────────────────────────────

@router.post("/telemetry/missions/{mission_id}/metrics", status_code=200)
async def ingest_metric(mission_id: str, event: MetricEvent):
    """Receive a metric from the sandbox and broadcast to all connected HUD clients."""
    if event.mission_id != mission_id:
        raise HTTPException(status_code=400, detail="mission_id mismatch")

    payload = event.model_dump()
    payload["type"] = "metric"

    # Append to JSONL for back-fill
    path = _telemetry_path(mission_id)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(payload) + "\n")

    await manager.broadcast(mission_id, payload)
    return {"accepted": True, "subscribers": manager.connection_count(mission_id)}


# ── WebSocket: HUD → FastAPI ───────────────────────────────────────────────────

@router.websocket("/ws/missions/{mission_id}/telemetry")
async def telemetry_ws(mission_id: str, ws: WebSocket):
    await manager.connect(mission_id, ws)
    try:
        # Back-fill historical data before streaming live events
        backfilled = await _backfill(mission_id, ws)
        await ws.send_json({"type": "backfill_complete", "events": backfilled})
        logger.info("WS back-fill: mission=%s events=%d", mission_id, backfilled)

        # Keep connection alive; server pushes via broadcast()
        while True:
            await ws.receive_text()   # client ping / keepalive
    except WebSocketDisconnect:
        manager.disconnect(mission_id, ws)
