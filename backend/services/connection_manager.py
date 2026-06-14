from __future__ import annotations

from collections import defaultdict
from typing import List

from fastapi import WebSocket

from backend.logging_config import get_logger

logger = get_logger(__name__)


class ConnectionManager:
    """Tracks active WebSocket connections per mission."""

    def __init__(self) -> None:
        self._connections: dict[str, List[WebSocket]] = defaultdict(list)

    async def connect(self, mission_id: str, ws: WebSocket) -> None:
        await ws.accept()
        self._connections[mission_id].append(ws)
        logger.info("WS connected: mission=%s total=%d", mission_id, len(self._connections[mission_id]))

    def disconnect(self, mission_id: str, ws: WebSocket) -> None:
        self._connections[mission_id].discard(ws) if hasattr(self._connections[mission_id], "discard") else None
        try:
            self._connections[mission_id].remove(ws)
        except ValueError:
            pass
        logger.info("WS disconnected: mission=%s remaining=%d", mission_id, len(self._connections[mission_id]))

    async def broadcast(self, mission_id: str, data: dict) -> None:
        dead = []
        for ws in self._connections[mission_id]:
            try:
                await ws.send_json(data)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.disconnect(mission_id, ws)

    def connection_count(self, mission_id: str) -> int:
        return len(self._connections[mission_id])


manager = ConnectionManager()
