from __future__ import annotations

from typing import Optional
from pydantic import BaseModel


class NodeMission(BaseModel):
    mission_id: str
    sandbox_id: Optional[str]


class NodeStatus(BaseModel):
    host: str
    is_local: bool
    alive: bool
    real_available_gb: Optional[float]
    missions: list[NodeMission]
