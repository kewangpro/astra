from __future__ import annotations

from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field
from backend.models.mission import MissionStatus


class MissionCreate(BaseModel):
    goal: str
    task_type: str
    target_metric: dict = Field(default_factory=dict)
    autonomy_mode: str = "supervised"


class MissionUpdate(BaseModel):
    status: Optional[MissionStatus] = None
    current_iteration: Optional[int] = None
    best_metric_value: Optional[str] = None
    current_plan: Optional[dict] = None
    last_checkpoint_path: Optional[str] = None
    error_log: Optional[str] = None


class MissionRead(BaseModel):
    model_config = {"from_attributes": True}

    id: str
    goal: str
    task_type: str
    target_metric: dict
    autonomy_mode: str
    status: str
    current_iteration: int
    best_metric_value: Optional[str]
    last_checkpoint_path: Optional[str]
    error_log: Optional[str]
    created_at: datetime
    updated_at: datetime
    completed_at: Optional[datetime]
