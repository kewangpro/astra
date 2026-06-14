from __future__ import annotations

from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field


class ModelRecordCreate(BaseModel):
    name: str
    domain: str
    framework: Optional[str] = None
    architecture: Optional[str] = None
    weights_path: Optional[str] = None
    checkpoint_path: Optional[str] = None
    best_metric_name: Optional[str] = None
    best_metric_value: Optional[float] = None
    extra_metadata: dict = Field(default_factory=dict)
    experiment_id: Optional[str] = None


class ModelRecordUpdate(BaseModel):
    name: Optional[str] = None
    weights_path: Optional[str] = None
    checkpoint_path: Optional[str] = None
    best_metric_name: Optional[str] = None
    best_metric_value: Optional[float] = None
    is_champion: Optional[bool] = None
    extra_metadata: Optional[dict] = None


class ModelRecordRead(BaseModel):
    model_config = {"from_attributes": True}

    id: str
    name: str
    domain: str
    framework: Optional[str]
    architecture: Optional[str]
    weights_path: Optional[str]
    checkpoint_path: Optional[str]
    best_metric_name: Optional[str]
    best_metric_value: Optional[float]
    is_champion: bool
    extra_metadata: dict
    experiment_id: Optional[str]
    created_at: datetime
    updated_at: datetime
