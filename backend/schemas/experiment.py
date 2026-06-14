from __future__ import annotations

from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field


class ExperimentCreate(BaseModel):
    name: str
    domain: str
    algorithm: Optional[str] = None
    environment: Optional[str] = None
    hyperparameters: dict = Field(default_factory=dict)
    notes: Optional[str] = None


class ExperimentUpdate(BaseModel):
    name: Optional[str] = None
    status: Optional[str] = None
    hyperparameters: Optional[dict] = None
    notes: Optional[str] = None


class ExperimentRead(BaseModel):
    model_config = {"from_attributes": True}

    id: str
    name: str
    domain: str
    algorithm: Optional[str]
    environment: Optional[str]
    hyperparameters: dict
    status: str
    notes: Optional[str]
    created_at: datetime
    updated_at: datetime
