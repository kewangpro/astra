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


class TournamentEntry(BaseModel):
    model_id: str
    name: str
    checkpoint_path: str
    mean_score: float
    std_score: float
    min_score: float
    max_score: float
    win_rate: float
    scores: list[float]
    rank: int


class TournamentRequest(BaseModel):
    env_id: str
    model_ids: Optional[list[str]] = None
    n_episodes: int = Field(default=5, ge=1, le=50)
    update_champion: bool = False


class TournamentResponse(BaseModel):
    env_id: str
    episodes: int
    leaderboard: list[TournamentEntry]
    champion_id: Optional[str] = None

