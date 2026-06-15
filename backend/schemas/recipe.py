from __future__ import annotations

from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field


class RecipeRead(BaseModel):
    """Lightweight read schema for disk-based and DB-backed recipes."""
    name: str
    filename: str
    domain: Optional[str] = None
    version: Optional[str] = None
    description: Optional[str] = None
    created_at: Optional[str] = None
    content: dict = Field(default_factory=dict)


class RecipeRecordRead(BaseModel):
    """Full read schema for DB-backed RecipeRecord."""
    model_config = {"from_attributes": True}

    id: str
    name: str
    version: str
    domain: str
    task_type: str
    description: Optional[str] = None
    hyperparameters: dict
    curriculum: Optional[dict] = None
    reward_shaping: Optional[dict] = None
    full_content: dict
    mission_id: Optional[str] = None
    parent_recipe_id: Optional[str] = None
    score: Optional[float] = None
    target_metric: Optional[dict] = None
    generation: int
    consecutive_wins: int
    is_golden: bool
    created_at: datetime
    updated_at: datetime


class CrystallizeResponse(BaseModel):
    recipe: RecipeRecordRead
    yaml_path: str


class EvolveResponse(BaseModel):
    child: RecipeRecordRead
    parent_id: str


class RecipeSearchHit(BaseModel):
    name: str
    domain: Optional[str] = None
    is_golden: bool = False
    distance: float
