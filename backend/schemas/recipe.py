from __future__ import annotations

from typing import Optional
from pydantic import BaseModel, Field


class RecipeRead(BaseModel):
    name: str
    filename: str
    domain: Optional[str] = None
    version: Optional[str] = None
    description: Optional[str] = None
    created_at: Optional[str] = None
    content: dict = Field(default_factory=dict)
