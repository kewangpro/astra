from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Optional
from sqlalchemy import String, JSON, DateTime, Text, Integer, Float, Boolean
from sqlalchemy.orm import Mapped, mapped_column
from backend.database import Base


class RecipeRecord(Base):
    __tablename__ = "recipe_records"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    name: Mapped[str] = mapped_column(String(255), unique=True, nullable=False, index=True)
    version: Mapped[str] = mapped_column(String(50), nullable=False, default="1.0.0")
    domain: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    task_type: Mapped[str] = mapped_column(String(50), nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text)

    hyperparameters: Mapped[dict] = mapped_column(JSON, default=dict)
    curriculum: Mapped[Optional[dict]] = mapped_column(JSON)
    reward_shaping: Mapped[Optional[dict]] = mapped_column(JSON)
    full_content: Mapped[dict] = mapped_column(JSON, default=dict)

    # Provenance
    mission_id: Mapped[Optional[str]] = mapped_column(String(36), index=True)
    parent_recipe_id: Mapped[Optional[str]] = mapped_column(String(36))

    # Scoring & evolution
    score: Mapped[Optional[float]] = mapped_column(Float)
    target_metric: Mapped[Optional[dict]] = mapped_column(JSON)
    generation: Mapped[int] = mapped_column(Integer, default=0)  # 0=hand-crafted, 1=crystallized, 2+=evolved
    consecutive_wins: Mapped[int] = mapped_column(Integer, default=0)
    is_golden: Mapped[bool] = mapped_column(Boolean, default=False)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )
