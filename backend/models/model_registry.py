from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Optional
from sqlalchemy import String, JSON, DateTime, Text, ForeignKey, Float
from sqlalchemy.orm import Mapped, mapped_column, relationship
from backend.database import Base


class ModelRecord(Base):
    __tablename__ = "model_records"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    domain: Mapped[str] = mapped_column(String(100), nullable=False)
    framework: Mapped[Optional[str]] = mapped_column(String(100))
    architecture: Mapped[Optional[str]] = mapped_column(String(255))
    weights_path: Mapped[Optional[str]] = mapped_column(Text)
    checkpoint_path: Mapped[Optional[str]] = mapped_column(Text)
    best_metric_name: Mapped[Optional[str]] = mapped_column(String(100))
    best_metric_value: Mapped[Optional[float]] = mapped_column(Float)
    is_champion: Mapped[bool] = mapped_column(default=False)
    extra_metadata: Mapped[dict] = mapped_column(JSON, default=dict)
    experiment_id: Mapped[Optional[str]] = mapped_column(String(36), ForeignKey("experiments.id"), index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )

    experiment: Mapped[Optional[Experiment]] = relationship("Experiment", back_populates="model_records")
