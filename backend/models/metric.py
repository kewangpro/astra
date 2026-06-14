from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Optional
from sqlalchemy import String, DateTime, Float, Integer, ForeignKey
from sqlalchemy.orm import Mapped, mapped_column, relationship
from backend.database import Base


class Metric(Base):
    __tablename__ = "metrics"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    experiment_id: Mapped[Optional[str]] = mapped_column(String(36), ForeignKey("experiments.id"), index=True)
    mission_id: Mapped[Optional[str]] = mapped_column(String(36), index=True)
    name: Mapped[str] = mapped_column(String(100), nullable=False)
    value: Mapped[float] = mapped_column(Float, nullable=False)
    step: Mapped[Optional[int]] = mapped_column(Integer)
    iteration: Mapped[Optional[int]] = mapped_column(Integer)
    recorded_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    experiment: Mapped[Optional[Experiment]] = relationship("Experiment", back_populates="metrics")
