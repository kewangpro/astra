from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Optional
from sqlalchemy import String, JSON, DateTime, Text, Integer
from sqlalchemy.orm import Mapped, mapped_column
from backend.database import Base


class MissionStatus(str, Enum):
    PENDING = "pending"
    PLANNING = "planning"
    RUNNING = "running"
    PAUSED = "paused"
    EVALUATING = "evaluating"
    COMPLETED = "completed"
    FAILED = "failed"


class Mission(Base):
    __tablename__ = "missions"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    goal: Mapped[str] = mapped_column(Text, nullable=False)
    task_type: Mapped[str] = mapped_column(String(50), nullable=False)
    target_metric: Mapped[dict] = mapped_column(JSON, default=dict)
    autonomy_mode: Mapped[str] = mapped_column(String(50), default="supervised")
    status: Mapped[str] = mapped_column(String(50), default=MissionStatus.PENDING, index=True)
    current_iteration: Mapped[int] = mapped_column(Integer, default=0)
    best_metric_value: Mapped[Optional[str]] = mapped_column(String(100))
    best_metric_iteration: Mapped[Optional[int]] = mapped_column(Integer)
    current_metric_value: Mapped[Optional[str]] = mapped_column(String(100))
    current_plan: Mapped[Optional[dict]] = mapped_column(JSON)
    container_id: Mapped[Optional[str]] = mapped_column(String(255))
    subprocess_pid: Mapped[Optional[int]] = mapped_column(Integer)
    last_checkpoint_path: Mapped[Optional[str]] = mapped_column(Text)
    error_log: Mapped[Optional[str]] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
