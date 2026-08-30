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
    # Target not reached, but the search has exhausted every lever it has
    # (escalation maxed, no new best for a long stretch). Terminal, like
    # COMPLETED/FAILED — the loop stops and keeps the best checkpoint instead
    # of burning compute indefinitely on an unreachable target.
    STALLED = "stalled"


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
    pivot_escalation_count: Mapped[Optional[int]] = mapped_column(Integer, default=0)
    best_policy_kwargs: Mapped[Optional[dict]] = mapped_column(JSON)
    pivot_pre_best: Mapped[Optional[str]] = mapped_column(String(100))
    current_plan: Mapped[Optional[dict]] = mapped_column(JSON)
    container_id: Mapped[Optional[str]] = mapped_column(String(255))
    subprocess_pid: Mapped[Optional[int]] = mapped_column(Integer)
    remote_pid: Mapped[Optional[int]] = mapped_column(Integer)  # SSHSandbox: pid on settings.sandbox_host
    last_checkpoint_path: Mapped[Optional[str]] = mapped_column(Text)
    error_log: Mapped[Optional[str]] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))

    @property
    def host(self) -> Optional[str]:
        """Which node this mission's sandbox is/was running on, for the Nodes panel
        and mission grid badge. Derived, not stored — reuses the subprocess_pid/
        remote_pid columns already written by SandboxManager.launch()/recover()."""
        if self.subprocess_pid:
            return "local"
        if self.remote_pid:
            from backend.config import settings
            return settings.sandbox_host or None
        return None
