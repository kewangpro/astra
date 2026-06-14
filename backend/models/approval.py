from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Optional
from sqlalchemy import String, JSON, DateTime, Text
from sqlalchemy.orm import Mapped, mapped_column
from backend.database import Base


class ApprovalStatus(str, Enum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"


class GateType(str, Enum):
    EXECUTE_CODE = "execute_code"
    RESOURCE_ALLOCATION = "resource_allocation"
    DEPLOY_MODEL = "deploy_model"


class ApprovalGate(Base):
    __tablename__ = "approval_gates"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    mission_id: Mapped[str] = mapped_column(String(36), index=True, nullable=False)
    gate_type: Mapped[str] = mapped_column(String(50), nullable=False)
    status: Mapped[str] = mapped_column(String(50), default=ApprovalStatus.PENDING, index=True)
    payload: Mapped[Optional[dict]] = mapped_column(JSON)   # e.g. script path, resource request
    reviewer_note: Mapped[Optional[str]] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    resolved_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
