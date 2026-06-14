"""
Approvals router — manages approval gates for the autonomous loop.

GET    /approvals                     → list all pending gates
GET    /approvals/{gate_id}           → get a single gate
POST   /approvals/{gate_id}/approve   → approve a gate
POST   /approvals/{gate_id}/reject    → reject a gate
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional, List

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.database import get_db
from backend.models.approval import ApprovalGate, ApprovalStatus
from backend.logging_config import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/approvals", tags=["approvals"])


class ApprovalGateRead(BaseModel):
    model_config = {"from_attributes": True}
    id: str
    mission_id: str
    gate_type: str
    status: str
    payload: Optional[dict]
    reviewer_note: Optional[str]
    created_at: datetime
    resolved_at: Optional[datetime]


class ApprovalDecision(BaseModel):
    note: Optional[str] = None


@router.get("", response_model=List[ApprovalGateRead])
async def list_approvals(pending_only: bool = True, db: AsyncSession = Depends(get_db)):
    q = select(ApprovalGate)
    if pending_only:
        q = q.where(ApprovalGate.status == ApprovalStatus.PENDING.value)
    result = await db.execute(q.order_by(ApprovalGate.created_at.desc()))
    return result.scalars().all()


@router.get("/{gate_id}", response_model=ApprovalGateRead)
async def get_approval(gate_id: str, db: AsyncSession = Depends(get_db)):
    gate = await db.get(ApprovalGate, gate_id)
    if not gate:
        raise HTTPException(status_code=404, detail="Approval gate not found")
    return gate


@router.post("/{gate_id}/approve", response_model=ApprovalGateRead)
async def approve_gate(gate_id: str, body: ApprovalDecision, db: AsyncSession = Depends(get_db)):
    gate = await db.get(ApprovalGate, gate_id)
    if not gate:
        raise HTTPException(status_code=404, detail="Approval gate not found")
    if gate.status != ApprovalStatus.PENDING.value:
        raise HTTPException(status_code=409, detail=f"Gate already resolved: {gate.status}")
    gate.status = ApprovalStatus.APPROVED.value
    gate.reviewer_note = body.note
    gate.resolved_at = datetime.now(timezone.utc)
    await db.commit()
    await db.refresh(gate)
    logger.info("Approval gate %s approved (mission=%s)", gate_id, gate.mission_id)
    return gate


@router.post("/{gate_id}/reject", response_model=ApprovalGateRead)
async def reject_gate(gate_id: str, body: ApprovalDecision, db: AsyncSession = Depends(get_db)):
    gate = await db.get(ApprovalGate, gate_id)
    if not gate:
        raise HTTPException(status_code=404, detail="Approval gate not found")
    if gate.status != ApprovalStatus.PENDING.value:
        raise HTTPException(status_code=409, detail=f"Gate already resolved: {gate.status}")
    gate.status = ApprovalStatus.REJECTED.value
    gate.reviewer_note = body.note
    gate.resolved_at = datetime.now(timezone.utc)
    await db.commit()
    await db.refresh(gate)
    logger.info("Approval gate %s rejected (mission=%s)", gate_id, gate.mission_id)
    return gate
