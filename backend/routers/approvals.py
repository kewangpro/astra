"""
Approvals router — manages approval gates for the autonomous loop.

GET    /approvals                          → list all pending gates
GET    /approvals/{gate_id}                → get a single gate
POST   /approvals/{gate_id}/approve        → approve a gate
POST   /approvals/{gate_id}/reject         → reject a gate
POST   /approvals/{gate_id}/auto-approve   → LLM safety classification → approve if safe
"""
from __future__ import annotations

import os
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


class AutoApproveResult(BaseModel):
    gate_id: str
    safe: bool
    reason: str
    classifier: str
    action: str  # "approved" | "blocked"


@router.post("/{gate_id}/auto-approve", response_model=AutoApproveResult)
async def auto_approve_gate(gate_id: str, db: AsyncSession = Depends(get_db)):
    """
    Run LLM safety classification on the script attached to an EXECUTE_CODE gate.
    If the classifier deems it safe, the gate is approved automatically.
    If unsafe, the gate remains PENDING and the verdict is returned for human review.
    """
    from backend.agent.code_safety_classifier import CodeSafetyClassifier
    from backend.routers.agent import get_code_provider

    gate = await db.get(ApprovalGate, gate_id)
    if not gate:
        raise HTTPException(status_code=404, detail="Approval gate not found")
    if gate.status != ApprovalStatus.PENDING.value:
        raise HTTPException(status_code=409, detail=f"Gate already resolved: {gate.status}")

    # Read script from payload
    script_path = (gate.payload or {}).get("script_path")
    if not script_path or not os.path.isfile(script_path):
        raise HTTPException(status_code=422, detail="Gate has no readable script_path in payload")

    with open(script_path, "r") as f:
        script = f.read()

    classifier = CodeSafetyClassifier(get_code_provider())
    verdict = await classifier.classify(script)

    if verdict.safe:
        gate.status = ApprovalStatus.APPROVED.value
        gate.reviewer_note = f"[auto-approved] {verdict.reason} (classifier={verdict.classifier})"
        gate.resolved_at = datetime.now(timezone.utc)
        await db.commit()
        await db.refresh(gate)
        logger.info("Auto-approve: gate %s APPROVED — %s", gate_id, verdict.reason)
        action = "approved"
    else:
        logger.warning("Auto-approve: gate %s BLOCKED — %s", gate_id, verdict.reason)
        action = "blocked"

    return AutoApproveResult(
        gate_id=gate_id,
        safe=verdict.safe,
        reason=verdict.reason,
        classifier=verdict.classifier,
        action=action,
    )


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
