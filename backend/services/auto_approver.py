"""
auto_approver — shared service for approval gate auto-approval.

Called by both the approvals router (frontend-triggered) and the state machine
(inline at gate creation, so missions don't stall when no browser is open).
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy.ext.asyncio import AsyncSession

from backend.agent.code_safety_classifier import CodeSafetyClassifier
from backend.database import AsyncSessionLocal
from backend.models.approval import ApprovalGate, ApprovalStatus
from backend.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class AutoApproveResult:
    gate_id: str
    safe: bool
    reason: str
    classifier: str
    action: str  # "approved" | "blocked" | "skipped"


async def try_auto_approve(
    gate_id: str,
    code_provider,
    db: Optional[AsyncSession] = None,
) -> AutoApproveResult:
    """
    Run CodeSafetyClassifier on the script attached to an EXECUTE_CODE gate.
    Approves the gate in DB if safe. Returns the verdict regardless.

    `db` is optional — if not provided a fresh session is opened internally.
    `code_provider` is an InferenceProvider instance (may be None if not loaded).
    """
    async def _get_gate(session: AsyncSession) -> Optional[ApprovalGate]:
        return await session.get(ApprovalGate, gate_id)

    if db is not None:
        gate = await _get_gate(db)
    else:
        async with AsyncSessionLocal() as session:
            gate = await _get_gate(session)

    if not gate:
        return AutoApproveResult(gate_id=gate_id, safe=False, reason="gate not found", classifier="none", action="skipped")
    if gate.status != ApprovalStatus.PENDING.value:
        return AutoApproveResult(gate_id=gate_id, safe=False, reason="gate already resolved", classifier="none", action="skipped")

    script_path = (gate.payload or {}).get("script_path")
    if not script_path or not os.path.isfile(script_path):
        return AutoApproveResult(gate_id=gate_id, safe=False, reason="no readable script_path", classifier="none", action="skipped")

    if code_provider is None:
        return AutoApproveResult(gate_id=gate_id, safe=False, reason="code provider not available", classifier="none", action="skipped")

    with open(script_path) as f:
        script = f.read()

    classifier = CodeSafetyClassifier(code_provider)
    verdict = await classifier.classify(script)

    if verdict.safe:
        async with AsyncSessionLocal() as session:
            gate_obj = await session.get(ApprovalGate, gate_id)
            if gate_obj and gate_obj.status == ApprovalStatus.PENDING.value:
                gate_obj.status = ApprovalStatus.APPROVED.value
                gate_obj.reviewer_note = f"[auto-approved] {verdict.reason} (classifier={verdict.classifier})"
                gate_obj.resolved_at = datetime.now(timezone.utc)
                await session.commit()
        logger.info("AutoApprover: gate %s APPROVED — %s", gate_id, verdict.reason)
        action = "approved"
    else:
        logger.warning("AutoApprover: gate %s BLOCKED — %s", gate_id, verdict.reason)
        action = "blocked"

    return AutoApproveResult(
        gate_id=gate_id,
        safe=verdict.safe,
        reason=verdict.reason,
        classifier=verdict.classifier,
        action=action,
    )
