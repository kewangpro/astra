from __future__ import annotations

import os
import re
from typing import Optional, List
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from backend.config import settings
from backend.database import get_db
from backend.models.mission import Mission, MissionStatus
from backend.models.manifest import RequirementManifest
from backend.schemas.mission import MissionCreate, MissionRead, MissionUpdate

router = APIRouter(prefix="/missions", tags=["missions"])


def _parse_target_metric(goal: str) -> dict:
    """Extract a target metric dict from free-text goal. Returns {} if nothing recognized."""
    m = re.search(r"(\d+(?:\.\d+)?)\s*%\s*accuracy", goal, re.IGNORECASE)
    if m:
        return {"accuracy": float(m.group(1)) / 100}
    m = re.search(r"accuracy\s+of\s+(\d+(?:\.\d+)?)", goal, re.IGNORECASE)
    if m:
        val = float(m.group(1))
        return {"accuracy": val if val <= 1.0 else val / 100}
    m = re.search(r"(?:mean_?)?reward\s+of\s+(\d+(?:\.\d+)?)", goal, re.IGNORECASE)
    if m:
        return {"mean_reward": float(m.group(1))}
    m = re.search(r"(?:eval_)?loss\s+(?:of\s+|<=?\s*)(\d+(?:\.\d+)?)", goal, re.IGNORECASE)
    if m:
        return {"eval_loss": float(m.group(1))}
    # Generic: "achieve {metric_name} of {value}" — catch-all for custom metrics
    m = re.search(
        r"achieve\s+([a-zA-Z][a-zA-Z0-9_]*)\s+of\s+(\d+(?:\.\d+)?)",
        goal, re.IGNORECASE,
    )
    if m:
        return {m.group(1).lower(): float(m.group(2))}
    return {}


@router.post("", response_model=MissionRead, status_code=status.HTTP_201_CREATED)
async def create_mission(payload: MissionCreate, db: AsyncSession = Depends(get_db)):
    payload_dict = payload.model_dump()
    if not payload_dict.get("target_metric"):
        payload_dict["target_metric"] = _parse_target_metric(payload.goal)
    mission = Mission(**payload_dict)
    db.add(mission)
    await db.commit()
    await db.refresh(mission)
    return mission


@router.get("", response_model=List[MissionRead])
async def list_missions(status_filter: Optional[str] = None, db: AsyncSession = Depends(get_db)):
    q = select(Mission)
    if status_filter:
        q = q.where(Mission.status == status_filter)
    result = await db.execute(q.order_by(Mission.created_at.desc()))
    return result.scalars().all()


@router.get("/{mission_id}", response_model=MissionRead)
async def get_mission(mission_id: str, db: AsyncSession = Depends(get_db)):
    mission = await db.get(Mission, mission_id)
    if not mission:
        raise HTTPException(status_code=404, detail="Mission not found")
    return mission


@router.patch("/{mission_id}", response_model=MissionRead)
async def update_mission(mission_id: str, payload: MissionUpdate, db: AsyncSession = Depends(get_db)):
    mission = await db.get(Mission, mission_id)
    if not mission:
        raise HTTPException(status_code=404, detail="Mission not found")
    for k, v in payload.model_dump(exclude_none=True).items():
        if k == "status" and v is not None:
            setattr(mission, k, v.value if isinstance(v, MissionStatus) else v)
        else:
            setattr(mission, k, v)
    await db.commit()
    await db.refresh(mission)
    return mission


@router.get("/{mission_id}/manifest")
async def get_manifest(mission_id: str, db: AsyncSession = Depends(get_db)):
    """Return the requirement manifest for a mission."""
    mission = await db.get(Mission, mission_id)
    if not mission:
        raise HTTPException(status_code=404, detail="Mission not found")
    path = os.path.join(settings.data_path, "missions", mission_id, "requirements.json")
    if not os.path.isfile(path):
        raise HTTPException(status_code=404, detail="Manifest not yet generated (mission has not started)")
    return RequirementManifest.load(path).to_dict()


@router.delete("/{mission_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_mission(mission_id: str, db: AsyncSession = Depends(get_db)):
    mission = await db.get(Mission, mission_id)
    if not mission:
        raise HTTPException(status_code=404, detail="Mission not found")
    await db.delete(mission)
    await db.commit()
