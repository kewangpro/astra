"""
Analysis router — exposes SpatialAnalyzer and PolicyAuditor results via API.

POST /analysis/missions/{id}/saliency  → Grad-CAM saliency map
POST /analysis/missions/{id}/audit     → action-distribution histogram
"""
from __future__ import annotations

from typing import Optional, List

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from backend.analysis.spatial_analyzer import SpatialAnalyzer
from backend.analysis.policy_auditor import PolicyAuditor
from backend.config import settings
import os

router = APIRouter(prefix="/analysis", tags=["analysis"])


class SaliencyRequest(BaseModel):
    checkpoint_path: str
    observation: list           # raw observation array
    layer_name: Optional[str] = None


class AuditRequest(BaseModel):
    checkpoint_path: str
    observations: List[list]    # list of raw observations
    n_actions: int
    n_samples: Optional[int] = None


@router.post("/missions/{mission_id}/saliency")
async def saliency_map(mission_id: str, body: SaliencyRequest):
    """Compute a Grad-CAM saliency map for the given observation."""
    analyzer = SpatialAnalyzer(body.checkpoint_path)
    result = analyzer.generate_saliency_map(body.observation, layer_name=body.layer_name)
    return {"mission_id": mission_id, **result}


@router.post("/missions/{mission_id}/audit")
async def policy_audit(mission_id: str, body: AuditRequest):
    """Compute action-distribution histogram and detect mode collapse."""
    auditor = PolicyAuditor(body.checkpoint_path, body.n_actions)
    result = auditor.compute_histogram(body.observations, n_samples=body.n_samples)
    return {"mission_id": mission_id, **result}
