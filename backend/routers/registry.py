from __future__ import annotations

import asyncio
import os
from typing import Optional, List
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from backend.database import get_db
from backend.models.experiment import Experiment
from backend.models.model_registry import ModelRecord
from backend.schemas.experiment import ExperimentCreate, ExperimentRead, ExperimentUpdate
from backend.schemas.model_registry import (
    ModelRecordCreate,
    ModelRecordRead,
    ModelRecordUpdate,
    TournamentRequest,
    TournamentResponse,
)
from backend.evaluator.benchmark import run_tournament_match

router = APIRouter(prefix="/registry", tags=["registry"])



# ── Experiments ────────────────────────────────────────────────────────────────

@router.post("/experiments", response_model=ExperimentRead, status_code=status.HTTP_201_CREATED)
async def create_experiment(payload: ExperimentCreate, db: AsyncSession = Depends(get_db)):
    exp = Experiment(**payload.model_dump())
    db.add(exp)
    await db.commit()
    await db.refresh(exp)
    return exp


@router.get("/experiments", response_model=List[ExperimentRead])
async def list_experiments(domain: Optional[str] = None, db: AsyncSession = Depends(get_db)):
    q = select(Experiment)
    if domain:
        q = q.where(Experiment.domain == domain)
    result = await db.execute(q.order_by(Experiment.created_at.desc()))
    return result.scalars().all()


@router.get("/experiments/{experiment_id}", response_model=ExperimentRead)
async def get_experiment(experiment_id: str, db: AsyncSession = Depends(get_db)):
    exp = await db.get(Experiment, experiment_id)
    if not exp:
        raise HTTPException(status_code=404, detail="Experiment not found")
    return exp


@router.patch("/experiments/{experiment_id}", response_model=ExperimentRead)
async def update_experiment(experiment_id: str, payload: ExperimentUpdate, db: AsyncSession = Depends(get_db)):
    exp = await db.get(Experiment, experiment_id)
    if not exp:
        raise HTTPException(status_code=404, detail="Experiment not found")
    for k, v in payload.model_dump(exclude_none=True).items():
        setattr(exp, k, v)
    await db.commit()
    await db.refresh(exp)
    return exp


@router.delete("/experiments/{experiment_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_experiment(experiment_id: str, db: AsyncSession = Depends(get_db)):
    exp = await db.get(Experiment, experiment_id)
    if not exp:
        raise HTTPException(status_code=404, detail="Experiment not found")
    await db.delete(exp)
    await db.commit()


# ── Model Records ──────────────────────────────────────────────────────────────

@router.post("/models", response_model=ModelRecordRead, status_code=status.HTTP_201_CREATED)
async def create_model_record(payload: ModelRecordCreate, db: AsyncSession = Depends(get_db)):
    record = ModelRecord(**payload.model_dump())
    db.add(record)
    await db.commit()
    await db.refresh(record)
    return record


@router.get("/models", response_model=List[ModelRecordRead])
async def list_model_records(
    domain: Optional[str] = None,
    champion_only: bool = False,
    db: AsyncSession = Depends(get_db),
):
    q = select(ModelRecord)
    if domain:
        q = q.where(ModelRecord.domain == domain)
    if champion_only:
        q = q.where(ModelRecord.is_champion == True)  # noqa: E712
    result = await db.execute(q.order_by(ModelRecord.created_at.desc()))
    return result.scalars().all()


@router.get("/models/{model_id}", response_model=ModelRecordRead)
async def get_model_record(model_id: str, db: AsyncSession = Depends(get_db)):
    record = await db.get(ModelRecord, model_id)
    if not record:
        raise HTTPException(status_code=404, detail="Model record not found")
    return record


@router.patch("/models/{model_id}", response_model=ModelRecordRead)
async def update_model_record(model_id: str, payload: ModelRecordUpdate, db: AsyncSession = Depends(get_db)):
    record = await db.get(ModelRecord, model_id)
    if not record:
        raise HTTPException(status_code=404, detail="Model record not found")
    for k, v in payload.model_dump(exclude_none=True).items():
        setattr(record, k, v)
    await db.commit()
    await db.refresh(record)
    return record


@router.delete("/models/{model_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_model_record(model_id: str, db: AsyncSession = Depends(get_db)):
    record = await db.get(ModelRecord, model_id)
    if not record:
        raise HTTPException(status_code=404, detail="Model record not found")
    await db.delete(record)
    await db.commit()


# ── Tournament Arena ───────────────────────────────────────────────────────────

@router.post("/tournament", response_model=TournamentResponse)
async def run_tournament(payload: TournamentRequest, db: AsyncSession = Depends(get_db)):
    entries: List[dict] = []
    if payload.model_ids:
        for mid in payload.model_ids:
            rec = await db.get(ModelRecord, mid)
            if rec:
                path = rec.checkpoint_path or rec.weights_path
                if path and os.path.exists(path):
                    entries.append({"id": rec.id, "name": rec.name, "path": path})
    else:
        q = select(ModelRecord).where(
            (ModelRecord.domain == payload.env_id) | (ModelRecord.domain.ilike(f"%{payload.env_id}%"))
        )
        res = await db.execute(q)
        records = res.scalars().all()
        for rec in records:
            path = rec.checkpoint_path or rec.weights_path
            if path and os.path.exists(path):
                entries.append({"id": rec.id, "name": rec.name, "path": path})

    if len(entries) < 2:
        runs_dir = "runs"
        if os.path.isdir(runs_dir):
            for root, _, files in os.walk(runs_dir):
                for f in files:
                    if f.endswith((".zip", ".pth")):
                        full_p = os.path.join(root, f)
                        if not any(e["path"] == full_p for e in entries):
                            name = f"{os.path.basename(root)}/{f}"
                            entries.append({"id": f"auto-{len(entries)+1}", "name": name, "path": full_p})
                            if len(entries) >= 6:
                                break
                if len(entries) >= 6:
                    break

    if len(entries) < 2:
        raise HTTPException(
            status_code=400,
            detail=f"At least 2 valid model checkpoints are required to run a tournament for '{payload.env_id}'. Found {len(entries)}."
        )

    result = await asyncio.to_thread(
        run_tournament_match,
        checkpoint_entries=entries,
        env_id=payload.env_id,
        n_episodes=payload.n_episodes,
    )

    if payload.update_champion and result.get("champion_id"):
        champ_id = result["champion_id"]
        champ_rec = await db.get(ModelRecord, champ_id)
        if champ_rec:
            all_in_domain = await db.execute(
                select(ModelRecord).where(ModelRecord.domain == champ_rec.domain)
            )
            for other in all_in_domain.scalars():
                other.is_champion = (other.id == champ_id)
            await db.commit()

    return result

