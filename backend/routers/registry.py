from __future__ import annotations

from typing import Optional, List
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from backend.database import get_db
from backend.models.experiment import Experiment
from backend.models.model_registry import ModelRecord
from backend.schemas.experiment import ExperimentCreate, ExperimentRead, ExperimentUpdate
from backend.schemas.model_registry import ModelRecordCreate, ModelRecordRead, ModelRecordUpdate

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
