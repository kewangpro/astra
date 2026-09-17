from __future__ import annotations

import asyncio
import glob
import json
import os
from typing import Optional, List
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from backend.database import get_db
from backend.logging_config import get_logger
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
from backend.evaluator.benchmark import run_tournament_match, _load_env_kwargs

logger = get_logger(__name__)
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


async def _auto_sync_disk_checkpoints(db: AsyncSession) -> None:
    """Scan data/missions for trained checkpoints and ensure they exist in ModelRecord."""
    missions_dir = "data/missions"
    if not os.path.isdir(missions_dir):
        return

    try:
        res = await db.execute(select(ModelRecord.checkpoint_path).where(ModelRecord.checkpoint_path.is_not(None)))
        existing_paths = set(res.scalars().all())
    except Exception:
        existing_paths = set()

    added = False
    for cfg_path in sorted(glob.glob(f"{missions_dir}/*/checkpoints/train_config.json")):
        try:
            with open(cfg_path) as f:
                cfg = json.load(f)
            ckpt_dir = os.path.dirname(cfg_path)
            m_id = os.path.basename(os.path.dirname(ckpt_dir))
            env_id = cfg.get("env_id")
            algo = cfg.get("algorithm", "RL")
            if not env_id:
                continue

            for fn in ("best_model.zip", "best_model.pth"):
                p = os.path.join(ckpt_dir, fn)
                if os.path.exists(p) and p not in existing_paths:
                    score = None
                    metric_name = "task_success" if "agent" in env_id.lower() else "mean_reward"
                    score_file = os.path.join(ckpt_dir, "best_score.txt")
                    if os.path.exists(score_file):
                        try:
                            score = float(open(score_file).read().strip())
                        except Exception:
                            pass

                    framework = "torch" if fn.endswith(".pth") else ("mlx" if "mlx" in algo.lower() else "stable-baselines3")
                    name = f"{algo} {env_id} ({m_id[:8]})"
                    rec = ModelRecord(
                        name=name,
                        domain=env_id,
                        framework=framework,
                        architecture=algo,
                        checkpoint_path=p,
                        weights_path=p,
                        best_metric_name=metric_name,
                        best_metric_value=score,
                        is_champion=False,
                        extra_metadata={
                            "mission_id": m_id,
                            "env_kwargs": cfg.get("env_kwargs", {}),
                            "trainer_type": cfg.get("trainer_type", ""),
                        },
                    )
                    db.add(rec)
                    existing_paths.add(p)
                    added = True
        except Exception:
            pass

    if added:
        try:
            await db.commit()
        except Exception as e:
            logger.warning("Failed to commit auto-synced checkpoints: %s", e)


@router.get("/models", response_model=List[ModelRecordRead])
async def list_model_records(
    domain: Optional[str] = None,
    champion_only: bool = False,
    db: AsyncSession = Depends(get_db),
):
    await _auto_sync_disk_checkpoints(db)
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
    await _auto_sync_disk_checkpoints(db)
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

    # Fallback 1: auto-discover checkpoints from data/missions
    if len(entries) < 2:
        missions_dir = "data/missions"
        if os.path.isdir(missions_dir):
            for cfg_path in sorted(glob.glob(f"{missions_dir}/*/checkpoints/train_config.json")):
                try:
                    with open(cfg_path) as f:
                        cfg = json.load(f)
                    m_env = cfg.get("env_id") or ""
                    if (
                        m_env == payload.env_id
                        or payload.env_id.lower() in m_env.lower()
                        or m_env.lower() in payload.env_id.lower()
                    ):
                        ckpt_dir = os.path.dirname(cfg_path)
                        m_id = os.path.basename(os.path.dirname(ckpt_dir))
                        algo = cfg.get("algorithm", "RL")
                        for fn in ("best_model.zip", "best_model.pth", "last_model.zip"):
                            p = os.path.join(ckpt_dir, fn)
                            if os.path.exists(p) and not any(e["path"] == p for e in entries):
                                entries.append({
                                    "id": f"mission-{m_id[:8]}-{fn.split('.')[0]}",
                                    "name": f"{algo} ({m_id[:8]})",
                                    "path": p,
                                })
                                break
                except Exception:
                    pass
                if len(entries) >= 6:
                    break

            # If still need candidates, search iter/ subdirectories
            if len(entries) < 2:
                for cfg_path in sorted(glob.glob(f"{missions_dir}/*/checkpoints/train_config.json")):
                    try:
                        with open(cfg_path) as f:
                            cfg = json.load(f)
                        m_env = cfg.get("env_id") or ""
                        if (
                            m_env == payload.env_id
                            or payload.env_id.lower() in m_env.lower()
                            or m_env.lower() in payload.env_id.lower()
                        ):
                            ckpt_dir = os.path.dirname(cfg_path)
                            m_id = os.path.basename(os.path.dirname(ckpt_dir))
                            algo = cfg.get("algorithm", "RL")
                            iter_dir = os.path.join(ckpt_dir, "iter")
                            if os.path.isdir(iter_dir):
                                for iter_f in sorted(os.listdir(iter_dir), reverse=True):
                                    if iter_f.endswith((".zip", ".pth")):
                                        iter_p = os.path.join(iter_dir, iter_f)
                                        if not any(e["path"] == iter_p for e in entries):
                                            iter_lbl = iter_f.replace("checkpoint_iter_", "iter-")
                                            entries.append({
                                                "id": f"mission-{m_id[:8]}-{iter_lbl}",
                                                "name": f"{algo} ({m_id[:8]} {iter_lbl})",
                                                "path": iter_p,
                                            })
                                            if len(entries) >= 6:
                                                break
                    except Exception:
                        pass
                    if len(entries) >= 6:
                        break

    # Fallback 2: auto-discover from runs/ directory if it exists
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

    # Resolve env_kwargs from candidates (e.g. obs_type='features' for Snake)
    env_kwargs = None
    for e in entries:
        kw = _load_env_kwargs(e["path"])
        if kw:
            env_kwargs = kw
            break
    if not env_kwargs and payload.env_id == "Snake-v0":
        env_kwargs = {"obs_type": "features", "max_steps": 2000}
    if payload.env_id == "Tetris-v0":
        env_kwargs = dict(env_kwargs or {})
        env_kwargs["max_steps"] = min(env_kwargs.get("max_steps", 500), 500)

    result = await asyncio.to_thread(
        run_tournament_match,
        checkpoint_entries=entries,
        env_id=payload.env_id,
        n_episodes=payload.n_episodes,
        env_kwargs=env_kwargs,
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

