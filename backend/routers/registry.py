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
from backend.config import settings
from backend.logging_config import get_logger
from backend.models.experiment import Experiment
from backend.models.mission import Mission
from backend.models.model_registry import ModelRecord
from backend.schemas.experiment import ExperimentCreate, ExperimentRead, ExperimentUpdate
from backend.schemas.model_registry import (
    ModelInferRequest,
    ModelInferResponse,
    ModelRecordCreate,
    ModelRecordRead,
    ModelRecordUpdate,
    TournamentRequest,
    TournamentResponse,
)
from backend.evaluator.benchmark import is_lm_tournament_env, run_tournament_match
from backend.services.lora_infer import (
    LoraInferError,
    adapter_rel_from_path,
    resolve_infer_config,
    run_lora_infer,
)

logger = get_logger(__name__)
router = APIRouter(prefix="/registry", tags=["registry"])


def canonical_checkpoint_path(path: Optional[str]) -> Optional[str]:
    """Same zip under data/… vs ./data/… must compare equal."""
    if not path:
        return None
    return os.path.abspath(os.path.normpath(path))


def _missions_dir() -> str:
    return os.path.join(settings.data_path, "missions")


def mission_id_from_model(record: ModelRecord) -> Optional[str]:
    """Mission UUID stored on the record, or parsed from data/missions/<id>/..."""
    meta = record.extra_metadata or {}
    if isinstance(meta, dict):
        mid = meta.get("mission_id")
        if mid:
            return str(mid)
    path = record.checkpoint_path or record.weights_path or ""
    parts = os.path.normpath(path).split(os.sep)
    if "missions" in parts:
        i = parts.index("missions")
        if i + 1 < len(parts) and parts[i + 1]:
            return parts[i + 1]
    return None


async def _alive_mission_ids(db: AsyncSession) -> set:
    res = await db.execute(select(Mission.id))
    return set(res.scalars().all())


async def purge_models_for_mission(db: AsyncSession, mission_id: str) -> int:
    """Drop registry rows that belong to a deleted mission. Caller commits."""
    res = await db.execute(select(ModelRecord))
    n = 0
    for rec in res.scalars().all():
        if mission_id_from_model(rec) == mission_id:
            await db.delete(rec)
            n += 1
    return n


async def _prune_orphan_model_records(db: AsyncSession) -> None:
    """Remove models whose mission row is gone (delete used to leave them behind)."""
    alive = await _alive_mission_ids(db)
    res = await db.execute(select(ModelRecord))
    removed = False
    for rec in res.scalars().all():
        mid = mission_id_from_model(rec)
        if mid and mid not in alive:
            await db.delete(rec)
            removed = True
            logger.info(
                "registry: pruned orphan model %s (mission %s no longer exists)",
                rec.id, mid,
            )
    if removed:
        try:
            await db.commit()
        except Exception as e:
            logger.warning("Failed to commit orphan model prune: %s", e)


async def _prune_duplicate_model_records(db: AsyncSession) -> None:
    """Drop extra registry rows that point at the same checkpoint file.

    Auto-sync used string equality on checkpoint_path, so `data/missions/…`
    and `./data/missions/…` each got a row. Tournaments then ran the same
    zip twice (identical 20-seed scores, Seaquest 80942f33 / 601c2404).
    """
    res = await db.execute(select(ModelRecord))
    recs = list(res.scalars().all())
    by_key: dict = {}
    for rec in recs:
        key = canonical_checkpoint_path(rec.checkpoint_path or rec.weights_path)
        if not key:
            continue
        by_key.setdefault(key, []).append(rec)
    removed = False
    for group in by_key.values():
        if len(group) < 2:
            continue
        keep = max(
            group,
            key=lambda r: (
                bool(r.is_champion),
                float(r.best_metric_value) if r.best_metric_value is not None else float("-inf"),
                str(r.created_at or ""),
            ),
        )
        for rec in group:
            if rec.id == keep.id:
                rec.checkpoint_path = canonical_checkpoint_path(rec.checkpoint_path) or rec.checkpoint_path
                rec.weights_path = canonical_checkpoint_path(rec.weights_path) or rec.weights_path
                continue
            await db.delete(rec)
            removed = True
            logger.info(
                "registry: pruned duplicate model %s (same checkpoint as %s)",
                rec.id, keep.id,
            )
    if removed:
        try:
            await db.commit()
        except Exception as e:
            logger.warning("Failed to commit duplicate model prune: %s", e)



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
    """Scan data/missions for trained checkpoints and ensure they exist in ModelRecord.

    Skip directories whose mission row was deleted — leftover checkpoints used
    to reappear on the Models page after DELETE /missions/{id}.
    """
    missions_dir = _missions_dir()
    if not os.path.isdir(missions_dir):
        return

    try:
        res = await db.execute(select(ModelRecord.checkpoint_path).where(ModelRecord.checkpoint_path.is_not(None)))
        existing_paths = {
            canonical_checkpoint_path(p) for p in res.scalars().all() if p
        }
        existing_paths.discard(None)
    except Exception:
        existing_paths = set()

    try:
        alive = await _alive_mission_ids(db)
    except Exception:
        alive = set()

    added = False
    for cfg_path in sorted(glob.glob(os.path.join(missions_dir, "*", "checkpoints", "train_config.json"))):
        try:
            with open(cfg_path) as f:
                cfg = json.load(f)
            ckpt_dir = os.path.dirname(cfg_path)
            m_id = os.path.basename(os.path.dirname(ckpt_dir))
            if m_id not in alive:
                continue
            env_id = cfg.get("env_id")
            algo = cfg.get("algorithm", "RL")
            if not env_id:
                continue

            for fn in ("best_model.zip", "best_model.pth"):
                p = canonical_checkpoint_path(os.path.join(ckpt_dir, fn))
                if p and os.path.exists(p) and p not in existing_paths:
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

    await _refresh_model_scores_from_disk(db)


async def _refresh_model_scores_from_disk(db: AsyncSession) -> None:
    """Bump registry scores when checkpoints/best_score.txt is higher.

    Auto-sync only wrote the score on first insert. Pac-Man 9b49aa78 stayed
    at 246.67 on the models page after play and the mission both reached ~530.
    """
    try:
        res = await db.execute(select(ModelRecord).where(ModelRecord.checkpoint_path.is_not(None)))
        records = res.scalars().all()
    except Exception:
        return
    changed = False
    for rec in records:
        path = rec.checkpoint_path or rec.weights_path
        if not path:
            continue
        score_file = os.path.join(os.path.dirname(path), "best_score.txt")
        if not os.path.exists(score_file):
            continue
        try:
            score = float(open(score_file).read().strip())
        except Exception:
            continue
        current = rec.best_metric_value
        if current is None or score > float(current) + 1e-6:
            rec.best_metric_value = score
            changed = True
    if changed:
        try:
            await db.commit()
        except Exception as e:
            logger.warning("Failed to refresh model scores from disk: %s", e)


@router.get("/models", response_model=List[ModelRecordRead])
async def list_model_records(
    domain: Optional[str] = None,
    champion_only: bool = False,
    db: AsyncSession = Depends(get_db),
):
    await _auto_sync_disk_checkpoints(db)
    await _prune_orphan_model_records(db)
    await _prune_duplicate_model_records(db)
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
    await _refresh_model_scores_from_disk(db)
    await db.refresh(record)
    return record


@router.patch("/models/{model_id}", response_model=ModelRecordRead)
async def update_model_record(model_id: str, payload: ModelRecordUpdate, db: AsyncSession = Depends(get_db)):
    record = await db.get(ModelRecord, model_id)
    if not record:
        raise HTTPException(status_code=404, detail="Model record not found")
    fields = payload.model_dump(exclude_none=True)
    for k, v in fields.items():
        setattr(record, k, v)
    if fields.get("is_champion"):
        q = select(ModelRecord).where(
            ModelRecord.domain == record.domain,
            ModelRecord.id != record.id,
            ModelRecord.is_champion == True,  # noqa: E712
        )
        others = (await db.execute(q)).scalars().all()
        for other in others:
            other.is_champion = False
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


@router.post("/models/{model_id}/infer", response_model=ModelInferResponse)
async def infer_model(
    model_id: str,
    payload: ModelInferRequest,
    db: AsyncSession = Depends(get_db),
):
    """One-shot LoRA completion on Mini for non-game / routing champions."""
    record = await db.get(ModelRecord, model_id)
    if not record:
        raise HTTPException(status_code=404, detail="Model record not found")

    mission_plan = None
    mission_last_ckpt = None
    mid = mission_id_from_model(record)
    if mid:
        mission = await db.get(Mission, mid)
        if mission:
            mission_plan = mission.current_plan
            mission_last_ckpt = mission.last_checkpoint_path

    try:
        cfg = resolve_infer_config(
            checkpoint_path=record.checkpoint_path,
            weights_path=record.weights_path,
            extra_metadata=record.extra_metadata,
            mission_plan=mission_plan,
            mission_last_checkpoint=mission_last_ckpt,
            domain=record.domain or "",
        )
        result = await asyncio.to_thread(
            run_lora_infer,
            model_id=record.id,
            prompt=payload.prompt,
            cfg=cfg,
            max_tokens=payload.max_tokens,
            temperature=payload.temperature,
        )
    except LoraInferError as exc:
        raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc

    return ModelInferResponse(
        completion=result.completion,
        prompt_used=result.prompt_used,
        model_id=result.model_id,
        latency_ms=result.latency_ms,
        adapter=result.adapter,
        base_model=result.base_model,
    )


# ── Tournament Arena ───────────────────────────────────────────────────────────

def _lm_adapter_path(record: ModelRecord) -> Optional[str]:
    """Resolve a Mini-relative or local adapter path for LM tournament entries."""
    meta = record.extra_metadata if isinstance(record.extra_metadata, dict) else {}
    candidates = [
        meta.get("remote_checkpoint_path"),
        meta.get("adapter"),
        meta.get("adapter_path"),
        meta.get("adapter_rel"),
        record.checkpoint_path,
        record.weights_path,
    ]
    for c in candidates:
        if not c:
            continue
        s = str(c).strip()
        rel = adapter_rel_from_path(s)
        if rel:
            return rel
        # Absolute Mini home path without adapters/ substring edge cases
        if "/finetune/adapters/" in s.replace("\\", "/"):
            return s
        if os.path.exists(s):
            return s
    return None


def _add_tournament_entry(
    entries: List[dict],
    seen_paths: set,
    *,
    eid: str,
    name: str,
    path: Optional[str],
    lm_arena: bool = False,
    metric_name: Optional[str] = None,
    metric_value: Optional[float] = None,
) -> None:
    """Append a tournament candidate. LM arenas accept Mini ``adapters/…`` paths."""
    if not path:
        return
    raw = str(path).strip()
    if lm_arena:
        key = adapter_rel_from_path(raw) or raw
        local_ok = os.path.exists(raw)
        mini_ok = bool(adapter_rel_from_path(raw)) or "/finetune/adapters/" in raw.replace("\\", "/")
        if not local_ok and not mini_ok:
            return
        if key in seen_paths:
            return
        seen_paths.add(key)
        entries.append({
            "id": eid,
            "name": name,
            "path": key if mini_ok and not local_ok else (canonical_checkpoint_path(raw) or raw),
            "metric_name": metric_name,
            "metric_value": metric_value,
        })
        return

    key = canonical_checkpoint_path(raw)
    if not key or not os.path.exists(key) or key in seen_paths:
        return
    seen_paths.add(key)
    entries.append({"id": eid, "name": name, "path": key})


@router.post("/tournament", response_model=TournamentResponse)
async def run_tournament(payload: TournamentRequest, db: AsyncSession = Depends(get_db)):
    await _auto_sync_disk_checkpoints(db)
    await _prune_orphan_model_records(db)
    await _prune_duplicate_model_records(db)
    entries: List[dict] = []
    seen_paths: set = set()
    lm_arena = is_lm_tournament_env(payload.env_id)

    def _add_entry(eid: str, name: str, path: Optional[str], **kwargs) -> None:
        _add_tournament_entry(
            entries, seen_paths, eid=eid, name=name, path=path, lm_arena=lm_arena, **kwargs
        )

    def _add_record(rec: ModelRecord) -> None:
        path = rec.checkpoint_path or rec.weights_path
        if lm_arena:
            path = _lm_adapter_path(rec) or path
        _add_entry(
            rec.id,
            rec.name,
            path,
            metric_name=rec.best_metric_name,
            metric_value=rec.best_metric_value,
        )

    if payload.model_ids:
        for mid in payload.model_ids:
            rec = await db.get(ModelRecord, mid)
            if rec:
                _add_record(rec)
    else:
        q = select(ModelRecord).where(
            (ModelRecord.domain == payload.env_id) | (ModelRecord.domain.ilike(f"%{payload.env_id}%"))
        )
        res = await db.execute(q)
        records = res.scalars().all()
        for rec in records:
            _add_record(rec)

    # Fallback 1: auto-discover checkpoints from data/missions
    if len(entries) < 2:
        missions_dir = _missions_dir()
        if os.path.isdir(missions_dir):
            try:
                alive = await _alive_mission_ids(db)
            except Exception:
                alive = set()
            for cfg_path in sorted(glob.glob(os.path.join(missions_dir, "*", "checkpoints", "train_config.json"))):
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
                        if m_id not in alive:
                            continue
                        algo = cfg.get("algorithm", "RL")
                        for fn in ("best_model.zip", "best_model.pth", "last_model.zip"):
                            p = os.path.join(ckpt_dir, fn)
                            if os.path.exists(p):
                                _add_entry(
                                    f"mission-{m_id[:8]}-{fn.split('.')[0]}",
                                    f"{algo} ({m_id[:8]})",
                                    p,
                                )
                                break
                except Exception:
                    pass
                if len(entries) >= 6:
                    break

            # If still need candidates, search iter/ subdirectories
            if len(entries) < 2:
                for cfg_path in sorted(glob.glob(os.path.join(missions_dir, "*", "checkpoints", "train_config.json"))):
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
                            if m_id not in alive:
                                continue
                            algo = cfg.get("algorithm", "RL")
                            iter_dir = os.path.join(ckpt_dir, "iter")
                            if os.path.isdir(iter_dir):
                                for iter_f in sorted(os.listdir(iter_dir), reverse=True):
                                    if iter_f.endswith((".zip", ".pth")):
                                        iter_p = os.path.join(iter_dir, iter_f)
                                        before = len(entries)
                                        iter_lbl = iter_f.replace("checkpoint_iter_", "iter-")
                                        _add_entry(
                                            f"mission-{m_id[:8]}-{iter_lbl}",
                                            f"{algo} ({m_id[:8]} {iter_lbl})",
                                            iter_p,
                                        )
                                        if len(entries) > before and len(entries) >= 6:
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
                        name = f"{os.path.basename(root)}/{f}"
                        _add_entry(f"auto-{len(entries)+1}", name, full_p)
                        if len(entries) >= 6:
                            break
                if len(entries) >= 6:
                    break

    if len(entries) < 2:
        raise HTTPException(
            status_code=400,
            detail=f"At least 2 valid model checkpoints are required to run a tournament for '{payload.env_id}'. Found {len(entries)}."
        )

    # Overlay only — each checkpoint loads its own train_config env_kwargs
    # so a reward-shaped Pac-Man is scored like the model player, not on
    # whichever table the first zip happened to carry.
    overlay = {}
    if payload.env_id == "Tetris-v0":
        overlay["max_steps"] = 500

    result = await asyncio.to_thread(
        run_tournament_match,
        checkpoint_entries=entries,
        env_id=payload.env_id,
        n_episodes=payload.n_episodes,
        env_kwargs=overlay or None,
    )

    if payload.update_champion and result.get("champion_id"):
        champ_id = result["champion_id"]
        champ_rec = await db.get(ModelRecord, champ_id)
        if champ_rec:
            # Champions must record matching suite_rev (AstraEval tournament seeds).
            suite_rev = result.get("suite_rev")
            suite_id = result.get("suite_id") or "rl_tournament"
            meta = dict(champ_rec.extra_metadata or {})
            prior_rev = meta.get("suite_rev")
            if prior_rev is not None and suite_rev is not None and int(prior_rev) != int(suite_rev):
                raise HTTPException(
                    status_code=409,
                    detail=(
                        f"champion suite_rev mismatch: model has suite_rev={prior_rev}, "
                        f"tournament used suite_rev={suite_rev} (suite_id={suite_id}). "
                        f"Re-run on matching suite_rev or clear metadata."
                    ),
                )
            meta["suite_id"] = suite_id
            meta["suite_rev"] = suite_rev
            champ_rec.extra_metadata = meta
            all_in_domain = await db.execute(
                select(ModelRecord).where(ModelRecord.domain == champ_rec.domain)
            )
            for other in all_in_domain.scalars():
                other.is_champion = (other.id == champ_id)
            await db.commit()

    return result

