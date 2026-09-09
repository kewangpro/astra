from __future__ import annotations

import os
import re
from typing import Optional, List
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from backend.config import settings
from backend.logging_config import get_logger
from backend.database import get_db
from backend.models.mission import Mission, MissionStatus
from backend.models.manifest import RequirementManifest
from backend.models.approval import ApprovalGate, ApprovalStatus
from backend.schemas.mission import MissionCreate, MissionRead, MissionUpdate

router = APIRouter(prefix="/missions", tags=["missions"])

# A target may sit this far above a recipe's declared ceiling before creation is
# rejected — covers eval noise and small measurement drift, not a real gap.
_CEILING_TARGET_MARGIN = 0.005


def _reject_unreachable_target(task_type: str, target_metric: dict) -> None:
    """Reject a mission whose target exceeds the recipe's empirically-observed
    ceiling. Real incident: mission ce2828f4 (DPO, target pass_rate=0.85) ran
    700+ iterations / 20 days stuck at 0.833 — the model was already converged
    and no in-loop lever could close the last 0.017. Sister mission 43517dd5,
    same recipe with target 0.80, completed at 0.833 immediately."""
    if not task_type or not target_metric:
        return
    from backend.agent.code_generator import recipe_metric_ceiling

    ceiling = recipe_metric_ceiling(task_type)
    for name, tgt in target_metric.items():
        cap = ceiling.get(name)
        if cap is None:
            continue
        try:
            if float(tgt) > float(cap) + _CEILING_TARGET_MARGIN:
                raise HTTPException(
                    status_code=422,
                    detail=(
                        f"target {name}={tgt} exceeds the empirically-observed ceiling for the "
                        f"'{task_type}' recipe ({name}≈{cap}). Lower the target, or raise "
                        f"metric_ceiling in the recipe if that ceiling has genuinely been beaten."
                    ),
                )
        except (TypeError, ValueError):
            continue


def _infer_task_type_from_goal(goal: str, default: str = "rl") -> str:
    """Infer task_type from semantic keywords in the goal text."""
    g = goal.lower()
    if "rejection-sampling" in g or "rejection sampling" in g or re.search(r"\brft\b", g):
        return "rft"
    if "distill" in g or "distillation" in g:
        return "distill"
    if re.search(r"\bdpo\b", g):
        return "dpo"
    if re.search(r"\bgrpo\b", g):
        return "grpo"
    if "prompt" in g and any(k in g for k in ("optimi", "variant", "conductor", "routing")):
        return "prompt"
    if re.search(r"\bsft\b", g) or "supervised fine-tuning" in g:
        return "sft"
    if re.search(r"\bmlx[-_ ]lora\b", g):
        return "mlx_lora"
    if any(k in g for k in ("scikit-learn", "sklearn", "classifier", "randomforest", "logisticregression", "iris", "digits", "breast_cancer", "wine")):
        return "ml"
    return default


def _parse_target_metric(goal: str) -> dict:
    """Extract a target metric dict from free-text goal. Returns {} if nothing recognized."""
    # 1. Percentages: e.g. 90% accuracy, 90% pass rate, 90% pass_rate
    m = re.search(r"(\d+(?:\.\d+)?)\s*%\s*(accuracy|pass_?rate|pass\s+rate)", goal, re.IGNORECASE)
    if m:
        name = "pass_rate" if "pass" in m.group(2).lower() else "accuracy"
        return {name: float(m.group(1)) / 100}

    # 2. 'accuracy/pass_rate of X' or 'pass rate to/reach X'
    m = re.search(r"(accuracy|pass_?rate|pass\s+rate)\s+(?:of|to|reach|at\s+least|>=?)\s*(\d+(?:\.\d+)?)%?", goal, re.IGNORECASE)
    if m:
        val = float(m.group(2))
        name = "pass_rate" if "pass" in m.group(1).lower() else "accuracy"
        return {name: val if val <= 1.0 else val / 100}

    # 3. 'reach/achieve/target/hit X% [metric]'
    m = re.search(r"(?:reach|achieve|target|hit|hitting)\s+(?:a\s+)?(\d+(?:\.\d+)?)\s*%\s*([\w\s]+?)(?:\s+(?:in|per|on|within)\b|$)", goal, re.IGNORECASE)
    if m:
        metric_name = re.sub(r"\s+", "_", m.group(2).strip().lower())
        return {metric_name: float(m.group(1)) / 100}

    # 4. Standard reward / loss patterns
    m = re.search(r"(?:mean_?)?reward\s+of\s+(\d+(?:\.\d+)?)", goal, re.IGNORECASE)
    if m:
        return {"mean_reward": float(m.group(1))}
    m = re.search(r"(?:eval_)?loss\s+(?:of\s+|<=?\s*)(\d+(?:\.\d+)?)", goal, re.IGNORECASE)
    if m:
        return {"eval_loss": float(m.group(1))}

    # 5. Generic: "(achieve|reach|target|hit) {metric name} of {value}"
    m = re.search(
        r"(?:achieve|reach|target|hit)\s+([\w][\w\s]*?)\s+of\s+(\d+(?:\.\d+)?)",
        goal, re.IGNORECASE,
    )
    if m:
        metric_name = re.sub(r"\s+", "_", m.group(1).strip().lower())
        return {metric_name: float(m.group(2))}

    # 6. Generic: "(achieve|reach|target|hit) {value} {metric name}"
    m = re.search(
        r"(?:achieve|reach|target|hit)\s+(\d+(?:\.\d+)?)\s+([\w][\w\s]*?)(?:\s+(?:in|per|on|within)\b|$)",
        goal, re.IGNORECASE,
    )
    if m:
        metric_name = re.sub(r"\s+", "_", m.group(2).strip().lower())
        return {metric_name: float(m.group(1))}
    return {}


logger = get_logger(__name__)


#: Goal metrics that only a fine-tune/prompt mission can produce. An RL mission
#: scores via _run_goal_metric_eval, which needs a Gym env and an SB3/actor-critic
#: checkpoint; it has no way to produce a routing pass_rate at all.
_FINETUNE_ONLY_METRICS = frozenset({"pass_rate"})

#: task_type is Optional[str] = "rl" in the schema, so an omitted field and an
#: explicit "rl" both arrive as "rl". _infer_task_type_from_goal catches goals
#: that NAME their method, but an unnamed one ("fine-tune the model to 90% pass
#: rate") falls through to the default and would dispatch down the RL path —
#: the same failure as mission 6d999c84, now originating server-side rather than
#: from the UI. Before task_type became optional this was a 422; keep it loud.
_RL_TASK_TYPES = frozenset({"rl"})


def _reject_incoherent_task_type(task_type: str, target_metric: dict) -> None:
    """422 when the task type cannot produce the metric the goal asks for.

    Cheap to check at creation, and the alternative is a mission that dispatches,
    trains something, and reports success without ever producing the number its
    goal named.
    """
    if task_type not in _RL_TASK_TYPES or not target_metric:
        return
    clash = _FINETUNE_ONLY_METRICS & set(target_metric)
    if not clash:
        return
    name = sorted(clash)[0]
    raise HTTPException(
        status_code=422,
        detail=(
            f"task_type 'rl' cannot produce '{name}' — RL missions are scored by "
            f"rollout in a Gym environment and have no routing eval. The goal text "
            f"did not name a method, so task_type fell back to the 'rl' default. "
            f"Set task_type explicitly (rft / distill / dpo / grpo / prompt / sft / "
            f"mlx_lora / ml), or name the method in the goal."
        ),
    )


@router.post("", response_model=MissionRead, status_code=status.HTTP_201_CREATED)
async def create_mission(payload: MissionCreate, db: AsyncSession = Depends(get_db)):
    payload_dict = payload.model_dump()
    if not payload_dict.get("target_metric"):
        payload_dict["target_metric"] = _parse_target_metric(payload.goal)

    # Reconcile task_type if omitted or if "rl" was submitted as default but goal indicates another paradigm
    submitted_type = payload_dict.get("task_type")
    inferred_type = _infer_task_type_from_goal(payload.goal, default="rl")
    if not submitted_type or submitted_type == "auto":
        payload_dict["task_type"] = inferred_type
    elif submitted_type == "rl" and inferred_type != "rl":
        # "rl" is the schema default, so an explicit rl and an omitted field are
        # indistinguishable here — the override exists because the frontend used
        # to send "rl" for everything (mission 6d999c84 was a rejection-sampling
        # goal dispatched down the RL path, completing in 9 minutes with no
        # metric). Log it: silently reinterpreting a caller's stated intent is
        # the one case where this rule is wrong, and a line in the log is the
        # difference between "astra chose for me" and "astra ignored me".
        logger.info(
            "Mission create: task_type 'rl' overridden to '%s' from goal text — "
            "pass an explicit non-rl task_type, or 'auto', to silence this",
            inferred_type,
        )
        payload_dict["task_type"] = inferred_type

    _reject_incoherent_task_type(
        payload_dict.get("task_type", ""), payload_dict.get("target_metric") or {}
    )
    _reject_unreachable_target(
        payload_dict.get("task_type", ""), payload_dict.get("target_metric") or {}
    )
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
    from sqlalchemy.orm.attributes import flag_modified
    for k, v in payload.model_dump(exclude_none=True).items():
        if k == "status" and v is not None:
            setattr(mission, k, v.value if isinstance(v, MissionStatus) else v)
        else:
            setattr(mission, k, v)
            if isinstance(v, dict):
                flag_modified(mission, k)
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

    # Cancel running loop task
    from backend.routers.agent import _running_tasks
    task = _running_tasks.pop(mission_id, None)
    if task and not task.done():
        task.cancel()

    # Reject all pending approval gates so the loop (if still polling) unblocks and exits
    result = await db.execute(
        select(ApprovalGate).where(
            ApprovalGate.mission_id == mission_id,
            ApprovalGate.status == ApprovalStatus.PENDING.value,
        )
    )
    for gate in result.scalars().all():
        gate.status = ApprovalStatus.REJECTED.value
        gate.reviewer_note = "mission deleted"

    await db.delete(mission)
    await db.commit()
