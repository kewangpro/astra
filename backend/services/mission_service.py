"""
Mission service — centralized preparation, canonical goal formatting,
and recipe resolution for Astra missions.
"""
from __future__ import annotations

import os
import re
from typing import Optional, Any
import yaml
from fastapi import HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.config import settings
from backend.logging_config import get_logger
from backend.models.recipe import RecipeRecord
from backend.schemas.mission import MissionCreate

logger = get_logger(__name__)


async def resolve_recipe(recipe_name: str, db: AsyncSession) -> tuple[dict[str, Any], str, str]:
    """
    Resolve a recipe by name from DB or disk.
    Returns (content_dict, domain, task_type).
    Raises HTTPException(404) if not found.
    """
    clean_name = recipe_name.removesuffix(".yaml").removesuffix(".yml")

    # 1. Try DB first
    record = await db.get(RecipeRecord, clean_name)
    if not record:
        q = select(RecipeRecord).where(RecipeRecord.name == clean_name)
        res = await db.execute(q)
        record = res.scalars().first()

    if record:
        content = record.full_content if isinstance(getattr(record, "full_content", None), dict) else {}
        domain = getattr(record, "domain", "rl")
        task_type = getattr(record, "task_type", "rl") or "rl"
        if getattr(record, "description", None) and "description" not in content:
            content["description"] = record.description
        if getattr(record, "target_metric", None) and "target_metric" not in content:
            content["target_metric"] = record.target_metric
        return content, domain, task_type

    # 2. Try disk (platform recipes/ then ASTRA_PRODUCT_RECIPES packs)
    from backend.services.recipe_paths import find_recipe_file

    fpath = find_recipe_file(clean_name)
    if fpath:
        with open(fpath, "r") as f:
            content = yaml.safe_load(f) or {}
        domain = content.get("domain", "rl")
        raw_task = content.get("task_type")
        if not raw_task:
            task_type = "rl" if any(k in clean_name for k in ("env", "game", "snake", "tetris", "minatar")) else "rft"
        else:
            task_type = str(raw_task).lower()
        return content, domain, task_type

    raise HTTPException(status_code=404, detail=f"Recipe '{clean_name}' not found in DB or disk")


def _metric_val_in_text(val: Any, text: str) -> bool:
    if val is None:
        return False
    s = str(val)
    if s in text:
        return True
    try:
        f = float(val)
        if f.is_integer() and str(int(f)) in text:
            return True
    except (ValueError, TypeError):
        pass
    return False


_USE_CASE_IN_GOAL_RE = re.compile(
    r"(?:"
    r"--use-case\s+([A-Za-z_][\w.]*)"
    r"|use[_-]?case\s*[:=]\s*([A-Za-z_][\w.]*)"
    r"|use[_-]?case\s+module\s+([A-Za-z_][\w.]*)"
    r")",
    re.IGNORECASE,
)


def infer_use_case_from_goal(goal: Optional[str]) -> Optional[str]:
    """Extract an explicit UseCasePlugin module from mission creation goal text.

    Matches markers such as ``--use-case ensemble``, ``use_case: ensemble``,
    or ``use-case module ensemble``. Does not maintain a product alias table —
    free-form product names are resolved by the Lead Agent at plan time.
    """
    if not goal:
        return None
    m = _USE_CASE_IN_GOAL_RE.search(goal)
    if not m:
        return None
    return next((g for g in m.groups() if g), None)


def resolve_mission_use_case(
    payload_use_case: Optional[str],
    goal: Optional[str],
) -> Optional[str]:
    """Mission-create binding: explicit field, else marker in the goal prompt."""
    raw = (payload_use_case or "").strip() or infer_use_case_from_goal(goal) or ""
    return raw or None


def _inject_use_case_into_plan(plan: dict[str, Any], use_case: str) -> None:
    """Put use_case on plan hyperparameters (and staged HPs) without clobbering other keys."""
    hp = dict(plan.get("hyperparameters") or {})
    hp["use_case"] = use_case
    plan["hyperparameters"] = hp
    stages = plan.get("stages")
    if isinstance(stages, list):
        for stage in stages:
            if not isinstance(stage, dict):
                continue
            shp = dict(stage.get("hyperparameters") or {})
            shp["use_case"] = use_case
            stage["hyperparameters"] = shp


def format_canonical_goal(
    goal: Optional[str],
    recipe_content: Optional[dict[str, Any]],
    task_type: str,
    target_metric: dict[str, Any],
    recipe_name: Optional[str] = None,
    domain: Optional[str] = None,
) -> str:
    """
    Guarantees every mission displays the standardized format:
      'Train a <env_id> <algo> agent to achieve <target_metric>'
    """
    # 1. If explicit goal already contains the target value, return as-is
    if goal and target_metric and any(_metric_val_in_text(v, goal) for v in target_metric.values()):
        return goal.strip()

    # 2. Format target string (e.g. "25.0 score" or "300 lines_cleared")
    target_parts = [f"{v} {k}" for k, v in target_metric.items()]
    target_str = ", ".join(target_parts) if target_parts else ""

    # If explicit goal is provided, append target if not present
    if goal:
        return f"{goal.rstrip('.')} to achieve {target_str}" if target_str else goal.strip()

    rc = recipe_content if isinstance(recipe_content, dict) else {}

    # 3. Canonical RL pattern
    if task_type == "rl":
        env_id = rc.get("env_id") or domain or "environment"
        algo = rc.get("algorithm") or "agent"
        desc = rc.get("description")
        if rc.get("env_id") and rc.get("algorithm"):
            if target_str:
                return f"Train a {env_id} {algo} agent to achieve {target_str}"
            return f"Train a {env_id} {algo} agent"
        elif desc:
            return f"{desc.rstrip('.')} to achieve {target_str}" if target_str else desc.strip()
        else:
            if target_str:
                return f"Train a {env_id} {algo} agent to achieve {target_str}"
            return f"Train a {env_id} {algo} agent"

    # 4. Multi-stage or Non-RL pattern
    if rc.get("description"):
        base = rc["description"].rstrip(".")
    else:
        base = f"Execute recipe {recipe_name or 'custom'}"

    return f"{base} to achieve {target_str}." if target_str else f"{base}."


async def prepare_mission_params(payload: MissionCreate, db: AsyncSession) -> dict[str, Any]:
    """
    Unified preparation of Mission attributes from MissionCreate payload,
    handling optional recipe seeding, goal canonicalization, plan generation,
    and target metric normalization.
    """
    recipe_content: Optional[dict[str, Any]] = None
    domain: Optional[str] = None
    clean_recipe_name: Optional[str] = None

    if payload.recipe:
        clean_recipe_name = payload.recipe.removesuffix(".yaml").removesuffix(".yml")
        recipe_content, domain, inferred_task = await resolve_recipe(clean_recipe_name, db)
    else:
        inferred_task = "rl"

    # Task type resolution
    task_type = (payload.task_type or inferred_task or "rl").lower()

    # Target metric resolution & normalization
    target_metric = dict(payload.target_metric or {})
    if not target_metric and recipe_content:
        target_metric = dict(recipe_content.get("target_metric") or {})
    # Recipes often declare metric_ceiling (empirical max) without a separate
    # target_metric — Recipes UI one-click dispatch used to leave {} and stall
    # completion. Fall back to ceiling so Command Center and Recipes agree.
    if not target_metric and recipe_content:
        ceiling = recipe_content.get("metric_ceiling")
        if isinstance(ceiling, dict) and ceiling:
            target_metric = dict(ceiling)

    # Normalize old {"metric": "score", "target": 15.0} format if present
    if isinstance(target_metric, dict) and "metric" in target_metric and "target" in target_metric:
        target_metric = {str(target_metric["metric"]): target_metric["target"]}

    # Build canonical goal
    goal = format_canonical_goal(
        goal=payload.goal,
        recipe_content=recipe_content,
        task_type=task_type,
        target_metric=target_metric,
        recipe_name=clean_recipe_name,
        domain=domain,
    )

    # Build current_plan if recipe is present
    current_plan: Optional[dict[str, Any]] = None
    if clean_recipe_name and recipe_content:
        current_plan = {"recipe": clean_recipe_name, "task_type": task_type}
        stages = recipe_content.get("stages")
        if stages:
            current_plan["stages"] = stages
            current_plan["stage_index"] = 0
            current_plan["stage_checkpoints"] = {}
            first_stage = stages[0]
            current_plan["active_task_type"] = first_stage.get("task", "sft")
            if first_stage.get("recipe"):
                current_plan["recipe"] = first_stage["recipe"]
            if first_stage.get("hyperparameters"):
                current_plan["hyperparameters"] = dict(first_stage["hyperparameters"])

    # Product plugin binds at mission create (field or goal marker), not recipe HP.
    use_case = resolve_mission_use_case(payload.use_case, goal)
    if use_case:
        if current_plan is None:
            current_plan = {"hyperparameters": {"use_case": use_case}}
        else:
            _inject_use_case_into_plan(current_plan, use_case)
    elif current_plan is not None:
        # Drop any recipe-stamped use_case so it cannot masquerade as create binding.
        hp = dict(current_plan.get("hyperparameters") or {})
        if "use_case" in hp:
            hp.pop("use_case", None)
            if hp:
                current_plan["hyperparameters"] = hp
            else:
                current_plan.pop("hyperparameters", None)
        for stage in current_plan.get("stages") or []:
            if isinstance(stage, dict) and isinstance(stage.get("hyperparameters"), dict):
                stage["hyperparameters"] = {
                    k: v for k, v in stage["hyperparameters"].items() if k != "use_case"
                }

    # AstraData / AstraEval bind axes (stamp from recipe when present).
    if recipe_content and current_plan is not None:
        ds_id = recipe_content.get("dataset_id")
        ds_rev = recipe_content.get("dataset_rev")
        suite_id = recipe_content.get("eval_suite") or recipe_content.get("suite_id")
        suite_rev = recipe_content.get("suite_rev")
        if ds_id:
            current_plan["dataset_id"] = str(ds_id)
            if ds_rev is not None:
                current_plan["dataset_rev"] = int(ds_rev)
        if suite_id:
            current_plan["suite_id"] = str(suite_id)
            if suite_rev is not None:
                current_plan["suite_rev"] = int(suite_rev)

        # Ensemble FailSlice → focus_cases when use_case bound and bank has a slice.
        if use_case == "ensemble":
            try:
                from astradata.fail_slice import FailSliceStore

                csv = FailSliceStore().focus_cases_csv(suite_id="routing_bare")
                if csv:
                    hp = dict(current_plan.get("hyperparameters") or {})
                    if not hp.get("focus_cases"):
                        hp["focus_cases"] = csv
                        current_plan["hyperparameters"] = hp
                        current_plan["fail_slice_focus"] = True
            except Exception:
                pass

    return {
        "goal": goal,
        "task_type": task_type,
        "target_metric": target_metric,
        "autonomy_mode": payload.autonomy_mode,
        "current_plan": current_plan,
    }
