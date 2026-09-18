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

    # 2. Try disk
    for ext in (".yaml", ".yml"):
        fpath = os.path.join(settings.recipes_path, f"{clean_name}{ext}")
        if os.path.exists(fpath):
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
    target_metric = dict(payload.target_metric)
    if not target_metric and recipe_content:
        target_metric = dict(recipe_content.get("target_metric") or {})

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
                current_plan["hyperparameters"] = first_stage["hyperparameters"]

    return {
        "goal": goal,
        "task_type": task_type,
        "target_metric": target_metric,
        "autonomy_mode": payload.autonomy_mode,
        "current_plan": current_plan,
    }
