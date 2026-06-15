"""
RecipeCrystallizer — Step 5.1.

Distills a successfully completed mission into a reusable YAML recipe:
  1. Reads the mission's final plan, best metric, and iteration count.
  2. Queries vector memory for lessons learned during the run.
  3. Constructs a structured recipe dict.
  4. Persists the record in the DB and writes the YAML to recipes/.
  5. Indexes the recipe in the semantic recipe library (Step 5.2).
"""
from __future__ import annotations

import os
import re
from datetime import datetime, timezone
from typing import Optional

import yaml
from sqlalchemy import select

from backend.config import settings
from backend.database import AsyncSessionLocal
from backend.logging_config import get_logger
from backend.models.mission import Mission
from backend.models.recipe import RecipeRecord
from backend.services import vector_memory
from backend.services import recipe_library

logger = get_logger(__name__)

_GOLDEN_WIN_THRESHOLD = 3  # consecutive wins needed for Golden status


def _slugify(text: str) -> str:
    """Convert arbitrary text to a safe slug."""
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text[:60].strip("_")


def _build_recipe_name(domain: str, task_type: str, version: int) -> str:
    return f"{_slugify(domain)}_{_slugify(task_type)}_v{version}"


def _next_version(existing_names: list[str], base: str) -> int:
    """Find the next version number for a recipe base name."""
    versions = []
    for name in existing_names:
        if name.startswith(base + "_v"):
            suffix = name[len(base) + 2:]
            if suffix.isdigit():
                versions.append(int(suffix))
    return max(versions, default=0) + 1


def _build_recipe_content(mission: Mission, score: Optional[float], lessons: list[dict]) -> dict:
    plan = mission.current_plan or {}
    hyperparams = plan.get("hyperparameters", {})
    curriculum = plan.get("curriculum_phases")
    algorithm = plan.get("algorithm", "unknown")

    # Distil top lessons into description annotations
    lesson_notes = [l["text"][:120] for l in lessons[:3]] if lessons else []

    content: dict = {
        "task_type": (plan.get("task_type") or mission.task_type or "unknown").upper(),
        "domain": mission.goal.split()[0] if mission.goal else "unknown",
        "algorithm": algorithm,
        "description": (
            f"Auto-crystallized recipe from mission {mission.id[:8]}. "
            f"Achieved score {score:.4f}." if score else
            f"Auto-crystallized recipe from mission {mission.id[:8]}."
        ),
        "hyperparameters": hyperparams,
        "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "provenance": {
            "mission_id": mission.id,
            "iterations": mission.current_iteration,
            "best_score": score,
        },
    }
    if curriculum:
        content["curriculum"] = {"phases": curriculum}
    if lesson_notes:
        content["lessons"] = lesson_notes
    if mission.target_metric:
        content["target_metric"] = mission.target_metric

    return content


async def crystallize(
    mission_id: str,
    *,
    plan: Optional[dict] = None,
    score: Optional[float] = None,
) -> Optional[RecipeRecord]:
    """
    Distil a completed mission into a RecipeRecord.

    Parameters
    ----------
    mission_id:
        The ID of the completed mission.
    plan:
        The final plan dict (avoids a second DB read if caller has it).
    score:
        The final best metric value (avoids a second DB read if caller has it).

    Returns the persisted RecipeRecord, or None on failure.
    """
    async with AsyncSessionLocal() as session:
        mission = await session.get(Mission, mission_id)
        if not mission:
            logger.error("Crystallizer: mission %s not found", mission_id)
            return None

        # Merge caller-supplied values with DB values
        if plan:
            mission.current_plan = plan
        resolved_plan = mission.current_plan or {}
        resolved_score = score if score is not None else (
            float(mission.best_metric_value) if mission.best_metric_value else None
        )

        domain = resolved_plan.get("domain") or mission.goal.split()[0]
        task_type = resolved_plan.get("task_type") or mission.task_type

        # Retrieve lessons learned during this run
        lessons = vector_memory.query_lessons(
            mission.goal,
            domain=domain,
            n_results=5,
        )

        # Build recipe content
        content = _build_recipe_content(mission, resolved_score, lessons)
        content["domain"] = domain

        # Determine unique name
        result = await session.execute(select(RecipeRecord.name))
        existing_names = list(result.scalars().all())
        base = f"{_slugify(domain)}_{_slugify(task_type)}"
        version = _next_version(existing_names, base)
        name = _build_recipe_name(domain, task_type, version)
        content["name"] = name
        content["version"] = f"{version}.0.0"

        # Persist YAML to disk
        recipes_dir = settings.recipes_path
        os.makedirs(recipes_dir, exist_ok=True)
        yaml_path = os.path.join(recipes_dir, f"{name}.yaml")
        with open(yaml_path, "w") as f:
            yaml.dump(content, f, default_flow_style=False, sort_keys=False)
        logger.info("Crystallizer: wrote %s", yaml_path)

        # Persist RecipeRecord to DB
        record = RecipeRecord(
            name=name,
            version=f"{version}.0.0",
            domain=domain,
            task_type=task_type,
            description=content.get("description"),
            hyperparameters=resolved_plan.get("hyperparameters", {}),
            curriculum=content.get("curriculum"),
            full_content=content,
            mission_id=mission_id,
            score=resolved_score,
            target_metric=mission.target_metric,
            generation=1,  # crystallized
        )
        session.add(record)
        await session.commit()
        await session.refresh(record)

    # Index in semantic recipe library
    try:
        recipe_library.index_recipe(record)
    except Exception as exc:
        logger.warning("Crystallizer: recipe library indexing failed: %s", exc)

    logger.info("Crystallizer: recipe '%s' crystallized (score=%s)", name, resolved_score)
    return record
