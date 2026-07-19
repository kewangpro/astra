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


# Valid SB3 PPO kwargs — strip everything else from RL hyperparameters
_VALID_PPO_KWARGS = {
    "learning_rate", "n_steps", "batch_size", "n_epochs", "gamma",
    "gae_lambda", "clip_range", "clip_range_vf", "ent_coef", "vf_coef",
    "max_grad_norm", "target_kl",
}

# Rename map: LLM-invented keys → canonical SB3 names
_PPO_RENAMES = {
    "entropy_coeff": "ent_coef",
    "entropy_coef": "ent_coef",
    "entropy": "ent_coef",
    "clip_ratio": "clip_range",
}

# Valid actor_critic kwargs
_VALID_AC_KWARGS = {
    "learning_rate", "gamma", "episodes", "batch_size",
    "replay_buffer_size", "epsilon_min", "epsilon_decay", "telemetry_interval",
}


def _clean_rl_hyperparams(raw: dict, trainer_type: str = "") -> dict:
    """Rename invalid keys and drop trainer-irrelevant kwargs."""
    if trainer_type == "actor_critic":
        return {k: v for k, v in raw.items() if k in _VALID_AC_KWARGS}
    # The lookahead_* custom trainers (lookahead_dqn/ppo/a2c) are hand-rolled
    # PyTorch training loops, not real SB3 API surfaces — there's no fixed
    # kwarg allowlist to validate against the way there is for genuine SB3
    # PPO, so pass their hyperparameters through unfiltered rather than
    # incorrectly stripping them against the SB3-PPO-specific allowlist below.
    if trainer_type.startswith("lookahead_"):
        return dict(raw)
    cleaned = {}
    for k, v in raw.items():
        k = _PPO_RENAMES.get(k, k)
        if k in _VALID_PPO_KWARGS:
            cleaned[k] = v
    return cleaned


def _infer_domain(plan: dict, task_type: str, goal: str) -> str:
    """Derive a meaningful domain name from the plan, not the raw goal string."""
    if plan.get("domain") and plan["domain"].lower() not in ("train", "unknown", ""):
        return plan["domain"]
    if task_type == "rl":
        env_id = plan.get("env_id") or plan.get("hyperparameters", {}).get("env_id", "")
        if env_id:
            # "CartPole-v1" → "CartPole"
            return env_id.split("-")[0]
    if task_type in ("ml", "sft"):
        ds = (plan.get("dataset_path") or
              plan.get("hyperparameters", {}).get("dataset_path", ""))
        if ds:
            return ds.replace("load_", "").split(".")[0]
    return task_type.upper()


def _build_recipe_content(mission: Mission, score: Optional[float], lessons: list[dict]) -> dict:
    plan = mission.current_plan or {}
    task_type = (plan.get("task_type") or mission.task_type or "unknown").lower()
    trainer_type = plan.get("trainer_type", "")
    # For actor_critic missions, surface the trainer as the algorithm name
    algorithm = trainer_type if trainer_type == "actor_critic" else plan.get("algorithm", "unknown")
    raw_hp = plan.get("hyperparameters", {})

    # Clean hyperparameters: strip trainer-irrelevant keys
    if task_type == "rl":
        hyperparams = _clean_rl_hyperparams(raw_hp, trainer_type=trainer_type)
    else:
        hyperparams = {k: v for k, v in raw_hp.items() if k != "dataset_path"}

    curriculum = plan.get("curriculum_phases")
    lesson_notes = [l["text"][:120] for l in lessons[:3]] if lessons else []
    domain = _infer_domain(plan, task_type, mission.goal)

    score_str = f" Achieved score {score:.4f}." if score else ""
    content: dict = {
        "task_type": task_type.upper(),
        "domain": domain,
        "algorithm": algorithm,
        "description": (
            f"Auto-crystallized recipe from mission {mission.id[:8]}.{score_str}"
        ),
        "hyperparameters": hyperparams,
        "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "provenance": {
            "mission_id": mission.id,
            "iterations": mission.current_iteration,
            "best_score": score,
        },
    }

    # For actor_critic/lookahead_* missions, record trainer_type explicitly.
    if trainer_type:
        content["trainer_type"] = trainer_type

    # For RL recipes, surface env_id as a top-level field
    if task_type == "rl":
        env_id = (plan.get("env_id") or
                  raw_hp.get("env_id") or
                  raw_hp.get("dataset_path"))
        if env_id:
            content["env_id"] = env_id

    # For ML/SFT, surface the dataset name
    if task_type in ("ml", "sft"):
        dataset = (plan.get("dataset_path") or
                   raw_hp.get("dataset_path", "").replace("load_", ""))
        if dataset:
            content["dataset"] = dataset

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

        task_type = resolved_plan.get("task_type") or mission.task_type
        # Real incident: this used to be `resolved_plan.get("domain") or
        # mission.goal.split()[0]` — a naive fallback that grabbed the first
        # WORD of the goal string ("Train a Tetris-v0 DQN agent..." → "Train"),
        # completely bypassing _infer_domain() (already correctly used inside
        # _build_recipe_content() below) and then overwriting its correct
        # result. Produced a recipe with domain="Train" and a filename literally
        # named after the bug (train_rl_v13.yaml instead of tetris_rl_v1.yaml).
        domain = _infer_domain(resolved_plan, task_type, mission.goal)

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
