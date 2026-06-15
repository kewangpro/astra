"""
Strategy Evolution Engine — Step 5.3.

Implements:
  - MutationOperator: perturbs numeric hyperparameters within bounds.
  - SelectionPolicy: promotes a child recipe only if it beats its parent.
  - GenePool: aggregates top-performing recipes per domain as candidates.
  - GoldenPromoter: awards "Golden" status after N consecutive wins.
  - RegressionChecker: validates a new Golden candidate against prior benchmarks.
"""
from __future__ import annotations

import copy
import math
import os
import random
from datetime import datetime, timezone
from typing import Optional

import yaml
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.config import settings
from backend.database import AsyncSessionLocal
from backend.logging_config import get_logger
from backend.models.recipe import RecipeRecord
from backend.services import recipe_library

logger = get_logger(__name__)

# Promotion thresholds
GOLDEN_WIN_THRESHOLD = 3       # consecutive evaluation wins required
PROMOTION_IMPROVEMENT = 0.01   # child must beat parent by at least 1%
GENE_POOL_SIZE = 10            # top-N candidates per domain
MUTATION_STRENGTH = 0.15       # default ±15% numeric perturbation

# Hard floor / ceiling guards for common hyperparameters
_BOUNDS: dict[str, tuple] = {
    "learning_rate": (1e-6, 1.0),
    "gamma": (0.8, 0.9999),
    "batch_size": (8, 4096),
    "clip_range": (0.05, 0.5),
    "entropy_coef": (0.0, 0.5),
    "total_timesteps": (10_000, 50_000_000),
    "n_steps": (32, 8192),
    "n_epochs": (1, 100),
}


# ── MutationOperator ──────────────────────────────────────────────────────────

class MutationOperator:
    """Produces a child recipe by perturbing numeric hyperparameters."""

    def __init__(self, strength: float = MUTATION_STRENGTH, seed: Optional[int] = None) -> None:
        self._strength = strength
        self._rng = random.Random(seed)

    def mutate(self, parent: RecipeRecord) -> dict:
        """
        Return a new full_content dict with perturbed hyperparameters.
        Non-numeric values are left unchanged.
        """
        child_content = copy.deepcopy(parent.full_content)
        child_hp: dict = copy.deepcopy(parent.hyperparameters)

        for key, val in child_hp.items():
            if not isinstance(val, (int, float)):
                continue
            lo, hi = _BOUNDS.get(key, (None, None))
            delta = val * self._strength * (2 * self._rng.random() - 1)
            new_val = val + delta
            if lo is not None:
                new_val = max(new_val, lo)
            if hi is not None:
                new_val = min(new_val, hi)
            # Preserve int type for integer params
            if isinstance(val, int):
                new_val = max(int(round(new_val)), int(lo) if lo else 1)
            child_hp[key] = new_val

        child_content["hyperparameters"] = child_hp
        return child_content


# ── SelectionPolicy ───────────────────────────────────────────────────────────

class SelectionPolicy:
    """Decides whether a child recipe should replace its parent."""

    def __init__(self, min_improvement: float = PROMOTION_IMPROVEMENT) -> None:
        self._threshold = min_improvement

    def should_promote(self, child_score: Optional[float], parent_score: Optional[float]) -> bool:
        if child_score is None:
            return False
        if parent_score is None:
            return True
        return child_score > parent_score * (1 + self._threshold)


# ── GenePool ──────────────────────────────────────────────────────────────────

class GenePool:
    """Aggregates top-performing recipes per domain as evolution candidates."""

    async def get_candidates(self, domain: str, n: int = GENE_POOL_SIZE) -> list[RecipeRecord]:
        """Return top-N non-golden recipes for a domain, ranked by score (desc)."""
        async with AsyncSessionLocal() as session:
            result = await session.execute(
                select(RecipeRecord)
                .where(RecipeRecord.domain == domain)
                .where(RecipeRecord.score.isnot(None))
                .order_by(RecipeRecord.score.desc())
                .limit(n)
            )
            return list(result.scalars().all())

    async def get_best(self, domain: str) -> Optional[RecipeRecord]:
        candidates = await self.get_candidates(domain, n=1)
        return candidates[0] if candidates else None


# ── GoldenPromoter ────────────────────────────────────────────────────────────

class GoldenPromoter:
    """Awards Golden status after GOLDEN_WIN_THRESHOLD consecutive wins."""

    async def record_win(self, recipe_id: str) -> bool:
        """
        Increment consecutive_wins. Returns True if Golden was awarded this call.
        """
        async with AsyncSessionLocal() as session:
            recipe = await session.get(RecipeRecord, recipe_id)
            if not recipe:
                return False
            recipe.consecutive_wins = (recipe.consecutive_wins or 0) + 1
            if not recipe.is_golden and recipe.consecutive_wins >= GOLDEN_WIN_THRESHOLD:
                recipe.is_golden = True
                recipe.updated_at = datetime.now(timezone.utc)
                await session.commit()
                logger.info("GoldenPromoter: recipe '%s' promoted to GOLDEN", recipe.name)
                # Re-index in recipe library so is_golden flag is current
                await session.refresh(recipe)
                try:
                    recipe_library.index_recipe(recipe)
                except Exception:
                    pass
                return True
            await session.commit()
            return False

    async def reset_wins(self, recipe_id: str) -> None:
        """Reset consecutive_wins on a failed evaluation (regression)."""
        async with AsyncSessionLocal() as session:
            recipe = await session.get(RecipeRecord, recipe_id)
            if recipe:
                recipe.consecutive_wins = 0
                await session.commit()


# ── RegressionChecker ─────────────────────────────────────────────────────────

class RegressionChecker:
    """
    Validates that a candidate Golden recipe doesn't regress on benchmarks
    solved by prior Golden recipes in the same domain.

    The check is intentionally lightweight: it compares the candidate's
    score against the best score of any existing Golden recipe in the domain.
    Phase 6 replaces this with full environment rollouts.
    """

    async def passes(self, candidate: RecipeRecord) -> bool:
        """Return True if the candidate is safe to promote."""
        if not candidate.score:
            return False
        async with AsyncSessionLocal() as session:
            result = await session.execute(
                select(RecipeRecord)
                .where(RecipeRecord.domain == candidate.domain)
                .where(RecipeRecord.is_golden == True)  # noqa: E712
                .order_by(RecipeRecord.score.desc())
                .limit(1)
            )
            best_golden = result.scalars().first()

        if not best_golden or best_golden.score is None:
            return True  # no prior Golden to regress against

        # Candidate must match or beat the existing Golden
        passes = candidate.score >= best_golden.score * (1 - PROMOTION_IMPROVEMENT)
        if not passes:
            logger.warning(
                "RegressionChecker: candidate '%s' (%.4f) regresses vs Golden '%s' (%.4f)",
                candidate.name, candidate.score, best_golden.name, best_golden.score,
            )
        return passes


# ── Orchestration helper ──────────────────────────────────────────────────────

async def evolve_recipe(parent_id: str) -> Optional[RecipeRecord]:
    """
    Create a mutated child recipe from a parent.

    1. Load parent from DB.
    2. Mutate hyperparameters.
    3. Persist child with generation+1 and parent_recipe_id set.
    4. Index child in recipe library.
    5. Return child RecipeRecord (no score yet — evaluation happens externally).
    """
    async with AsyncSessionLocal() as session:
        parent = await session.get(RecipeRecord, parent_id)
        if not parent:
            logger.error("evolve_recipe: parent %s not found", parent_id)
            return None

        mutator = MutationOperator()
        child_content = mutator.mutate(parent)

        child_version = f"{(int(parent.version.split('.')[0]) if parent.version else 1) + 1}.0.0"
        child_name = f"{parent.name}_evo_{child_version.replace('.', '_')}"
        child_content["name"] = child_name
        child_content["version"] = child_version
        child_content["provenance"] = {
            "parent_recipe": parent.name,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }

        child = RecipeRecord(
            name=child_name,
            version=child_version,
            domain=parent.domain,
            task_type=parent.task_type,
            description=f"Evolved from {parent.name} (gen {parent.generation + 1})",
            hyperparameters=child_content.get("hyperparameters", {}),
            curriculum=parent.curriculum,
            reward_shaping=parent.reward_shaping,
            full_content=child_content,
            parent_recipe_id=parent.id,
            target_metric=parent.target_metric,
            generation=parent.generation + 1,
        )
        session.add(child)
        await session.commit()
        await session.refresh(child)

    # Write YAML to disk
    recipes_dir = settings.recipes_path
    os.makedirs(recipes_dir, exist_ok=True)
    yaml_path = os.path.join(recipes_dir, f"{child_name}.yaml")
    with open(yaml_path, "w") as f:
        yaml.dump(child_content, f, default_flow_style=False, sort_keys=False)

    # Index in recipe library
    try:
        recipe_library.index_recipe(child)
    except Exception as exc:
        logger.warning("evolve_recipe: library indexing failed: %s", exc)

    logger.info("evolve_recipe: created child '%s' from parent '%s'", child_name, parent.name)
    return child
