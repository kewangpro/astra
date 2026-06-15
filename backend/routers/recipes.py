"""
Recipe endpoints — Phase 5.

Serves both disk-based YAML recipes (hand-crafted) and DB-backed records
(crystallized + evolved). DB records take priority when names collide.

Phase 5 endpoints:
  POST /recipes/crystallize/{mission_id}  — distil a completed mission
  GET  /recipes/search                    — semantic search over recipe library
  POST /recipes/{recipe_id}/evolve        — spawn a mutated child recipe
  GET  /recipes/{recipe_id}/lineage       — parent chain for an evolved recipe
  GET  /recipes/db                        — list only DB-backed records
"""
from __future__ import annotations

import os
from typing import Optional, List

import yaml
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.config import settings
from backend.database import get_db
from backend.models.recipe import RecipeRecord
from backend.schemas.recipe import (
    RecipeRead,
    RecipeRecordRead,
    CrystallizeResponse,
    EvolveResponse,
    RecipeSearchHit,
)
from backend.services import crystallizer, recipe_library
from backend.services.evolution import evolve_recipe

router = APIRouter(prefix="/recipes", tags=["recipes"])


# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_disk_recipe(filename: str) -> RecipeRead:
    path = os.path.join(settings.recipes_path, filename)
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    return RecipeRead(
        name=os.path.splitext(filename)[0],
        filename=filename,
        domain=data.get("domain"),
        version=data.get("version"),
        description=data.get("description"),
        created_at=data.get("created_at"),
        content=data,
    )


# ── Disk + DB list ────────────────────────────────────────────────────────────

@router.get("", response_model=List[RecipeRead])
async def list_recipes(db: AsyncSession = Depends(get_db)):
    """List all recipes: disk-based (hand-crafted) merged with DB records."""
    disk_recipes: dict[str, RecipeRead] = {}
    if os.path.isdir(settings.recipes_path):
        for fname in sorted(os.listdir(settings.recipes_path)):
            if fname.endswith((".yaml", ".yml")):
                try:
                    r = _load_disk_recipe(fname)
                    disk_recipes[r.name] = r
                except Exception:
                    pass

    # DB records take priority on name collision
    result = await db.execute(select(RecipeRecord).order_by(RecipeRecord.created_at.desc()))
    for record in result.scalars().all():
        disk_recipes[record.name] = RecipeRead(
            name=record.name,
            filename=f"{record.name}.yaml",
            domain=record.domain,
            version=record.version,
            description=record.description,
            created_at=record.created_at.strftime("%Y-%m-%d"),
            content=record.full_content,
        )

    return list(disk_recipes.values())


@router.get("/db", response_model=List[RecipeRecordRead])
async def list_db_recipes(
    domain: Optional[str] = None,
    golden_only: bool = False,
    db: AsyncSession = Depends(get_db),
):
    """List DB-backed recipe records with optional filters."""
    q = select(RecipeRecord).order_by(RecipeRecord.score.desc().nullslast())
    if domain:
        q = q.where(RecipeRecord.domain == domain)
    if golden_only:
        q = q.where(RecipeRecord.is_golden == True)  # noqa: E712
    result = await db.execute(q)
    return result.scalars().all()


# ── Semantic search ───────────────────────────────────────────────────────────

@router.get("/search", response_model=List[RecipeSearchHit])
async def search_recipes(
    q: str = Query(..., description="Natural-language search query"),
    domain: Optional[str] = Query(None),
    n: int = Query(5, ge=1, le=20),
):
    """Semantic search over the recipe library."""
    hits = recipe_library.search_recipes(q, domain=domain, n_results=n)
    return [RecipeSearchHit(**h) for h in hits]


# ── Individual recipe ─────────────────────────────────────────────────────────

@router.get("/{recipe_name}", response_model=RecipeRead)
async def get_recipe(recipe_name: str, db: AsyncSession = Depends(get_db)):
    # Fixed-path sub-routes must be declared before this wildcard.
    # Check DB first
    result = await db.execute(
        select(RecipeRecord).where(RecipeRecord.name == recipe_name)
    )
    record = result.scalars().first()
    if record:
        return RecipeRead(
            name=record.name,
            filename=f"{record.name}.yaml",
            domain=record.domain,
            version=record.version,
            description=record.description,
            created_at=record.created_at.strftime("%Y-%m-%d"),
            content=record.full_content,
        )

    # Fallback to disk
    for ext in (".yaml", ".yml"):
        fpath = os.path.join(settings.recipes_path, recipe_name + ext)
        if os.path.isfile(fpath):
            return _load_disk_recipe(recipe_name + ext)

    raise HTTPException(status_code=404, detail=f"Recipe '{recipe_name}' not found")


# ── Crystallize ───────────────────────────────────────────────────────────────

@router.post("/crystallize/{mission_id}", response_model=CrystallizeResponse, status_code=201)
async def crystallize_mission(mission_id: str):
    """
    Distil a completed mission into a reusable recipe.
    Persists to DB, writes YAML to recipes/, and indexes in the semantic library.
    """
    record = await crystallizer.crystallize(mission_id)
    if not record:
        raise HTTPException(
            status_code=404,
            detail=f"Mission '{mission_id}' not found or has no plan",
        )
    yaml_path = os.path.join(settings.recipes_path, f"{record.name}.yaml")
    return CrystallizeResponse(
        recipe=RecipeRecordRead.model_validate(record),
        yaml_path=yaml_path,
    )


# ── Evolve ────────────────────────────────────────────────────────────────────

@router.post("/{recipe_id}/evolve", response_model=EvolveResponse, status_code=201)
async def evolve_recipe_endpoint(recipe_id: str, db: AsyncSession = Depends(get_db)):
    """
    Create a mutated child recipe from an existing DB recipe.
    The child is persisted but has no score yet; run it as a mission to evaluate it.
    """
    parent = await db.get(RecipeRecord, recipe_id)
    if not parent:
        raise HTTPException(status_code=404, detail=f"Recipe '{recipe_id}' not found")

    child = await evolve_recipe(recipe_id)
    if not child:
        raise HTTPException(status_code=500, detail="Evolution failed")

    return EvolveResponse(child=RecipeRecordRead.model_validate(child), parent_id=recipe_id)


# ── Lineage ───────────────────────────────────────────────────────────────────

@router.get("/{recipe_id}/lineage", response_model=List[RecipeRecordRead])
async def get_lineage(recipe_id: str, db: AsyncSession = Depends(get_db)):
    """Return the ancestor chain for an evolved recipe, oldest first."""
    chain: list[RecipeRecord] = []
    current_id: Optional[str] = recipe_id
    seen: set[str] = set()

    while current_id and current_id not in seen:
        seen.add(current_id)
        record = await db.get(RecipeRecord, current_id)
        if not record:
            break
        chain.append(record)
        current_id = record.parent_recipe_id

    chain.reverse()
    return chain
