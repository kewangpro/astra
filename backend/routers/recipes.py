"""
Recipe endpoints — serve YAML recipes from disk (recipes/ dir).
Full DB-backed CRUD and crystallization logic land in Phase 5.
"""
import os
import yaml
from fastapi import APIRouter, HTTPException
from backend.config import settings
from backend.schemas.recipe import RecipeRead

router = APIRouter(prefix="/recipes", tags=["recipes"])


def _load_recipe(filename: str) -> RecipeRead:
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


@router.get("", response_model=list[RecipeRead])
async def list_recipes():
    recipes_dir = settings.recipes_path
    if not os.path.isdir(recipes_dir):
        return []
    recipes = []
    for fname in sorted(os.listdir(recipes_dir)):
        if fname.endswith((".yaml", ".yml")):
            try:
                recipes.append(_load_recipe(fname))
            except Exception:
                pass
    return recipes


@router.get("/{recipe_name}", response_model=RecipeRead)
async def get_recipe(recipe_name: str):
    for ext in (".yaml", ".yml"):
        fname = recipe_name + ext
        path = os.path.join(settings.recipes_path, fname)
        if os.path.isfile(path):
            return _load_recipe(fname)
    raise HTTPException(status_code=404, detail=f"Recipe '{recipe_name}' not found")
