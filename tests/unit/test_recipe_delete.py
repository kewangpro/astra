"""Unit tests for deleting auto-crystallized recipes."""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException

from backend.routers.recipes import delete_recipe, is_auto_crystallized


def _record(**kwargs):
    rec = MagicMock()
    rec.id = kwargs.get("id", "rid")
    rec.name = kwargs.get("name", "tetris_rl_v1")
    rec.generation = kwargs.get("generation", 1)
    rec.description = kwargs.get("description", "Auto-crystallized recipe from mission abc12345.")
    return rec


def _db(record=None):
    db = AsyncMock()
    db.get = AsyncMock(return_value=record)
    db.delete = AsyncMock()
    db.commit = AsyncMock()
    return db


def test_is_auto_crystallized_generation():
    assert is_auto_crystallized(_record(generation=1)) is True
    assert is_auto_crystallized(_record(generation=2, description="evolved")) is True
    assert is_auto_crystallized(_record(generation=0, description="Hand-crafted Snake PPO")) is False


def test_is_auto_crystallized_description_fallback():
    assert is_auto_crystallized(_record(generation=0, description="Auto-crystallized recipe from mission x.")) is True


@pytest.mark.asyncio
async def test_delete_recipe_removes_yaml_and_db(tmp_path, monkeypatch):
    from backend.routers import recipes as recipes_mod

    monkeypatch.setattr(recipes_mod.settings, "recipes_path", str(tmp_path))
    yaml_file = tmp_path / "tetris_rl_v1.yaml"
    yaml_file.write_text("name: tetris_rl_v1\n")
    rec = _record()
    db = _db(rec)

    with patch.object(recipes_mod.recipe_library, "remove_recipe") as remove:
        await delete_recipe("rid", db=db)
        db.delete.assert_awaited_once_with(rec)
        db.commit.assert_awaited()
        remove.assert_called_once_with("rid")
    assert not yaml_file.exists()


@pytest.mark.asyncio
async def test_delete_recipe_rejects_hand_crafted():
    rec = _record(generation=0, description="Canonical MinAtar Breakout DQN")
    db = _db(rec)
    with pytest.raises(HTTPException) as ei:
        await delete_recipe("rid", db=db)
    assert ei.value.status_code == 403
    db.delete.assert_not_called()


@pytest.mark.asyncio
async def test_delete_recipe_404():
    db = _db(None)
    with pytest.raises(HTTPException) as ei:
        await delete_recipe("missing", db=db)
    assert ei.value.status_code == 404
