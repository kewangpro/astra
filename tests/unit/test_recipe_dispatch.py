"""Unit tests for Recipe dispatch endpoint (backend/routers/recipes.py)."""
from __future__ import annotations

import os
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi import HTTPException

from backend.routers.recipes import dispatch_recipe


def _make_db(record=None):
    db = AsyncMock()
    db.get = AsyncMock(return_value=record)
    scalars = MagicMock()
    scalars.first = MagicMock(return_value=record)
    result = MagicMock()
    result.scalars = MagicMock(return_value=scalars)
    db.execute = AsyncMock(return_value=result)
    db.add = MagicMock()
    db.commit = AsyncMock()
    db.refresh = AsyncMock()
    return db


@pytest.mark.asyncio
async def test_dispatch_recipe_from_db():
    mock_record = MagicMock()
    mock_record.name = "tetris_actor_critic_v1"
    mock_record.domain = "Tetris-v0"
    mock_record.task_type = "rl"
    mock_record.target_metric = {"lines_cleared": 100.0}
    mock_record.description = "Tetris Actor-Critic baseline"

    db = _make_db(record=mock_record)

    mock_loop = MagicMock()
    mock_loop.run = AsyncMock()

    with patch("backend.routers.agent._build_loop", return_value=mock_loop), \
         patch("backend.routers.agent._running_tasks", {}):
        resp = await dispatch_recipe("tetris_actor_critic_v1", db=db)
        assert resp.status == "dispatched"
        assert resp.recipe == "tetris_actor_critic_v1"
        assert resp.task_type == "rl"
        assert resp.goal == "Tetris Actor-Critic baseline"
        db.add.assert_called_once()
        db.commit.assert_called_once()


@pytest.mark.asyncio
async def test_dispatch_recipe_from_disk(tmp_path):
    # Recipe not in DB, should read from recipes/ directory
    db = _make_db(record=None)
    mock_loop = MagicMock()
    mock_loop.run = AsyncMock()

    with patch("backend.routers.agent._build_loop", return_value=mock_loop), \
         patch("backend.routers.agent._running_tasks", {}):
        # Dispatch canonical snake_dqn_v1 from recipes/
        resp = await dispatch_recipe("snake_dqn_v1", db=db)
        assert resp.status == "dispatched"
        assert resp.recipe == "snake_dqn_v1"
        assert resp.task_type == "rl"
        db.add.assert_called_once()
        db.commit.assert_called_once()


@pytest.mark.asyncio
async def test_dispatch_sft_recipe_from_disk(tmp_path):
    db = _make_db(record=None)
    mock_loop = MagicMock()
    mock_loop.run = AsyncMock()

    with patch("backend.routers.agent._build_loop", return_value=mock_loop), \
         patch("backend.routers.agent._running_tasks", {}):
        resp = await dispatch_recipe("sft_llama_lora_v1", db=db)
        assert resp.status == "dispatched"
        assert resp.recipe == "sft_llama_lora_v1"
        assert resp.task_type.lower() == "sft"
        db.add.assert_called_once()
        db.commit.assert_called_once()


@pytest.mark.asyncio
async def test_dispatch_non_existent_recipe():
    db = _make_db(record=None)
    with pytest.raises(HTTPException) as exc:
        await dispatch_recipe("definitely_non_existent_recipe_xyz_999", db=db)
    assert exc.value.status_code == 404
    assert "not found" in exc.value.detail
