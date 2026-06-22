"""Unit tests for POST /agent/missions/{id}/cancel."""
from __future__ import annotations

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch


def _make_mission(status: str = "running") -> MagicMock:
    m = MagicMock()
    m.id = "test-mission-id"
    m.status = status
    return m


def _make_db(mission=None):
    db = AsyncMock()
    db.get = AsyncMock(return_value=mission)
    return db


@pytest.mark.asyncio
async def test_cancel_running_task():
    """Running asyncio task is cancelled when the mission is stopped."""
    from backend.routers.agent import cancel_mission

    task = MagicMock(spec=asyncio.Task)
    task.done.return_value = False

    db = _make_db(mission=_make_mission("running"))

    with patch("backend.routers.agent._running_tasks", {"test-mission-id": task}):
        result = await cancel_mission("test-mission-id", db)

    task.cancel.assert_called_once()
    assert result["status"] == "cancelling"


@pytest.mark.asyncio
async def test_cancel_skips_finished_task():
    """Completed tasks are not cancelled (no-op)."""
    from backend.routers.agent import cancel_mission

    task = MagicMock(spec=asyncio.Task)
    task.done.return_value = True

    db = _make_db(mission=_make_mission("running"))

    with patch("backend.routers.agent._running_tasks", {"test-mission-id": task}):
        await cancel_mission("test-mission-id", db)

    task.cancel.assert_not_called()


@pytest.mark.asyncio
async def test_cancel_no_running_task():
    """Cancel succeeds even if there is no tracked task (idempotent)."""
    from backend.routers.agent import cancel_mission

    db = _make_db(mission=_make_mission("running"))

    with patch("backend.routers.agent._running_tasks", {}):
        result = await cancel_mission("test-mission-id", db)

    assert result["status"] == "cancelling"


@pytest.mark.asyncio
async def test_cancel_mission_not_found_raises_404():
    """Returns 404 when mission does not exist."""
    from fastapi import HTTPException
    from backend.routers.agent import cancel_mission

    db = _make_db(mission=None)

    with pytest.raises(HTTPException) as exc_info:
        await cancel_mission("nonexistent-id", db)

    assert exc_info.value.status_code == 404


@pytest.mark.asyncio
async def test_cancel_non_running_mission_raises_409():
    """Returns 409 when mission is not in a cancellable state."""
    from fastapi import HTTPException
    from backend.routers.agent import cancel_mission

    db = _make_db(mission=_make_mission("pending"))

    with patch("backend.routers.agent._running_tasks", {}):
        with pytest.raises(HTTPException) as exc_info:
            await cancel_mission("test-mission-id", db)

    assert exc_info.value.status_code == 409


@pytest.mark.asyncio
async def test_cancel_planning_mission():
    """Planning missions are also cancellable."""
    from backend.routers.agent import cancel_mission

    task = MagicMock(spec=asyncio.Task)
    task.done.return_value = False

    db = _make_db(mission=_make_mission("planning"))

    with patch("backend.routers.agent._running_tasks", {"test-mission-id": task}):
        result = await cancel_mission("test-mission-id", db)

    task.cancel.assert_called_once()
    assert result["status"] == "cancelling"
