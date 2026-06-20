"""Unit tests for DELETE /missions/{id} — task cancellation and gate rejection."""
from __future__ import annotations

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch


# ── helpers ───────────────────────────────────────────────────────────────────

def _make_gate(status: str = "pending") -> MagicMock:
    gate = MagicMock()
    gate.status = status
    gate.reviewer_note = None
    return gate


def _make_mission() -> MagicMock:
    mission = MagicMock()
    mission.id = "test-mission-id"
    return mission


def _make_db(mission=None, gates=None):
    """Return an async-compatible mock DB session."""
    db = AsyncMock()
    db.get = AsyncMock(return_value=mission)
    scalars = MagicMock()
    scalars.all = MagicMock(return_value=gates or [])
    result = MagicMock()
    result.scalars = MagicMock(return_value=scalars)
    db.execute = AsyncMock(return_value=result)
    db.delete = AsyncMock()
    db.commit = AsyncMock()
    return db


# ── tests ─────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_delete_cancels_running_task():
    """Running asyncio task is cancelled when the mission is deleted."""
    from backend.routers.missions import delete_mission

    task = MagicMock(spec=asyncio.Task)
    task.done.return_value = False

    mission = _make_mission()
    db = _make_db(mission=mission)

    with patch("backend.routers.agent._running_tasks", {"test-mission-id": task}):
        await delete_mission("test-mission-id", db)

    task.cancel.assert_called_once()


@pytest.mark.asyncio
async def test_delete_skips_cancel_for_finished_task():
    """Completed tasks are not cancelled (no-op)."""
    from backend.routers.missions import delete_mission

    task = MagicMock(spec=asyncio.Task)
    task.done.return_value = True

    mission = _make_mission()
    db = _make_db(mission=mission)

    with patch("backend.routers.agent._running_tasks", {"test-mission-id": task}):
        await delete_mission("test-mission-id", db)

    task.cancel.assert_not_called()


@pytest.mark.asyncio
async def test_delete_rejects_pending_gates():
    """All pending approval gates are rejected before deletion."""
    from backend.routers.missions import delete_mission

    gate1 = _make_gate("pending")
    gate2 = _make_gate("pending")
    mission = _make_mission()
    db = _make_db(mission=mission, gates=[gate1, gate2])

    with patch("backend.routers.agent._running_tasks", {}):
        await delete_mission("test-mission-id", db)

    assert gate1.status == "rejected"
    assert gate1.reviewer_note == "mission deleted"
    assert gate2.status == "rejected"
    assert gate2.reviewer_note == "mission deleted"


@pytest.mark.asyncio
async def test_delete_no_pending_gates():
    """Deletion succeeds when there are no pending gates."""
    from backend.routers.missions import delete_mission

    mission = _make_mission()
    db = _make_db(mission=mission, gates=[])

    with patch("backend.routers.agent._running_tasks", {}):
        await delete_mission("test-mission-id", db)

    db.delete.assert_called_once_with(mission)
    db.commit.assert_called_once()


@pytest.mark.asyncio
async def test_delete_no_running_task():
    """Deletion succeeds when the mission has no running task."""
    from backend.routers.missions import delete_mission

    mission = _make_mission()
    db = _make_db(mission=mission)

    with patch("backend.routers.agent._running_tasks", {}):
        await delete_mission("test-mission-id", db)

    db.delete.assert_called_once_with(mission)
    db.commit.assert_called_once()


@pytest.mark.asyncio
async def test_delete_removes_task_from_registry():
    """Task is removed from _running_tasks dict after cancellation."""
    from backend.routers.missions import delete_mission

    task = MagicMock(spec=asyncio.Task)
    task.done.return_value = False

    mission = _make_mission()
    db = _make_db(mission=mission)

    registry = {"test-mission-id": task}
    with patch("backend.routers.agent._running_tasks", registry):
        await delete_mission("test-mission-id", db)

    assert "test-mission-id" not in registry


@pytest.mark.asyncio
async def test_delete_mission_not_found_raises_404():
    """Returns 404 when mission does not exist."""
    from fastapi import HTTPException
    from backend.routers.missions import delete_mission

    db = _make_db(mission=None)

    with pytest.raises(HTTPException) as exc_info:
        await delete_mission("nonexistent-id", db)

    assert exc_info.value.status_code == 404
