"""Unit tests for POST /agent/missions/{id}/run and /resume."""
from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi import HTTPException


def _make_mission(status: str = "pending") -> MagicMock:
    m = MagicMock()
    m.id = "test-mission-id"
    m.status = status
    m.error_log = None
    m.completed_at = None
    return m


def _make_db(mission=None):
    db = AsyncMock()
    db.get = AsyncMock(return_value=mission)
    db.commit = AsyncMock()
    return db


@pytest.mark.asyncio
async def test_run_rejects_stalled():
    """A converged stall is a finished search — resume is a 409, not a restart."""
    from backend.routers.agent import run_mission

    db = _make_db(mission=_make_mission("stalled"))

    with pytest.raises(HTTPException) as exc_info:
        await run_mission("test-mission-id", db)

    assert exc_info.value.status_code == 409
    assert "stalled" in exc_info.value.detail.lower()
    db.commit.assert_not_called()


@pytest.mark.asyncio
async def test_run_failed_resets_to_pending_and_starts_loop():
    from backend.routers.agent import run_mission

    mission = _make_mission("failed")
    mission.error_log = "unhandled_error: hung"
    db = _make_db(mission=mission)
    loop = MagicMock()
    task = MagicMock()
    task.done.return_value = False

    with patch("backend.routers.agent._running_tasks", {}), \
         patch("backend.routers.agent._running_loops", {}), \
         patch("backend.routers.agent._build_loop", return_value=loop), \
         patch("backend.routers.agent.asyncio.create_task", return_value=task):
        result = await run_mission("test-mission-id", db)

    assert result["status"] == "loop_started"
    assert mission.status == "pending"
    assert mission.error_log is None
    db.commit.assert_awaited()
