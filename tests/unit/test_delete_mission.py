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


@pytest.fixture
def _patch_mission_delete_side_effects():
    with patch("backend.routers.missions.purge_models_for_mission", new_callable=AsyncMock) as purge:
        with patch("backend.routers.missions._remove_mission_workdir") as rm:
            yield {"purge": purge, "rm": rm}


# ── tests ─────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_delete_cancels_running_task(_patch_mission_delete_side_effects):
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
async def test_delete_skips_cancel_for_finished_task(_patch_mission_delete_side_effects):
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
async def test_delete_rejects_pending_gates(_patch_mission_delete_side_effects):
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
async def test_delete_no_pending_gates(_patch_mission_delete_side_effects):
    """Deletion succeeds when there are no pending gates."""
    from backend.routers.missions import delete_mission

    mission = _make_mission()
    db = _make_db(mission=mission, gates=[])

    with patch("backend.routers.agent._running_tasks", {}):
        await delete_mission("test-mission-id", db)

    db.delete.assert_called_once_with(mission)
    db.commit.assert_called_once()


@pytest.mark.asyncio
async def test_delete_no_running_task(_patch_mission_delete_side_effects):
    """Deletion succeeds when the mission has no running task."""
    from backend.routers.missions import delete_mission

    mission = _make_mission()
    db = _make_db(mission=mission)

    with patch("backend.routers.agent._running_tasks", {}):
        await delete_mission("test-mission-id", db)

    db.delete.assert_called_once_with(mission)
    db.commit.assert_called_once()


@pytest.mark.asyncio
async def test_delete_removes_task_from_registry(_patch_mission_delete_side_effects):
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
async def test_delete_mission_not_found_raises_404(_patch_mission_delete_side_effects):
    """Returns 404 when mission does not exist."""
    from fastapi import HTTPException
    from backend.routers.missions import delete_mission

    db = _make_db(mission=None)

    with pytest.raises(HTTPException) as exc_info:
        await delete_mission("nonexistent-id", db)

    assert exc_info.value.status_code == 404
    _patch_mission_delete_side_effects["purge"].assert_not_awaited()


@pytest.mark.asyncio
async def test_delete_purges_registry_and_workdir(_patch_mission_delete_side_effects):
    from backend.routers.missions import delete_mission

    mission = _make_mission()
    db = _make_db(mission=mission)
    with patch("backend.routers.agent._running_tasks", {}):
        await delete_mission("test-mission-id", db)
    _patch_mission_delete_side_effects["purge"].assert_awaited_once_with(db, "test-mission-id")
    _patch_mission_delete_side_effects["rm"].assert_called_once_with("test-mission-id")


def test_remove_mission_workdir_deletes_only_under_data(tmp_path, monkeypatch):
    from backend.config import settings
    from backend.routers.missions import _remove_mission_workdir

    monkeypatch.setattr(settings, "data_path", str(tmp_path))
    mid = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
    target = tmp_path / "missions" / mid
    target.mkdir(parents=True)
    (target / "train.py").write_text("x")
    _remove_mission_workdir(mid)
    assert not target.exists()
    _remove_mission_workdir("../etc")
    assert (tmp_path / "missions").exists()
