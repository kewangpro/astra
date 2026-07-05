"""Unit tests for state_recovery.recover_interrupted_missions."""
from __future__ import annotations

import uuid
from unittest.mock import MagicMock, patch

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from sqlalchemy.pool import StaticPool

import backend.models  # noqa: F401
from backend.database import Base
from backend.models.mission import Mission, MissionStatus


@pytest_asyncio.fixture
async def db_engine():
    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield engine
    await engine.dispose()


@pytest_asyncio.fixture
async def db_session(db_engine):
    maker = async_sessionmaker(db_engine, expire_on_commit=False, class_=AsyncSession)
    async with maker() as session:
        yield session


@pytest_asyncio.fixture
async def session_maker(db_engine):
    return async_sessionmaker(db_engine, expire_on_commit=False, class_=AsyncSession)


async def _seed_mission(session, status: str, pid: int | None = None, task_type: str = "rl") -> Mission:
    m = Mission(
        id=str(uuid.uuid4()),
        goal="test goal",
        task_type=task_type,
        status=status,
        subprocess_pid=pid,
    )
    session.add(m)
    await session.commit()
    return m


def _mock_sandbox_manager(recover_outcome: str = "dead") -> MagicMock:
    mgr = MagicMock()
    mgr.recover.return_value = recover_outcome
    mgr.terminate = MagicMock()
    return mgr


@pytest.mark.asyncio
async def test_no_interrupted_missions_returns_empty(session_maker):
    with patch("backend.services.state_recovery.AsyncSessionLocal", session_maker), \
         patch("backend.sandbox.manager.sandbox_manager", _mock_sandbox_manager()):
        from backend.services.state_recovery import recover_interrupted_missions
        result = await recover_interrupted_missions()
    assert result == {"restart": [], "resume": []}


@pytest.mark.asyncio
async def test_running_mission_reset_when_sandbox_gone(db_session, session_maker):
    mission = await _seed_mission(db_session, MissionStatus.RUNNING.value, pid=None)

    mock_mgr = _mock_sandbox_manager(recover_outcome="dead")
    with patch("backend.services.state_recovery.AsyncSessionLocal", session_maker), \
         patch("backend.sandbox.manager.sandbox_manager", mock_mgr):
        from backend.services.state_recovery import recover_interrupted_missions
        result = await recover_interrupted_missions()

    assert result["restart"] == [mission.id]
    assert result["resume"] == []
    mock_mgr.terminate.assert_not_called()
    await db_session.refresh(mission)
    assert mission.status == MissionStatus.PENDING.value


@pytest.mark.asyncio
async def test_paused_mission_recovered(db_session, session_maker):
    await _seed_mission(db_session, MissionStatus.PAUSED.value)

    mock_mgr = _mock_sandbox_manager(recover_outcome="dead")
    with patch("backend.services.state_recovery.AsyncSessionLocal", session_maker), \
         patch("backend.sandbox.manager.sandbox_manager", mock_mgr):
        from backend.services.state_recovery import recover_interrupted_missions
        result = await recover_interrupted_missions()

    assert len(result["restart"]) == 1


@pytest.mark.asyncio
async def test_planning_mission_recovered(db_session, session_maker):
    await _seed_mission(db_session, MissionStatus.PLANNING.value)

    mock_mgr = _mock_sandbox_manager(recover_outcome="dead")
    with patch("backend.services.state_recovery.AsyncSessionLocal", session_maker), \
         patch("backend.sandbox.manager.sandbox_manager", mock_mgr):
        from backend.services.state_recovery import recover_interrupted_missions
        result = await recover_interrupted_missions()

    assert len(result["restart"]) == 1


@pytest.mark.asyncio
async def test_completed_mission_not_touched(db_session, session_maker):
    await _seed_mission(db_session, MissionStatus.COMPLETED.value)

    mock_mgr = _mock_sandbox_manager(recover_outcome="dead")
    with patch("backend.services.state_recovery.AsyncSessionLocal", session_maker), \
         patch("backend.sandbox.manager.sandbox_manager", mock_mgr):
        from backend.services.state_recovery import recover_interrupted_missions
        result = await recover_interrupted_missions()

    assert result == {"restart": [], "resume": []}


@pytest.mark.asyncio
async def test_reattached_mission_resumes_in_place(db_session, session_maker):
    """When a sandbox is still alive, it must NOT be terminated — the mission
    stays in its current status/pid state and is returned for reattach-resume,
    not reset to PENDING. Killing a still-running job just to restart it from
    checkpoint throws away real progress (the original motivation for this
    change: a live, hours-long remote training run getting silently orphaned
    and left running unmanaged after an unrelated bug marked its mission FAILED)."""
    mission = await _seed_mission(db_session, MissionStatus.RUNNING.value, pid=12345)

    mock_mgr = _mock_sandbox_manager(recover_outcome="reattached")
    with patch("backend.services.state_recovery.AsyncSessionLocal", session_maker), \
         patch("backend.sandbox.manager.sandbox_manager", mock_mgr):
        from backend.services.state_recovery import recover_interrupted_missions
        result = await recover_interrupted_missions()

    assert result["resume"] == [mission.id]
    assert result["restart"] == []
    mock_mgr.terminate.assert_not_called()
    await db_session.refresh(mission)
    assert mission.status == MissionStatus.RUNNING.value
    assert mission.subprocess_pid == 12345


@pytest.mark.asyncio
async def test_remote_pid_passed_to_recover_and_kept_on_reattach(db_session, session_maker):
    """dpo/grpo missions (SSH-dispatched) must pass remote_pid to recover() the
    same way subprocess_pid is passed for local missions, and — like any other
    reattached sandbox — remote_pid must be left untouched, not cleared."""
    mission = Mission(
        id=str(uuid.uuid4()),
        goal="dpo test goal",
        task_type="dpo",
        status=MissionStatus.RUNNING.value,
        remote_pid=14516,
    )
    db_session.add(mission)
    await db_session.commit()

    mock_mgr = _mock_sandbox_manager(recover_outcome="reattached")
    with patch("backend.services.state_recovery.AsyncSessionLocal", session_maker), \
         patch("backend.sandbox.manager.sandbox_manager", mock_mgr):
        from backend.services.state_recovery import recover_interrupted_missions
        result = await recover_interrupted_missions()

    mock_mgr.recover.assert_called_once_with(mission.id, None, None, remote_pid=14516)
    assert result["resume"] == [mission.id]
    await db_session.refresh(mission)
    assert mission.remote_pid == 14516
    assert mission.status == MissionStatus.RUNNING.value


@pytest.mark.asyncio
async def test_multiple_missions_all_returned(db_session, session_maker):
    for _ in range(3):
        await _seed_mission(db_session, MissionStatus.RUNNING.value)

    mock_mgr = _mock_sandbox_manager(recover_outcome="dead")
    with patch("backend.services.state_recovery.AsyncSessionLocal", session_maker), \
         patch("backend.sandbox.manager.sandbox_manager", mock_mgr):
        from backend.services.state_recovery import recover_interrupted_missions
        result = await recover_interrupted_missions()

    assert len(result["restart"]) == 3
    assert result["resume"] == []


@pytest.mark.asyncio
async def test_mixed_outcomes_split_between_resume_and_restart(db_session, session_maker):
    """Reattached (alive) missions go to resume and are left alone; gone
    sandboxes go to restart and are reset to PENDING. Neither path calls
    terminate() — a still-alive sandbox is deliberately never touched."""
    m1 = await _seed_mission(db_session, MissionStatus.RUNNING.value, pid=111)
    m2 = await _seed_mission(db_session, MissionStatus.RUNNING.value, pid=None)

    mock_mgr = MagicMock()
    mock_mgr.recover.side_effect = lambda mission_id, pid, cid, remote_pid=None: (
        "reattached" if pid == 111 else "dead"
    )
    mock_mgr.terminate = MagicMock()

    with patch("backend.services.state_recovery.AsyncSessionLocal", session_maker), \
         patch("backend.sandbox.manager.sandbox_manager", mock_mgr):
        from backend.services.state_recovery import recover_interrupted_missions
        result = await recover_interrupted_missions()

    assert result["resume"] == [m1.id]
    assert result["restart"] == [m2.id]
    mock_mgr.terminate.assert_not_called()
    await db_session.refresh(m1)
    await db_session.refresh(m2)
    assert m1.status == MissionStatus.RUNNING.value   # reattached → left running
    assert m2.status == MissionStatus.PENDING.value   # gone → reset to pending
