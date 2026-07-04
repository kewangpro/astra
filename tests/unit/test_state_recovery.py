"""Unit tests for state_recovery.recover_interrupted_missions."""
from __future__ import annotations

import uuid
from unittest.mock import AsyncMock, MagicMock, patch

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


async def _seed_mission(session, status: str, pid: int | None = None) -> Mission:
    m = Mission(
        id=str(uuid.uuid4()),
        goal="test goal",
        task_type="rl",
        status=status,
        subprocess_pid=pid,
    )
    session.add(m)
    await session.commit()
    return m


def _mock_sandbox_manager(recover_outcome: str = "reset") -> MagicMock:
    mgr = MagicMock()
    mgr.recover.return_value = recover_outcome
    mgr.terminate = MagicMock()
    return mgr


@pytest.mark.asyncio
async def test_no_interrupted_missions_returns_empty(session_maker):
    with patch("backend.services.state_recovery.AsyncSessionLocal", session_maker), \
         patch("backend.sandbox.manager.sandbox_manager", _mock_sandbox_manager()):
        from backend.services.state_recovery import recover_interrupted_missions
        ids = await recover_interrupted_missions()
    assert ids == []


@pytest.mark.asyncio
async def test_running_mission_reset_when_sandbox_gone(db_session, session_maker):
    mission = await _seed_mission(db_session, MissionStatus.RUNNING.value, pid=None)

    mock_mgr = _mock_sandbox_manager(recover_outcome="reset")
    with patch("backend.services.state_recovery.AsyncSessionLocal", session_maker), \
         patch("backend.sandbox.manager.sandbox_manager", mock_mgr):
        from backend.services.state_recovery import recover_interrupted_missions
        ids = await recover_interrupted_missions()

    assert len(ids) == 1
    await db_session.refresh(mission)
    assert mission.status == MissionStatus.PENDING.value


@pytest.mark.asyncio
async def test_paused_mission_recovered(db_session, session_maker):
    await _seed_mission(db_session, MissionStatus.PAUSED.value)

    mock_mgr = _mock_sandbox_manager(recover_outcome="reset")
    with patch("backend.services.state_recovery.AsyncSessionLocal", session_maker), \
         patch("backend.sandbox.manager.sandbox_manager", mock_mgr):
        from backend.services.state_recovery import recover_interrupted_missions
        ids = await recover_interrupted_missions()

    assert len(ids) == 1


@pytest.mark.asyncio
async def test_planning_mission_recovered(db_session, session_maker):
    await _seed_mission(db_session, MissionStatus.PLANNING.value)

    mock_mgr = _mock_sandbox_manager(recover_outcome="reset")
    with patch("backend.services.state_recovery.AsyncSessionLocal", session_maker), \
         patch("backend.sandbox.manager.sandbox_manager", mock_mgr):
        from backend.services.state_recovery import recover_interrupted_missions
        ids = await recover_interrupted_missions()

    assert len(ids) == 1


@pytest.mark.asyncio
async def test_completed_mission_not_touched(db_session, session_maker):
    await _seed_mission(db_session, MissionStatus.COMPLETED.value)

    mock_mgr = _mock_sandbox_manager(recover_outcome="reset")
    with patch("backend.services.state_recovery.AsyncSessionLocal", session_maker), \
         patch("backend.sandbox.manager.sandbox_manager", mock_mgr):
        from backend.services.state_recovery import recover_interrupted_missions
        ids = await recover_interrupted_missions()

    assert ids == []


@pytest.mark.asyncio
async def test_reattached_mission_terminated_and_reset_to_pending(db_session, session_maker):
    """When a sandbox is still alive, it must be terminated and the mission reset to PENDING
    so the loop can restart cleanly from the last saved checkpoint."""
    mission = await _seed_mission(db_session, MissionStatus.RUNNING.value, pid=12345)

    mock_mgr = _mock_sandbox_manager(recover_outcome="reattached")
    with patch("backend.services.state_recovery.AsyncSessionLocal", session_maker), \
         patch("backend.sandbox.manager.sandbox_manager", mock_mgr):
        from backend.services.state_recovery import recover_interrupted_missions
        ids = await recover_interrupted_missions()

    assert len(ids) == 1
    mock_mgr.terminate.assert_called_once_with(mission.id)
    await db_session.refresh(mission)
    assert mission.status == MissionStatus.PENDING.value


@pytest.mark.asyncio
async def test_remote_pid_passed_to_recover_and_reset_on_reattach(db_session, session_maker):
    """dpo/grpo missions (SSH-dispatched) must pass remote_pid to recover() the
    same way subprocess_pid is passed for local missions, and it must be reset
    to None afterward like the other sandbox-tracking fields."""
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
        await recover_interrupted_missions()

    mock_mgr.recover.assert_called_once_with(mission.id, None, None, remote_pid=14516)
    await db_session.refresh(mission)
    assert mission.remote_pid is None
    assert mission.status == MissionStatus.PENDING.value


@pytest.mark.asyncio
async def test_multiple_missions_all_returned(db_session, session_maker):
    for _ in range(3):
        await _seed_mission(db_session, MissionStatus.RUNNING.value)

    mock_mgr = _mock_sandbox_manager(recover_outcome="reset")
    with patch("backend.services.state_recovery.AsyncSessionLocal", session_maker), \
         patch("backend.sandbox.manager.sandbox_manager", mock_mgr):
        from backend.services.state_recovery import recover_interrupted_missions
        ids = await recover_interrupted_missions()

    assert len(ids) == 3


@pytest.mark.asyncio
async def test_mixed_outcomes_both_reset_to_pending(db_session, session_maker):
    """Both reattached and gone missions end up PENDING — only reattached ones get terminate()."""
    m1 = await _seed_mission(db_session, MissionStatus.RUNNING.value, pid=111)
    m2 = await _seed_mission(db_session, MissionStatus.RUNNING.value, pid=None)

    mock_mgr = MagicMock()
    mock_mgr.recover.side_effect = lambda mission_id, pid, cid, remote_pid=None: (
        "reattached" if pid == 111 else "reset"
    )
    mock_mgr.terminate = MagicMock()

    with patch("backend.services.state_recovery.AsyncSessionLocal", session_maker), \
         patch("backend.sandbox.manager.sandbox_manager", mock_mgr):
        from backend.services.state_recovery import recover_interrupted_missions
        ids = await recover_interrupted_missions()

    assert len(ids) == 2
    await db_session.refresh(m1)
    await db_session.refresh(m2)
    assert m1.status == MissionStatus.PENDING.value   # reattached → terminated → pending
    assert m2.status == MissionStatus.PENDING.value   # gone → pending
    mock_mgr.terminate.assert_called_once_with(m1.id)  # only reattached one terminated
