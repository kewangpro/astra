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
    return mgr


@pytest.mark.asyncio
async def test_no_interrupted_missions_returns_zero(session_maker):
    with patch("backend.services.state_recovery.AsyncSessionLocal", session_maker), \
         patch("backend.sandbox.manager.sandbox_manager", _mock_sandbox_manager()):
        from backend.services.state_recovery import recover_interrupted_missions
        count = await recover_interrupted_missions()
    assert count == 0


@pytest.mark.asyncio
async def test_running_mission_reset_when_sandbox_gone(db_session, session_maker):
    mission = await _seed_mission(db_session, MissionStatus.RUNNING.value, pid=None)

    mock_mgr = _mock_sandbox_manager(recover_outcome="reset")
    with patch("backend.services.state_recovery.AsyncSessionLocal", session_maker), \
         patch("backend.sandbox.manager.sandbox_manager", mock_mgr):
        from backend.services.state_recovery import recover_interrupted_missions
        count = await recover_interrupted_missions()

    assert count == 1
    await db_session.refresh(mission)
    assert mission.status == MissionStatus.PENDING.value


@pytest.mark.asyncio
async def test_paused_mission_recovered(db_session, session_maker):
    await _seed_mission(db_session, MissionStatus.PAUSED.value)

    mock_mgr = _mock_sandbox_manager(recover_outcome="reset")
    with patch("backend.services.state_recovery.AsyncSessionLocal", session_maker), \
         patch("backend.sandbox.manager.sandbox_manager", mock_mgr):
        from backend.services.state_recovery import recover_interrupted_missions
        count = await recover_interrupted_missions()

    assert count == 1


@pytest.mark.asyncio
async def test_planning_mission_recovered(db_session, session_maker):
    await _seed_mission(db_session, MissionStatus.PLANNING.value)

    mock_mgr = _mock_sandbox_manager(recover_outcome="reset")
    with patch("backend.services.state_recovery.AsyncSessionLocal", session_maker), \
         patch("backend.sandbox.manager.sandbox_manager", mock_mgr):
        from backend.services.state_recovery import recover_interrupted_missions
        count = await recover_interrupted_missions()

    assert count == 1


@pytest.mark.asyncio
async def test_completed_mission_not_touched(db_session, session_maker):
    await _seed_mission(db_session, MissionStatus.COMPLETED.value)

    mock_mgr = _mock_sandbox_manager(recover_outcome="reset")
    with patch("backend.services.state_recovery.AsyncSessionLocal", session_maker), \
         patch("backend.sandbox.manager.sandbox_manager", mock_mgr):
        from backend.services.state_recovery import recover_interrupted_missions
        count = await recover_interrupted_missions()

    assert count == 0


@pytest.mark.asyncio
async def test_reattached_mission_stays_running(db_session, session_maker):
    mission = await _seed_mission(db_session, MissionStatus.RUNNING.value, pid=12345)

    mock_mgr = _mock_sandbox_manager(recover_outcome="reattached")
    with patch("backend.services.state_recovery.AsyncSessionLocal", session_maker), \
         patch("backend.sandbox.manager.sandbox_manager", mock_mgr):
        from backend.services.state_recovery import recover_interrupted_missions
        count = await recover_interrupted_missions()

    assert count == 1
    await db_session.refresh(mission)
    # Reattached → status should remain RUNNING (not reset to PENDING)
    assert mission.status == MissionStatus.RUNNING.value


@pytest.mark.asyncio
async def test_multiple_missions_all_counted(db_session, session_maker):
    for _ in range(3):
        await _seed_mission(db_session, MissionStatus.RUNNING.value)

    mock_mgr = _mock_sandbox_manager(recover_outcome="reset")
    with patch("backend.services.state_recovery.AsyncSessionLocal", session_maker), \
         patch("backend.sandbox.manager.sandbox_manager", mock_mgr):
        from backend.services.state_recovery import recover_interrupted_missions
        count = await recover_interrupted_missions()

    assert count == 3


@pytest.mark.asyncio
async def test_mixed_outcomes_handled(db_session, session_maker):
    m1 = await _seed_mission(db_session, MissionStatus.RUNNING.value, pid=111)
    m2 = await _seed_mission(db_session, MissionStatus.RUNNING.value, pid=None)

    mock_mgr = MagicMock()
    mock_mgr.recover.side_effect = lambda mission_id, pid, cid: (
        "reattached" if pid == 111 else "reset"
    )

    with patch("backend.services.state_recovery.AsyncSessionLocal", session_maker), \
         patch("backend.sandbox.manager.sandbox_manager", mock_mgr):
        from backend.services.state_recovery import recover_interrupted_missions
        count = await recover_interrupted_missions()

    assert count == 2
    await db_session.refresh(m1)
    await db_session.refresh(m2)
    assert m1.status == MissionStatus.RUNNING.value    # reattached
    assert m2.status == MissionStatus.PENDING.value    # reset
