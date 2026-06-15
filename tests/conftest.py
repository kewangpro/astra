from __future__ import annotations

import os
import uuid
from unittest.mock import patch, MagicMock, AsyncMock

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from sqlalchemy.pool import StaticPool

import backend.models  # noqa: F401 — registers all ORM models with Base.metadata
from backend.database import Base
from backend.models.mission import Mission, MissionStatus


@pytest_asyncio.fixture
async def db_engine():
    """Fresh in-memory SQLite engine per test."""
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
    """Per-test async session against the in-memory engine."""
    maker = async_sessionmaker(db_engine, expire_on_commit=False, class_=AsyncSession)
    async with maker() as session:
        yield session


@pytest.fixture
def patch_db(db_engine, monkeypatch):
    """Replace AsyncSessionLocal in all modules with a test session maker.
    Also stubs out filesystem-writing services so tests don't create real directories."""
    maker = async_sessionmaker(db_engine, expire_on_commit=False, class_=AsyncSession)
    monkeypatch.setattr("backend.database.AsyncSessionLocal", maker)
    monkeypatch.setattr("backend.loop.state_machine.AsyncSessionLocal", maker)
    monkeypatch.setattr("backend.services.evolution.AsyncSessionLocal", maker)

    # Stub file-writing services to avoid creating real data/ directories in tests
    monkeypatch.setattr("backend.services.session_summary.write_session_summary", lambda *a, **kw: None)
    monkeypatch.setattr("backend.services.mission_state.update", lambda *a, **kw: {})
    monkeypatch.setattr("backend.services.preflight.PreflightChecker.run",
                        lambda self, *a, **kw: type("R", (), {"passed": True, "summary": lambda s: "4/4 checks passed"})())

    return maker


@pytest_asyncio.fixture
async def test_mission(db_session):
    """A PENDING mission seeded in the test DB."""
    mission = Mission(
        id=str(uuid.uuid4()),
        goal="Train a Snake RL agent",
        task_type="rl",
        target_metric={"mean_reward": 100.0},
        autonomy_mode="full_autonomy",
        status=MissionStatus.PENDING.value,
        current_iteration=0,
    )
    db_session.add(mission)
    await db_session.commit()
    return mission
