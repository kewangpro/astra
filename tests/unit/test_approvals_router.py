"""Unit tests for GET /approvals — mission_id filter."""
from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock


def _make_db(gates: list) -> AsyncMock:
    """Return an async-compatible mock DB session whose execute() captures
    the constructed query so tests can assert on the actual SQL filter."""
    db = AsyncMock()
    scalars = MagicMock()
    scalars.all = MagicMock(return_value=gates)
    result = MagicMock()
    result.scalars = MagicMock(return_value=scalars)
    db.execute = AsyncMock(return_value=result)
    return db


@pytest.mark.asyncio
async def test_no_mission_id_returns_all_pending():
    from backend.routers.approvals import list_approvals

    db = _make_db(gates=["gate-a", "gate-b"])
    result = await list_approvals(pending_only=True, mission_id=None, db=db)
    assert result == ["gate-a", "gate-b"]
    db.execute.assert_called_once()


@pytest.mark.asyncio
async def test_mission_id_filter_applies_where_clause():
    """The generated query must actually include a mission_id predicate —
    not just accept the parameter and ignore it."""
    from backend.routers.approvals import list_approvals
    from backend.models.approval import ApprovalGate

    db = _make_db(gates=["gate-a"])
    await list_approvals(pending_only=False, mission_id="mission-123", db=db)

    executed_query = db.execute.call_args.args[0]
    compiled = str(executed_query.compile(compile_kwargs={"literal_binds": True}))
    assert "mission_id" in compiled
    assert "mission-123" in compiled


@pytest.mark.asyncio
async def test_pending_only_false_with_mission_id_returns_full_history():
    from backend.routers.approvals import list_approvals

    db = _make_db(gates=["gate-a", "gate-b", "gate-c"])
    result = await list_approvals(pending_only=False, mission_id="mission-123", db=db)
    assert result == ["gate-a", "gate-b", "gate-c"]
