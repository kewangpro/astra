"""Unit tests for nodes router (backend/routers/nodes.py)."""
from __future__ import annotations

import asyncio
from unittest.mock import MagicMock, patch
import pytest

from backend.models.mission import Mission, MissionStatus
from backend.routers.nodes import list_nodes


@pytest.mark.asyncio
async def test_list_nodes_enriches_with_active_missions():
    fake_node = {
        "host": "local",
        "is_local": True,
        "alive": False,
        "real_available_gb": 4.5,
        "missions": [],
    }
    active_m = MagicMock(spec=Mission)
    active_m.id = "test-mission-123"
    active_m.status = MissionStatus.RUNNING.value
    active_m.host = "local"
    active_m.subprocess_pid = None
    active_m.remote_pid = None

    with patch("backend.routers.nodes.sandbox_manager.node_status", return_value=[fake_node]), \
         patch("backend.routers.nodes.AsyncSessionLocal") as mock_session_cls:
        mock_session = MagicMock()
        mock_session_cls.return_value.__aenter__.return_value = mock_session
        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = [active_m]
        f = asyncio.Future()
        f.set_result(mock_result)
        mock_session.execute.return_value = f

        nodes = await list_nodes()
        assert len(nodes) == 1
        assert nodes[0]["host"] == "local"
        assert nodes[0]["alive"] is True
        assert len(nodes[0]["missions"]) == 1
        assert nodes[0]["missions"][0]["mission_id"] == "test-mission-123"
