"""Unit tests for Mission.host — derived (not stored) node identity used by
the Nodes panel and mission grid badge."""
from __future__ import annotations

from unittest.mock import patch

from backend.models.mission import Mission


def test_host_is_local_when_subprocess_pid_set():
    mission = Mission(subprocess_pid=12345, remote_pid=None)
    assert mission.host == "local"


def test_host_is_sandbox_host_when_remote_pid_set():
    mission = Mission(subprocess_pid=None, remote_pid=999)
    with patch("backend.config.settings.sandbox_host", "mac-mini.local"):
        assert mission.host == "mac-mini.local"


def test_host_is_none_when_remote_pid_set_but_no_sandbox_host_configured():
    mission = Mission(subprocess_pid=None, remote_pid=999)
    with patch("backend.config.settings.sandbox_host", ""):
        assert mission.host is None


def test_host_is_none_when_neither_pid_set():
    mission = Mission(subprocess_pid=None, remote_pid=None)
    assert mission.host is None


def test_host_prefers_local_when_both_pids_set():
    """subprocess_pid and remote_pid should never both be set in practice, but
    if they are, local wins — matches the order of checks in the property."""
    mission = Mission(subprocess_pid=12345, remote_pid=999)
    assert mission.host == "local"
