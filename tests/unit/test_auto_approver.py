"""Unit tests for backend/services/auto_approver.py — try_auto_approve()."""
from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock, patch, mock_open


def _make_gate(status: str = "pending", script_path: str = "/tmp/train.py") -> MagicMock:
    gate = MagicMock()
    gate.status = status
    gate.payload = {"script_path": script_path}
    gate.reviewer_note = None
    gate.resolved_at = None
    return gate


def _make_verdict(safe: bool, reason: str = "ok", classifier: str = "llm") -> MagicMock:
    v = MagicMock()
    v.safe = safe
    v.reason = reason
    v.classifier = classifier
    return v


@pytest.mark.asyncio
async def test_try_auto_approve_approves_safe_script(tmp_path):
    """Safe script → gate status set to approved, returns action='approved'."""
    from backend.services.auto_approver import try_auto_approve

    script = tmp_path / "train.py"
    script.write_text("print('hello')")
    gate = _make_gate(script_path=str(script))
    gate_in_session = _make_gate(script_path=str(script))

    mock_session = AsyncMock()
    mock_session.get = AsyncMock(return_value=gate_in_session)
    mock_session.commit = AsyncMock()
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=False)

    mock_classifier = AsyncMock()
    mock_classifier.classify = AsyncMock(return_value=_make_verdict(safe=True, reason="all requests target localhost"))

    code_provider = MagicMock()

    with patch("backend.services.auto_approver.AsyncSessionLocal", return_value=mock_session), \
         patch("backend.services.auto_approver.CodeSafetyClassifier", return_value=mock_classifier):
        result = await try_auto_approve("gate-123", code_provider)

    assert result.action == "approved"
    assert result.safe is True
    assert gate_in_session.status == "approved"


@pytest.mark.asyncio
async def test_try_auto_approve_blocks_unsafe_script(tmp_path):
    """Unsafe script → gate stays pending, returns action='blocked'."""
    from backend.services.auto_approver import try_auto_approve

    script = tmp_path / "train.py"
    script.write_text("import subprocess; subprocess.run(['rm', '-rf', '/'])")
    gate = _make_gate(script_path=str(script))

    mock_session = AsyncMock()
    mock_session.get = AsyncMock(return_value=gate)
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=False)

    mock_classifier = AsyncMock()
    mock_classifier.classify = AsyncMock(return_value=_make_verdict(safe=False, reason="destructive command detected"))

    code_provider = MagicMock()

    with patch("backend.services.auto_approver.AsyncSessionLocal", return_value=mock_session), \
         patch("backend.services.auto_approver.CodeSafetyClassifier", return_value=mock_classifier):
        result = await try_auto_approve("gate-123", code_provider)

    assert result.action == "blocked"
    assert result.safe is False
    assert gate.status == "pending"


@pytest.mark.asyncio
async def test_try_auto_approve_skips_missing_gate():
    """Gate not found → returns action='skipped'."""
    from backend.services.auto_approver import try_auto_approve

    mock_session = AsyncMock()
    mock_session.get = AsyncMock(return_value=None)
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=False)

    with patch("backend.services.auto_approver.AsyncSessionLocal", return_value=mock_session):
        result = await try_auto_approve("nonexistent-gate", MagicMock())

    assert result.action == "skipped"


@pytest.mark.asyncio
async def test_try_auto_approve_skips_already_resolved():
    """Gate already approved/rejected → returns action='skipped' without re-running classifier."""
    from backend.services.auto_approver import try_auto_approve

    gate = _make_gate(status="approved")

    mock_session = AsyncMock()
    mock_session.get = AsyncMock(return_value=gate)
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=False)

    code_provider = MagicMock()

    with patch("backend.services.auto_approver.AsyncSessionLocal", return_value=mock_session), \
         patch("backend.services.auto_approver.CodeSafetyClassifier") as mock_cls:
        result = await try_auto_approve("gate-123", code_provider)

    assert result.action == "skipped"
    mock_cls.assert_not_called()


@pytest.mark.asyncio
async def test_try_auto_approve_skips_missing_script():
    """Gate has no readable script_path → returns action='skipped'."""
    from backend.services.auto_approver import try_auto_approve

    gate = MagicMock()
    gate.status = "pending"
    gate.payload = {"script_path": "/nonexistent/path/train.py"}

    mock_session = AsyncMock()
    mock_session.get = AsyncMock(return_value=gate)
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=False)

    with patch("backend.services.auto_approver.AsyncSessionLocal", return_value=mock_session):
        result = await try_auto_approve("gate-123", MagicMock())

    assert result.action == "skipped"


@pytest.mark.asyncio
async def test_try_auto_approve_skips_when_no_provider():
    """code_provider=None → returns action='skipped' immediately."""
    from backend.services.auto_approver import try_auto_approve

    gate = _make_gate()

    mock_session = AsyncMock()
    mock_session.get = AsyncMock(return_value=gate)
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=False)

    with patch("backend.services.auto_approver.AsyncSessionLocal", return_value=mock_session), \
         patch("backend.services.auto_approver.CodeSafetyClassifier") as mock_cls, \
         patch("os.path.isfile", return_value=True):
        result = await try_auto_approve("gate-123", None)

    assert result.action == "skipped"
    mock_cls.assert_not_called()
