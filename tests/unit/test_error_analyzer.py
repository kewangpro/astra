"""Unit tests for ErrorAnalyzer (backend/agent/error_analyzer.py)."""
from __future__ import annotations

import asyncio
import os
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from backend.agent.error_analyzer import (
    ErrorAnalyzer,
    _extract_error_type,
    _extract_traceback,
    _MAX_TRACEBACK_LINES,
)


# ── helpers ───────────────────────────────────────────────────────────────────

SAMPLE_TRACEBACK = """\
Traceback (most recent call last):
  File "train.py", line 10, in <module>
    model = PPO("MlpPolicy", env, actor_lr=0.001)
TypeError: __init__() got an unexpected keyword argument 'actor_lr'
"""

IMPORT_ERROR = """\
Traceback (most recent call last):
  File "train.py", line 3, in <module>
    import gym
ModuleNotFoundError: No module named 'gym'
"""


def _make_provider(response: str = "# fixed code\npass") -> AsyncMock:
    provider = AsyncMock()
    provider.generate = AsyncMock(return_value=response)
    return provider


# ── _extract_error_type ───────────────────────────────────────────────────────

def test_extract_error_type_type_error():
    assert _extract_error_type(SAMPLE_TRACEBACK) == "TypeError"


def test_extract_error_type_module_not_found():
    assert _extract_error_type(IMPORT_ERROR) == "ModuleNotFoundError"


def test_extract_error_type_unknown():
    assert _extract_error_type("something went wrong") == "UnknownError"


def test_extract_error_type_value_error():
    tb = "ValueError: shapes (10,) and (5,) not aligned"
    assert _extract_error_type(tb) == "ValueError"


# ── _extract_traceback ────────────────────────────────────────────────────────

def test_extract_traceback_short():
    result = _extract_traceback(SAMPLE_TRACEBACK)
    assert "TypeError" in result
    assert "actor_lr" in result


def test_extract_traceback_truncates_long_output():
    long_output = "\n".join(f"line {i}" for i in range(_MAX_TRACEBACK_LINES + 20))
    result = _extract_traceback(long_output)
    lines = result.splitlines()
    assert len(lines) <= _MAX_TRACEBACK_LINES


# ── fix_script ────────────────────────────────────────────────────────────────

def test_fix_script_writes_fixed_file(tmp_path):
    script = tmp_path / "train.py"
    script.write_text("import gym\nenv = gym.make('CartPole-v1')\n")

    provider = _make_provider("import gymnasium as gym\nenv = gym.make('CartPole-v1')\n")
    analyzer = ErrorAnalyzer(provider)

    fixed_path = asyncio.get_event_loop().run_until_complete(
        analyzer.fix_script(str(script), IMPORT_ERROR, iteration=0)
    )

    assert os.path.exists(fixed_path)
    assert fixed_path.endswith(".fixed_0.py")
    assert "gymnasium" in open(fixed_path).read()


def test_fix_script_includes_prior_errors_in_prompt(tmp_path):
    script = tmp_path / "train.py"
    script.write_text("model = PPO('MlpPolicy', env, actor_lr=0.001)\n")

    captured = {}

    async def mock_generate(messages, config):
        captured["messages"] = messages
        return "# fixed"

    provider = AsyncMock()
    provider.generate = mock_generate
    analyzer = ErrorAnalyzer(provider)

    asyncio.get_event_loop().run_until_complete(
        analyzer.fix_script(
            str(script),
            SAMPLE_TRACEBACK,
            iteration=2,
            prior_errors=["TypeError: unexpected keyword argument 'entropy_coeff'"],
        )
    )

    user_content = captured["messages"][1].content
    assert "Previous fix attempts" in user_content
    assert "entropy_coeff" in user_content


def test_fix_script_strips_markdown_fences(tmp_path):
    script = tmp_path / "train.py"
    script.write_text("pass\n")

    fenced_response = "```python\nimport gymnasium as gym\n```"
    provider = _make_provider(fenced_response)
    analyzer = ErrorAnalyzer(provider)

    fixed_path = asyncio.get_event_loop().run_until_complete(
        analyzer.fix_script(str(script), SAMPLE_TRACEBACK, iteration=0)
    )
    content = open(fixed_path).read()
    assert "```" not in content
    assert "import gymnasium" in content


def test_fix_script_raises_if_script_missing(tmp_path):
    provider = _make_provider()
    analyzer = ErrorAnalyzer(provider)

    with pytest.raises(FileNotFoundError):
        asyncio.get_event_loop().run_until_complete(
            analyzer.fix_script(str(tmp_path / "nonexistent.py"), SAMPLE_TRACEBACK)
        )


def test_fix_script_uses_iteration_in_filename(tmp_path):
    script = tmp_path / "train.py"
    script.write_text("pass\n")

    provider = _make_provider()
    analyzer = ErrorAnalyzer(provider)

    for i in range(3):
        fixed_path = asyncio.get_event_loop().run_until_complete(
            analyzer.fix_script(str(script), SAMPLE_TRACEBACK, iteration=i)
        )
        assert f".fixed_{i}.py" in fixed_path


# ── _store_lesson ─────────────────────────────────────────────────────────────

def test_store_lesson_swallows_exceptions(tmp_path):
    provider = _make_provider()
    analyzer = ErrorAnalyzer(provider)

    # Should not raise even if vector_memory is unavailable
    with patch("backend.agent.error_analyzer.logger") as mock_log:
        analyzer._store_lesson("TypeError", SAMPLE_TRACEBACK, mission_id=None, domain=None)
    # The test passes if no exception is raised


def test_store_lesson_called_after_fix(tmp_path):
    script = tmp_path / "train.py"
    script.write_text("pass\n")

    provider = _make_provider()
    analyzer = ErrorAnalyzer(provider)

    with patch.object(analyzer, "_store_lesson") as mock_store:
        asyncio.get_event_loop().run_until_complete(
            analyzer.fix_script(
                str(script), SAMPLE_TRACEBACK, iteration=0,
                mission_id="test-mission", domain="rl"
            )
        )
        mock_store.assert_called_once()
        args = mock_store.call_args[0]
        assert args[0] == "TypeError"
        assert args[2] == "test-mission"
        assert args[3] == "rl"


# ── _strip_fences ─────────────────────────────────────────────────────────────

def test_strip_fences_removes_python_fence():
    from backend.agent.error_analyzer import ErrorAnalyzer
    result = ErrorAnalyzer._strip_fences("```python\ncode here\n```")
    assert "```" not in result
    assert "code here" in result


def test_strip_fences_removes_llm_stop_tokens():
    result = ErrorAnalyzer._strip_fences("code\n<|im_end|>\n")
    assert "<|im_end|>" not in result
    assert "code" in result


def test_strip_fences_passthrough_clean_code():
    code = "import numpy as np\nx = np.array([1, 2, 3])\n"
    assert ErrorAnalyzer._strip_fences(code).strip() == code.strip()
