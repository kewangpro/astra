"""Unit tests for demo_reasoning STaR scaffold (non-Ensemble)."""
from __future__ import annotations

from demo_reasoning.oracle import (
    build_prompt,
    load_cases,
    rationalize_hint,
    score_completion,
)
from demo_reasoning.usecase import get_runner


def test_demo_reasoning_cases_and_oracle():
    cases = load_cases()
    assert len(cases) >= 3
    c0 = cases[0]
    assert build_prompt(c0)
    assert score_completion(f"<think>x</think>\n{c0['answer']}", c0) == 1.0
    assert score_completion("wrong", c0) == 0.0
    hint = rationalize_hint(c0)
    assert c0["answer"] in hint
    assert "<think>" in hint


def test_demo_reasoning_get_runner_star():
    runner = get_runner("star")
    assert callable(runner)
