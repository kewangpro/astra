from __future__ import annotations

import pytest

from backend.evaluator.stress_tester import StressTester


def test_skipped_for_unknown_task_type():
    st = StressTester("unknown")
    result = st.run("/fake/checkpoint")
    assert result["status"] == "skipped"


def test_seeds_tested_count():
    st = StressTester("rl", num_seeds=5)
    result = st.run("/fake/checkpoint")
    assert result["seeds_tested"] == 5
    assert len(result["results"]) == 5


def test_seed_reproducibility():
    st = StressTester("rl", num_seeds=3)
    r1 = st.run("/fake/checkpoint")
    r2 = st.run("/fake/checkpoint")
    # Same seed sequence → same results
    for a, b in zip(r1["results"], r2["results"]):
        assert a["seed"] == b["seed"]
        for key in a:
            assert a[key] == b[key]


def test_results_keyed_by_seed():
    st = StressTester("sft", num_seeds=3)
    result = st.run("/fake/checkpoint")
    seeds = [r["seed"] for r in result["results"]]
    assert seeds == list(range(3))


def test_task_type_recorded():
    st = StressTester("ml")
    result = st.run("/fake/checkpoint")
    assert result["task_type"] == "ml"


def test_stress_report_summary_fields():
    """After hardening, run() includes summary stats."""
    st = StressTester("rl")
    result = st.run("/fake/checkpoint")
    # Check new fields added in Phase 6.3 hardening
    assert "mean" in result
    assert "std" in result
    assert "min" in result
    assert "max" in result
    assert "reproducible" in result
