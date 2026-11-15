"""Unit tests for AstraEval suites + gate + tournament seeds."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from astraeval.gate import reject_over_ceiling, should_unlock_hard_phase, suite_ceiling
from astraeval.registry import get_suite, parse_suite_ref, tournament_seed
from astraeval.suites.hstu_hr10 import HstuHr10Suite
from astraeval.suites.routing_bare import RoutingBareEvalSuite


def test_parse_suite_ref():
    assert parse_suite_ref("hstu_hr10@2") == ("hstu_hr10", 2)
    assert parse_suite_ref("routing_bare") == ("routing_bare", 1)


def test_tournament_seed_rule():
    assert tournament_seed(0) == 2000
    assert tournament_seed(7, suite_id="rl_tournament", suite_rev=1) == 2007


def test_hstu_hr10_suite_from_dict_and_json(tmp_path: Path):
    suite = HstuHr10Suite(suite_rev=1)
    r = suite.run({"hr_at_10": 0.12, "ndcg_at_10": 0.08})
    assert r.primary == pytest.approx(0.12)
    assert r.primary_metric == "hr_at_10"
    assert r.case_ids == []
    assert r.population_fingerprint

    side = tmp_path / "metrics.json"
    side.write_text(json.dumps({"hr_at_10": 0.09}))
    r2 = suite.run(side)
    assert r2.primary == pytest.approx(0.09)


def test_routing_bare_suite_from_report():
    suite = RoutingBareEvalSuite(suite_rev=5)
    r = suite.run(
        None,
        report={
            "primary": 0.81,
            "case_ids": ["route_x", "route_y"],
            "frozen_clock": "2026-11-14T00:00:00Z",
            "case_population": ["route_x", "route_y", "route_z"],
        },
    )
    assert r.primary == pytest.approx(0.81)
    assert r.case_ids == ["route_x", "route_y"]
    assert r.suite_rev == 5
    assert r.population_fingerprint
    # Same population + clock → same fingerprint
    r2 = suite.run(
        None,
        report={
            "primary": 0.5,
            "case_ids": ["route_y"],
            "frozen_clock": "2026-11-14T00:00:00Z",
            "case_population": ["route_x", "route_y", "route_z"],
        },
    )
    assert r2.population_fingerprint == r.population_fingerprint


def test_suite_ceiling_and_reject():
    cap = suite_ceiling("hstu_hr10@1")
    assert cap["hr_at_10"] == pytest.approx(0.35)
    assert reject_over_ceiling({"hr_at_10": 0.05}, "hstu_hr10@1") is None
    hit = reject_over_ceiling({"hr_at_10": 0.99}, "hstu_hr10@1")
    assert hit is not None
    assert hit[0] == "hr_at_10"


def test_get_suite_unknown():
    with pytest.raises(KeyError):
        get_suite("no_such_suite")


def test_hard_phase_unlock_hint():
    assert should_unlock_hard_phase(plateau_iters=2) is False
    assert should_unlock_hard_phase(plateau_iters=3) is True
