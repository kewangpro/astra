"""Unit tests for RequirementManifest, ManifestGenerator, ManifestEvaluator."""
from __future__ import annotations

import os
import tempfile

import pytest

from backend.models.manifest import Requirement, RequirementManifest
from backend.services.manifest_generator import generate_manifest
from backend.evaluator.manifest_evaluator import ManifestEvaluator


# ── manifest model ─────────────────────────────────────────────────────────────

class TestRequirementManifest:
    def test_is_complete_empty(self):
        m = RequirementManifest(mission_id="x")
        assert not m.is_complete()

    def test_is_complete_all_passed(self):
        m = RequirementManifest(
            mission_id="x",
            requirements=[Requirement(id="r1", description="d", category="c", check_type="no_sandbox_error", passed=True)],
        )
        assert m.is_complete()

    def test_is_complete_partial(self):
        m = RequirementManifest(
            mission_id="x",
            requirements=[
                Requirement(id="r1", description="d", category="c", check_type="no_sandbox_error", passed=True),
                Requirement(id="r2", description="d2", category="c", check_type="no_sandbox_error", passed=False),
            ],
        )
        assert not m.is_complete()

    def test_summary(self):
        m = RequirementManifest(
            mission_id="x",
            requirements=[
                Requirement(id="r1", description="d", category="c", check_type="no_sandbox_error", passed=True),
                Requirement(id="r2", description="d2", category="c", check_type="no_sandbox_error", passed=False),
            ],
        )
        s = m.summary()
        assert s["total"] == 2
        assert s["passed"] == 1
        assert s["complete"] is False

    def test_round_trip_serialization(self):
        m = generate_manifest("abc", "Train iris to 90% accuracy", "ml", {"accuracy": 0.9})
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "req.json")
            m.save(path)
            m2 = RequirementManifest.load(path)
        assert m2.mission_id == "abc"
        assert len(m2.requirements) == len(m.requirements)
        for r1, r2 in zip(m.requirements, m2.requirements):
            assert r1.id == r2.id
            assert r1.check_type == r2.check_type


# ── manifest generator ─────────────────────────────────────────────────────────

class TestManifestGenerator:
    def test_ml_generates_three_reqs(self):
        m = generate_manifest("m1", "iris to 90% accuracy", "ml", {"accuracy": 0.9})
        types = [r.check_type for r in m.requirements]
        assert "no_sandbox_error" in types
        assert "file_exists" in types
        assert "metric_threshold" in types
        assert len(m.requirements) == 3

    def test_metric_threshold_values(self):
        m = generate_manifest("m1", "goal", "ml", {"accuracy": 0.9})
        mt = next(r for r in m.requirements if r.check_type == "metric_threshold")
        assert mt.metric_name == "accuracy"
        assert mt.threshold == pytest.approx(0.9)
        assert mt.operator == ">="

    def test_lower_is_better_operator(self):
        m = generate_manifest("m1", "goal", "sft", {"eval_loss": 0.5})
        mt = next(r for r in m.requirements if r.check_type == "metric_threshold")
        assert mt.operator == "<="

    def test_multiple_metrics(self):
        m = generate_manifest("m1", "goal", "ml", {"accuracy": 0.9, "f1": 0.85})
        mt_reqs = [r for r in m.requirements if r.check_type == "metric_threshold"]
        assert len(mt_reqs) == 2

    def test_empty_target_metric_no_performance_req(self):
        m = generate_manifest("m1", "goal", "ml", {})
        mt_reqs = [r for r in m.requirements if r.check_type == "metric_threshold"]
        assert len(mt_reqs) == 0


# ── manifest evaluator ─────────────────────────────────────────────────────────

class TestManifestEvaluator:
    def _ev(self):
        return ManifestEvaluator()

    def test_no_sandbox_error_passes(self):
        m = generate_manifest("m1", "goal", "ml", {})
        m2 = self._ev().evaluate(m, {}, "/tmp", sandbox_ok=True)
        stab = next(r for r in m2.requirements if r.check_type == "no_sandbox_error")
        assert stab.passed

    def test_no_sandbox_error_fails_on_error(self):
        m = generate_manifest("m1", "goal", "ml", {})
        m2 = self._ev().evaluate(m, {}, "/tmp", sandbox_ok=False)
        stab = next(r for r in m2.requirements if r.check_type == "no_sandbox_error")
        assert not stab.passed

    def test_file_exists_passes(self):
        m = generate_manifest("m1", "goal", "ml", {})
        with tempfile.TemporaryDirectory() as d:
            os.makedirs(os.path.join(d, "checkpoints"))
            open(os.path.join(d, "checkpoints", "model.joblib"), "w").close()
            m2 = self._ev().evaluate(m, {}, d, sandbox_ok=True)
        art = next(r for r in m2.requirements if r.check_type == "file_exists")
        assert art.passed

    def test_file_exists_fails_when_missing(self):
        m = generate_manifest("m1", "goal", "ml", {})
        with tempfile.TemporaryDirectory() as d:
            m2 = self._ev().evaluate(m, {}, d, sandbox_ok=True)
        art = next(r for r in m2.requirements if r.check_type == "file_exists")
        assert not art.passed

    def test_metric_threshold_passes(self):
        m = generate_manifest("m1", "goal", "ml", {"accuracy": 0.9})
        with tempfile.TemporaryDirectory() as d:
            m2 = self._ev().evaluate(m, {"accuracy": 1.0}, d, sandbox_ok=True)
        perf = next(r for r in m2.requirements if r.check_type == "metric_threshold")
        assert perf.passed

    def test_metric_threshold_fails_below_target(self):
        m = generate_manifest("m1", "goal", "ml", {"accuracy": 0.9})
        with tempfile.TemporaryDirectory() as d:
            m2 = self._ev().evaluate(m, {"accuracy": 0.8}, d, sandbox_ok=True)
        perf = next(r for r in m2.requirements if r.check_type == "metric_threshold")
        assert not perf.passed

    def test_metric_suffix_match(self):
        """'accuracy' target should match 'validation_accuracy' in metrics."""
        m = generate_manifest("m1", "goal", "ml", {"accuracy": 0.9})
        with tempfile.TemporaryDirectory() as d:
            m2 = self._ev().evaluate(m, {"validation_accuracy": 1.0}, d, sandbox_ok=True)
        perf = next(r for r in m2.requirements if r.check_type == "metric_threshold")
        assert perf.passed

    def test_already_passed_not_re_evaluated(self):
        """A requirement that passed stays passed even with contradictory new evidence."""
        m = generate_manifest("m1", "goal", "ml", {"accuracy": 0.9})
        # First pass with good metrics
        with tempfile.TemporaryDirectory() as d:
            os.makedirs(os.path.join(d, "checkpoints"))
            open(os.path.join(d, "checkpoints", "model.joblib"), "w").close()
            m2 = self._ev().evaluate(m, {"accuracy": 1.0}, d, sandbox_ok=True)
        assert m2.is_complete()
        # Second pass with bad metrics — already-passed reqs must stay passed
        with tempfile.TemporaryDirectory() as d2:
            m3 = self._ev().evaluate(m2, {"accuracy": 0.1}, d2, sandbox_ok=False)
        assert m3.is_complete()

    def test_complete_only_when_all_pass(self):
        m = generate_manifest("m1", "goal", "ml", {"accuracy": 0.9})
        with tempfile.TemporaryDirectory() as d:
            # metric passes but no checkpoint, sandbox fails
            m2 = self._ev().evaluate(m, {"accuracy": 1.0}, d, sandbox_ok=False)
        assert not m2.is_complete()
