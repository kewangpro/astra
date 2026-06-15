"""Unit tests for mission_state.py — load, update, _primary_score."""
from __future__ import annotations

import json
import os

import pytest

from backend.services.mission_state import load, update, _primary_score


# ── _primary_score ────────────────────────────────────────────────────────────

class TestPrimaryScore:
    def test_returns_first_numeric_value(self):
        assert _primary_score({"mean_reward": 42.5}) == pytest.approx(42.5)

    def test_skips_non_numeric_returns_next(self):
        assert _primary_score({"label": "good", "acc": 0.9}) == pytest.approx(0.9)

    def test_returns_none_when_no_numeric(self):
        assert _primary_score({"label": "ok", "status": "done"}) is None

    def test_empty_dict_returns_none(self):
        assert _primary_score({}) is None

    def test_integer_value_cast_to_float(self):
        result = _primary_score({"steps": 1000})
        assert result == pytest.approx(1000.0)

    def test_string_numeric_skipped(self):
        # "42" is a string — should skip it
        assert _primary_score({"score": "not_a_number"}) is None

    def test_none_value_skipped(self):
        assert _primary_score({"score": None}) is None


# ── load ──────────────────────────────────────────────────────────────────────

class TestLoad:
    def test_returns_default_when_file_missing(self, tmp_path, monkeypatch):
        monkeypatch.setattr("backend.services.mission_state.settings",
                            type("S", (), {"data_path": str(tmp_path)})())
        state = load("mission-abc")
        assert state["mission_id"] == "mission-abc"
        assert state["best_score"] is None
        assert state["iteration_history"] == []

    def test_loads_existing_file(self, tmp_path, monkeypatch):
        monkeypatch.setattr("backend.services.mission_state.settings",
                            type("S", (), {"data_path": str(tmp_path)})())
        mission_id = "mission-xyz"
        p = tmp_path / "missions" / mission_id
        p.mkdir(parents=True)
        data = {"version": "1.0", "mission_id": mission_id, "best_score": 0.95,
                "best_hyperparameters": {}, "best_algorithm": None,
                "iteration_history": [], "lessons_learned": [], "last_updated": None}
        (p / "MISSION_MANIFEST.json").write_text(json.dumps(data))
        state = load(mission_id)
        assert state["best_score"] == pytest.approx(0.95)

    def test_returns_default_on_corrupt_json(self, tmp_path, monkeypatch):
        monkeypatch.setattr("backend.services.mission_state.settings",
                            type("S", (), {"data_path": str(tmp_path)})())
        mission_id = "broken"
        p = tmp_path / "missions" / mission_id
        p.mkdir(parents=True)
        (p / "MISSION_MANIFEST.json").write_text("{not valid json")
        state = load(mission_id)
        assert state["mission_id"] == mission_id
        assert state["best_score"] is None


# ── update ────────────────────────────────────────────────────────────────────

class TestUpdate:
    @pytest.fixture(autouse=True)
    def patch_settings(self, tmp_path, monkeypatch):
        monkeypatch.setattr("backend.services.mission_state.settings",
                            type("S", (), {"data_path": str(tmp_path)})())
        self.tmp_path = tmp_path

    def test_creates_manifest_on_first_update(self):
        state = update("m1", 1, {"hyperparameters": {"lr": 0.01}, "algorithm": "ppo"}, {"reward": 5.0})
        manifest = self.tmp_path / "missions" / "m1" / "MISSION_MANIFEST.json"
        assert manifest.exists()

    def test_best_score_updated_when_improved(self):
        update("m1", 1, {"hyperparameters": {}, "algorithm": "ppo"}, {"reward": 5.0})
        state = update("m1", 2, {"hyperparameters": {}, "algorithm": "ppo"}, {"reward": 8.0})
        assert state["best_score"] == pytest.approx(8.0)

    def test_best_score_not_downgraded(self):
        update("m1", 1, {"hyperparameters": {}, "algorithm": "ppo"}, {"reward": 8.0})
        state = update("m1", 2, {"hyperparameters": {}, "algorithm": "ppo"}, {"reward": 3.0})
        assert state["best_score"] == pytest.approx(8.0)

    def test_iteration_history_appended(self):
        update("m1", 1, {"hyperparameters": {}, "algorithm": "ppo"}, {"reward": 1.0})
        state = update("m1", 2, {"hyperparameters": {}, "algorithm": "ppo"}, {"reward": 2.0})
        assert len(state["iteration_history"]) == 2

    def test_iteration_history_capped_at_20(self):
        for i in range(25):
            update("m1", i, {"hyperparameters": {}, "algorithm": "ppo"}, {"reward": float(i)})
        state = load("m1")
        assert len(state["iteration_history"]) == 20

    def test_lessons_merged(self):
        update("m1", 1, {}, {"r": 1.0}, lessons=["lesson A"])
        state = update("m1", 2, {}, {"r": 2.0}, lessons=["lesson B"])
        assert "lesson A" in state["lessons_learned"]
        assert "lesson B" in state["lessons_learned"]

    def test_duplicate_lessons_deduplicated(self):
        update("m1", 1, {}, {"r": 1.0}, lessons=["repeat"])
        state = update("m1", 2, {}, {"r": 2.0}, lessons=["repeat"])
        assert state["lessons_learned"].count("repeat") == 1

    def test_last_updated_set(self):
        state = update("m1", 1, {}, {"r": 1.0})
        assert state["last_updated"] is not None

    def test_no_score_in_metrics_does_not_update_best(self):
        state = update("m1", 1, {}, {"label": "bad"})
        assert state["best_score"] is None

    def test_algorithm_tracked_in_best(self):
        state = update("m1", 1, {"algorithm": "sac", "hyperparameters": {}}, {"r": 9.0})
        assert state["best_algorithm"] == "sac"
