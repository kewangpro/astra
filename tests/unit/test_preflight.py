"""Unit tests for PreflightChecker and PreflightResult in services/preflight.py."""
from __future__ import annotations

import os
from unittest.mock import patch

import pytest

from backend.services.preflight import PreflightChecker, PreflightResult


# ── PreflightResult ───────────────────────────────────────────────────────────

class TestPreflightResult:
    def test_summary_all_passed(self):
        result = PreflightResult(
            passed=True,
            checks=[
                {"name": "a", "passed": True, "detail": ""},
                {"name": "b", "passed": True, "detail": ""},
            ],
        )
        assert result.summary() == "2/2 checks passed"

    def test_summary_some_failed(self):
        result = PreflightResult(
            passed=False,
            checks=[
                {"name": "a", "passed": True, "detail": ""},
                {"name": "b", "passed": False, "detail": "err"},
            ],
        )
        assert result.summary() == "1/2 checks passed"

    def test_summary_none_passed(self):
        result = PreflightResult(
            passed=False,
            checks=[{"name": "x", "passed": False, "detail": "err"}],
        )
        assert result.summary() == "0/1 checks passed"

    def test_summary_empty_checks(self):
        result = PreflightResult(passed=True, checks=[])
        assert result.summary() == "0/0 checks passed"


# ── _check_packages ───────────────────────────────────────────────────────────

class TestCheckPackages:
    def test_known_importable_package(self):
        # 'os' is always importable; inject it into the required list temporarily
        with patch("backend.services.preflight._REQUIRED_PACKAGES", {"test_type": ["os"]}):
            results = PreflightChecker._check_packages("test_type")
        assert len(results) == 1
        assert results[0]["passed"] is True
        assert results[0]["name"] == "import_os"

    def test_unknown_package_fails(self):
        with patch("backend.services.preflight._REQUIRED_PACKAGES",
                   {"test_type": ["_nonexistent_pkg_xyz_"]}):
            results = PreflightChecker._check_packages("test_type")
        assert results[0]["passed"] is False
        assert "_nonexistent_pkg_xyz_" in results[0]["detail"]

    def test_unknown_task_type_returns_empty(self):
        results = PreflightChecker._check_packages("completely_unknown_type")
        assert results == []

    def test_ml_packages_checked(self):
        # Real ml packages — check structure, not whether they're installed
        results = PreflightChecker._check_packages("ml")
        assert all("name" in r and "passed" in r for r in results)
        assert len(results) >= 1

    def test_multiple_packages_all_checked(self):
        with patch("backend.services.preflight._REQUIRED_PACKAGES",
                   {"multi": ["os", "sys"]}):
            results = PreflightChecker._check_packages("multi")
        assert len(results) == 2
        assert all(r["passed"] for r in results)


# ── _check_data_dir_writable ──────────────────────────────────────────────────

class TestCheckDataDirWritable:
    def test_writable_tmp_dir(self, tmp_path, monkeypatch):
        monkeypatch.setattr("backend.services.preflight.settings",
                            type("S", (), {"data_path": str(tmp_path)})())
        result = PreflightChecker._check_data_dir_writable("test-mission-id")
        assert result["passed"] is True
        assert result["name"] == "data_dir_writable"

    def test_non_writable_dir_fails(self, tmp_path, monkeypatch):
        monkeypatch.setattr("backend.services.preflight.settings",
                            type("S", (), {"data_path": "/nonexistent/path/xyz"})())
        result = PreflightChecker._check_data_dir_writable("test-mission-id")
        assert result["passed"] is False

    def test_probe_file_cleaned_up(self, tmp_path, monkeypatch):
        monkeypatch.setattr("backend.services.preflight.settings",
                            type("S", (), {"data_path": str(tmp_path)})())
        PreflightChecker._check_data_dir_writable("cleanup-test")
        probe = tmp_path / "missions" / "cleanup-test" / ".preflight_probe"
        assert not probe.exists()


# ── PreflightChecker.run ──────────────────────────────────────────────────────

class TestPreflightCheckerRun:
    def test_run_returns_result_object(self, tmp_path, monkeypatch):
        monkeypatch.setattr("backend.services.preflight.settings",
                            type("S", (), {"data_path": str(tmp_path)})())
        result = PreflightChecker().run("mission-run-test", "unknown_task")
        assert isinstance(result, PreflightResult)

    def test_run_passes_when_all_checks_pass(self, tmp_path, monkeypatch):
        monkeypatch.setattr("backend.services.preflight.settings",
                            type("S", (), {"data_path": str(tmp_path)})())
        # "unknown_task" has no required packages → only dir + python checks
        result = PreflightChecker().run("mission-run-test", "unknown_task")
        assert result.passed is True

    def test_run_fails_when_package_missing(self, tmp_path, monkeypatch):
        monkeypatch.setattr("backend.services.preflight.settings",
                            type("S", (), {"data_path": str(tmp_path)})())
        with patch("backend.services.preflight._REQUIRED_PACKAGES",
                   {"bad_type": ["_pkg_that_does_not_exist_"]}):
            result = PreflightChecker().run("mission-fail", "bad_type")
        assert result.passed is False
