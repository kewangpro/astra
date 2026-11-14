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


# ── _check_remote_platform_imports ────────────────────────────────────────────

class TestCheckRemotePlatformImports:
    def test_skipped_for_local_task_types(self):
        out = PreflightChecker._check_remote_platform_imports("rl")
        assert out["passed"] is True
        assert "skipped" in out["detail"]

    def test_skipped_without_sandbox_host(self, monkeypatch):
        monkeypatch.setattr("backend.services.preflight.settings.sandbox_host", "")
        out = PreflightChecker._check_remote_platform_imports("dpo")
        assert out["passed"] is True
        assert "skipped" in out["detail"]

    def test_passes_when_imports_ok(self, monkeypatch):
        monkeypatch.setattr("backend.services.preflight.settings.sandbox_host", "mac-mini.local")
        monkeypatch.setattr("backend.services.preflight.settings.sandbox_python", "")
        with patch("backend.services.preflight.subprocess.run") as run:
            run.return_value = type("R", (), {"returncode": 0, "stdout": "", "stderr": ""})()
            out = PreflightChecker._check_remote_platform_imports("dpo")
        assert out["passed"] is True
        assert out["name"] == "remote_platform_imports"
        cmd = run.call_args.args[0][2]
        assert "import astragym, astrarl, astratune" in cmd
        assert "finetune-env" in cmd or "python" in cmd

    def test_fails_with_vendor_hint(self, monkeypatch):
        monkeypatch.setattr("backend.services.preflight.settings.sandbox_host", "mac-mini.local")
        with patch("backend.services.preflight.subprocess.run") as run:
            run.return_value = type(
                "R", (),
                {"returncode": 1, "stdout": "", "stderr": "ModuleNotFoundError: No module named 'astratune'"},
            )()
            out = PreflightChecker._check_remote_platform_imports("grpo")
        assert out["passed"] is False
        assert "VENDOR.md" in out["detail"]
        assert "pip install -e" in out["detail"]
        assert "astratune" in out["detail"]

    def test_unreachable_host_does_not_block(self, monkeypatch):
        monkeypatch.setattr("backend.services.preflight.settings.sandbox_host", "mac-mini.local")
        with patch("backend.services.preflight.subprocess.run",
                   side_effect=RuntimeError("ssh timeout")):
            out = PreflightChecker._check_remote_platform_imports("distill")
        assert out["passed"] is True
        assert "unreachable" in out["detail"]


# ── _check_remote_astra_pkg_root ──────────────────────────────────────────────

class TestCheckRemoteAstraPkgRoot:
    def test_skipped_for_local_task_types(self):
        out = PreflightChecker._check_remote_astra_pkg_root("rl")
        assert out["passed"] is True
        assert "skipped" in out["detail"]

    def test_skipped_without_sandbox_host(self, monkeypatch):
        monkeypatch.setattr("backend.services.preflight.settings.sandbox_host", "")
        out = PreflightChecker._check_remote_astra_pkg_root("dpo")
        assert out["passed"] is True
        assert "skipped" in out["detail"]

    def test_passes_when_directory_exists(self, monkeypatch):
        monkeypatch.setattr("backend.services.preflight.settings.sandbox_host", "mac-mini.local")
        with patch("backend.services.preflight.subprocess.run") as run:
            run.return_value = type("R", (), {"returncode": 0, "stdout": "yes\n", "stderr": ""})()
            out = PreflightChecker._check_remote_astra_pkg_root("prompt")
        assert out["passed"] is True
        assert out["name"] == "remote_astra_pkg_root"
        cmd = run.call_args.args[0][2]
        assert "test -d" in cmd
        assert "astra-pkg" in cmd or "astra_pkg" in out["detail"]

    def test_fails_with_recipe_rsync_hint(self, monkeypatch):
        monkeypatch.setattr("backend.services.preflight.settings.sandbox_host", "mac-mini.local")
        with patch("backend.services.preflight.subprocess.run") as run:
            run.return_value = type("R", (), {"returncode": 0, "stdout": "no\n", "stderr": ""})()
            out = PreflightChecker._check_remote_astra_pkg_root("grpo")
        assert out["passed"] is False
        assert "astra_pkg_root" in out["detail"] or "ASTRA_PKG_ROOT" in out["detail"]
        assert "rsync" in out["detail"] or "VENDOR.md" in out["detail"]

    def test_unreachable_host_does_not_block(self, monkeypatch):
        monkeypatch.setattr("backend.services.preflight.settings.sandbox_host", "mac-mini.local")
        with patch("backend.services.preflight.subprocess.run",
                   side_effect=RuntimeError("ssh timeout")):
            out = PreflightChecker._check_remote_astra_pkg_root("agrpo")
        assert out["passed"] is True
        assert "unreachable" in out["detail"]
