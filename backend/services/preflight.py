"""
PreflightChecker — Step 7.4.

Runs a suite of lightweight checks before a mission loop starts to ensure
a stable baseline exists. Failures are logged as warnings (not fatal) so
a single missing optional dependency doesn't abort a valid mission.
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from typing import Optional

from backend.config import settings
from backend.logging_config import get_logger

logger = get_logger(__name__)

# Packages that must be importable per task type
_REQUIRED_PACKAGES: dict = {
    "ml":  ["sklearn", "joblib"],
    "rl":  ["stable_baselines3", "gymnasium"],
    "sft": ["transformers", "peft", "trl"],
}


@dataclass
class PreflightResult:
    passed: bool
    checks: list = field(default_factory=list)   # list of {"name", "passed", "detail"}

    def summary(self) -> str:
        total = len(self.checks)
        ok = sum(1 for c in self.checks if c["passed"])
        return f"{ok}/{total} checks passed"


class PreflightChecker:

    def run(self, mission_id: str, task_type: str) -> PreflightResult:
        checks = []

        checks.append(self._check_data_dir_writable(mission_id))
        checks.append(self._check_sandbox_python())
        checks += self._check_packages(task_type)

        passed = all(c["passed"] for c in checks)
        result = PreflightResult(passed=passed, checks=checks)
        if passed:
            logger.info("Preflight: PASS — %s — mission=%s", result.summary(), mission_id)
        else:
            failed = [c["name"] for c in checks if not c["passed"]]
            logger.warning("Preflight: WARN — %s failed — mission=%s", failed, mission_id)
        return result

    # ── individual checks ─────────────────────────────────────────────────────

    @staticmethod
    def _check_data_dir_writable(mission_id: str) -> dict:
        mission_dir = os.path.join(settings.data_path, "missions", mission_id)
        try:
            os.makedirs(mission_dir, exist_ok=True)
            probe = os.path.join(mission_dir, ".preflight_probe")
            with open(probe, "w") as f:
                f.write("ok")
            os.remove(probe)
            return {"name": "data_dir_writable", "passed": True, "detail": mission_dir}
        except Exception as exc:
            return {"name": "data_dir_writable", "passed": False, "detail": str(exc)}

    @staticmethod
    def _check_sandbox_python() -> dict:
        from pathlib import Path
        project_root = Path(__file__).resolve().parents[2]
        venv_python = project_root / ".venv" / "bin" / "python"
        if venv_python.exists():
            return {"name": "sandbox_python", "passed": True, "detail": str(venv_python)}
        # Fall back to sys.executable
        return {"name": "sandbox_python", "passed": True, "detail": f"venv not found, using {sys.executable}"}

    @staticmethod
    def _check_packages(task_type: str) -> list:
        results = []
        for pkg in _REQUIRED_PACKAGES.get(task_type, []):
            try:
                __import__(pkg)
                results.append({"name": f"import_{pkg}", "passed": True, "detail": pkg})
            except ImportError:
                results.append({"name": f"import_{pkg}", "passed": False,
                                 "detail": f"{pkg} not importable in current environment"})
        return results
