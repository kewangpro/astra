"""
PreflightChecker — Step 7.4.

Runs a suite of lightweight checks before a mission loop starts to ensure
a stable baseline exists. Failures are logged as warnings (not fatal) so
a single missing optional dependency doesn't abort a valid mission.
"""
from __future__ import annotations

import os
import subprocess
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
        checks += self._check_remote_script(task_type)

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
    def _check_remote_script(task_type: str) -> list:
        """For finetune-remote task types, confirm the training script actually
        exists at {finetune_dir}/{script} on the sandbox host.

        astra dispatches an os.execv wrapper naming that path; if it is wrong the
        run dies immediately after the SSH round-trip, having already generated a
        script, launched a sandbox and burned an iteration — mission a8675c39
        failed exactly this way. One `test -f` before dispatch turns a launch
        failure into a preflight warning that names the path it looked for.

        This is a LAYOUT check, not just an existence check. The remote's
        ~/finetune IS the repo's finetune/ directory, while a checkout nests it
        one level deeper — the mismatch that produced ensemble's doubled
        finetune/finetune/data_ft/ path. astra builds remote paths as an absolute
        finetune_dir plus a distinct suffix so it cannot double them itself, but
        a recipe pointing finetune_dir at a repo root rather than the deployed
        directory would still resolve to a script that isn't there, and this is
        what catches that.

        Non-fatal by design: preflight warns, it does not block. A transient SSH
        failure must not stop a mission that would otherwise run, so an
        unreachable host passes with the reason recorded rather than failing.
        """
        from backend.sandbox.manager import _FINETUNE_REMOTE_TASK_TYPES
        if task_type not in _FINETUNE_REMOTE_TASK_TYPES:
            return []
        from backend.agent.code_generator import _resolve_hyperparams
        hp = _resolve_hyperparams(task_type, {})
        finetune_dir = hp.get("finetune_dir", "")
        script = "bare_eval.py" if task_type == "prompt" else f"{task_type}_train.py"
        name = f"remote_script_{task_type}"
        if not (finetune_dir and settings.sandbox_host):
            return [{"name": name, "passed": True,
                     "detail": "no finetune_dir/sandbox_host configured — skipped"}]
        remote_path = f"{finetune_dir}/{script}"
        try:
            r = subprocess.run(
                ["ssh", settings.sandbox_host,
                 f"test -f {remote_path} && echo yes || echo no"],
                capture_output=True, text=True, timeout=30,
            )
        except Exception as exc:
            return [{"name": name, "passed": True,
                     "detail": f"host unreachable, not blocking: {exc}"}]
        if r.stdout.strip() == "yes":
            return [{"name": name, "passed": True, "detail": remote_path}]
        return [{"name": name, "passed": False,
                 "detail": f"{remote_path} not found on {settings.sandbox_host} — "
                           f"check finetune_dir points at the DEPLOYED directory, "
                           f"not a repo checkout (their layouts differ by one level)"}]

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
