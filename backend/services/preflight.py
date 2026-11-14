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
    "hstu": ["torch"],
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
        checks.append(self._check_remote_platform_imports(task_type))
        checks.append(self._check_remote_astra_pkg_root(task_type))

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
        """Confirm Astra train CLIs (or bare_eval for prompt) on the sandbox host.

        Missions execv ``python -m astrarl.cli`` / ``python -m astratune.cli`` —
        not Ensemble ``*_train.py``. Prompt missions still need workspace
        ``bare_eval.py``. Unreachable host is non-blocking.
        """
        from backend.sandbox.manager import _FINETUNE_REMOTE_TASK_TYPES
        if task_type not in _FINETUNE_REMOTE_TASK_TYPES:
            return []
        from backend.agent.code_generator import _resolve_hyperparams
        hp = _resolve_hyperparams(task_type, {})
        finetune_dir = hp.get("finetune_dir", "")
        python_bin = (
            (hp.get("python_bin") or "").strip()
            or (settings.sandbox_python or "").strip()
            or "/Users/kewang/finetune-env/bin/python"
        )

        # prompt: product scoring tool still lives under the workspace
        if task_type == "prompt":
            name = "remote_script_bare_eval"
            if not (finetune_dir and settings.sandbox_host):
                return [{"name": name, "passed": True,
                         "detail": "no finetune_dir/sandbox_host configured — skipped"}]
            remote_path = f"{finetune_dir}/bare_eval.py"
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
                     "detail": f"{remote_path} not found on {settings.sandbox_host}"}]

        # Train missions: importable Astra CLI modules + Ensemble use-case plugin
        name = "remote_train_cli"
        if not settings.sandbox_host:
            return [{"name": name, "passed": True,
                     "detail": "no sandbox_host configured — skipped"}]
        remote_cmd = (
            f'{python_bin} -c '
            f'"import astrarl.cli, astratune.cli, ensemble_usecase"'
        )
        if finetune_dir:
            ws_parent = finetune_dir.rsplit("/", 1)[0] if "/" in finetune_dir else ""
            if ws_parent:
                remote_cmd = (
                    f'PYTHONPATH="{ws_parent}:{finetune_dir}:$PYTHONPATH" {remote_cmd}'
                )
        try:
            r = subprocess.run(
                ["ssh", settings.sandbox_host, remote_cmd],
                capture_output=True, text=True, timeout=60,
            )
        except Exception as exc:
            return [{"name": name, "passed": True,
                     "detail": f"host unreachable, not blocking: {exc}"}]
        if r.returncode == 0:
            return [{"name": name, "passed": True,
                     "detail": f"astrarl.cli+astratune.cli+ensemble_usecase ok ({python_bin})"}]
        err = (r.stderr or r.stdout or "").strip() or f"exit {r.returncode}"
        return [{
            "name": name,
            "passed": False,
            "detail": (
                f"Astra train CLI / Ensemble plugin missing on {settings.sandbox_host}. "
                f"pip install -e ~/astra-pkg and put Ensemble repo on PYTHONPATH "
                f"(see finetune/VENDOR.md). Original: {err}"
            ),
        }]

    @staticmethod
    def _check_remote_platform_imports(task_type: str) -> dict:
        """SSH import check for astragym/astrarl/astratune in Mini finetune-env.

        Failures here become opaque ``ModuleNotFoundError`` mid-mission once the
        facade wrapper runs. Surface a VENDOR-style install hint at preflight
        instead. Unreachable host is non-blocking (same policy as remote_script).
        """
        from backend.sandbox.manager import _FINETUNE_REMOTE_TASK_TYPES
        name = "remote_platform_imports"
        if task_type not in _FINETUNE_REMOTE_TASK_TYPES:
            return {"name": name, "passed": True, "detail": "not a finetune-remote task — skipped"}
        if not settings.sandbox_host:
            return {"name": name, "passed": True,
                    "detail": "no sandbox_host configured — skipped"}

        from backend.agent.code_generator import _resolve_hyperparams
        hp = _resolve_hyperparams(task_type, {})
        python_bin = (
            (hp.get("python_bin") or "").strip()
            or (settings.sandbox_python or "").strip()
            or "/Users/kewang/finetune-env/bin/python"
        )
        remote_cmd = (
            f'{python_bin} -c '
            f'"import astragym, astrarl, astratune"'
        )
        try:
            r = subprocess.run(
                ["ssh", settings.sandbox_host, remote_cmd],
                capture_output=True, text=True, timeout=60,
            )
        except Exception as exc:
            return {"name": name, "passed": True,
                    "detail": f"host unreachable, not blocking: {exc}"}

        if r.returncode == 0:
            return {"name": name, "passed": True,
                    "detail": f"{python_bin}: astragym,astrarl,astratune ok"}

        err = (r.stderr or r.stdout or "").strip() or f"exit {r.returncode}"
        return {
            "name": name,
            "passed": False,
            "detail": (
                f"Astra platform packages missing on {settings.sandbox_host} "
                f"({python_bin}). On Mini run: "
                f"~/finetune-env/bin/pip install -e ~/astra-pkg "
                f"(see Ensemble finetune/VENDOR.md). Original error: {err}"
            ),
        }

    @staticmethod
    def _check_remote_astra_pkg_root(task_type: str) -> dict:
        """SSH ``test -d`` for the absolute astra_pkg_root baked into facades.

        Remote train.py inserts this path on ``sys.path`` before astratune/astrarl
        imports. A missing tree fails mid-mission with opaque ImportErrors after
        codegen already burned an iteration. Unreachable host is non-blocking
        (same policy as remote_script / remote_platform_imports).
        """
        from backend.sandbox.manager import _FINETUNE_REMOTE_TASK_TYPES
        name = "remote_astra_pkg_root"
        if task_type not in _FINETUNE_REMOTE_TASK_TYPES:
            return {"name": name, "passed": True, "detail": "not a finetune-remote task — skipped"}
        if not settings.sandbox_host:
            return {"name": name, "passed": True,
                    "detail": "no sandbox_host configured — skipped"}

        from backend.agent.code_generator import (
            _resolve_astra_pkg_root,
            _resolve_hyperparams,
        )
        hp = _resolve_hyperparams(task_type, {})
        root = _resolve_astra_pkg_root(hp)
        if not root:
            return {"name": name, "passed": True,
                    "detail": "no astra_pkg_root resolved — skipped"}

        try:
            r = subprocess.run(
                ["ssh", settings.sandbox_host,
                 f"test -d {root} && echo yes || echo no"],
                capture_output=True, text=True, timeout=30,
            )
        except Exception as exc:
            return {"name": name, "passed": True,
                    "detail": f"host unreachable, not blocking: {exc}"}

        if r.stdout.strip() == "yes":
            return {"name": name, "passed": True, "detail": root}

        return {
            "name": name,
            "passed": False,
            "detail": (
                f"{root} not found on {settings.sandbox_host} — "
                f"check recipe astra_pkg_root / ASTRA_PKG_ROOT, or rsync the "
                f"Astra tree to Mini (see Ensemble finetune/VENDOR.md)"
            ),
        }

    @staticmethod
    def _check_packages(task_type: str) -> list:
        from backend.sandbox.manager import _FINETUNE_REMOTE_TASK_TYPES
        if task_type in _FINETUNE_REMOTE_TASK_TYPES:
            return []
        results = []
        for pkg in _REQUIRED_PACKAGES.get(task_type, []):
            try:
                __import__(pkg)
                results.append({"name": f"import_{pkg}", "passed": True, "detail": pkg})
            except ImportError:
                results.append({"name": f"import_{pkg}", "passed": False,
                                 "detail": f"{pkg} not importable in current environment"})
        return results
