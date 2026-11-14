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
        """For finetune-remote task types, confirm the training script actually
        exists at {finetune_dir}/{script} on the sandbox host.

        astra dispatches a facade wrapper that chdirs into finetune_dir and
        invokes astratune/astrarl.make (or, for a few legacy paths, names that
        script path); if it is wrong the run dies immediately after the SSH
        round-trip, having already generated a script, launched a sandbox and
        burned an iteration — mission a8675c39 failed exactly this way. One
        `test -f` before dispatch turns a launch failure into a preflight
        warning that names the path it looked for.

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
        if task_type == "post-training":
            scripts = ["sft_train.py", "dpo_train.py", "grpo_train.py"]
        elif task_type == "prompt":
            scripts = ["bare_eval.py"]
        elif task_type == "agrpo":
            # Worker key is agrpo; Ensemble script is agent_grpo_train.py
            # (not agrpo_train.py — that false miss crashed nothing but
            # poisoned preflight + HUD for every Agent GRPO mission).
            scripts = ["agent_grpo_train.py"]
        else:
            scripts = [f"{task_type}_train.py"]

        results = []
        for script in scripts:
            name = f"remote_script_{script.removesuffix('.py')}"
            if not (finetune_dir and settings.sandbox_host):
                results.append({"name": name, "passed": True,
                                "detail": "no finetune_dir/sandbox_host configured — skipped"})
                continue
            remote_path = f"{finetune_dir}/{script}"
            try:
                r = subprocess.run(
                    ["ssh", settings.sandbox_host,
                     f"test -f {remote_path} && echo yes || echo no"],
                    capture_output=True, text=True, timeout=30,
                )
            except Exception as exc:
                results.append({"name": name, "passed": True,
                                "detail": f"host unreachable, not blocking: {exc}"})
                continue
            if r.stdout.strip() == "yes":
                results.append({"name": name, "passed": True, "detail": remote_path})
            else:
                results.append({"name": name, "passed": False,
                                "detail": f"{remote_path} not found on {settings.sandbox_host} — "
                                          f"check finetune_dir points at the DEPLOYED directory, "
                                          f"not a repo checkout (their layouts differ by one level)"})
        return results

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
