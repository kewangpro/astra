"""
ManifestEvaluator — Step 7.2.

Iterates over a RequirementManifest and toggles each requirement's
passed flag based on the current evaluation evidence.
"""
from __future__ import annotations

import glob
import os
from datetime import datetime, timezone
from typing import Optional

from backend.models.manifest import Requirement, RequirementManifest
from backend.logging_config import get_logger

logger = get_logger(__name__)


class ManifestEvaluator:

    def evaluate(
        self,
        manifest: RequirementManifest,
        metrics: dict,
        mission_dir: str,
        sandbox_ok: bool,
    ) -> RequirementManifest:
        """
        Check each pending requirement and flip passed=True when satisfied.
        Already-passed requirements are not re-evaluated (pass is permanent).
        Returns the updated manifest.
        """
        for req in manifest.requirements:
            if req.passed:
                continue
            try:
                passed, evidence = self._check(req, metrics, mission_dir, sandbox_ok)
            except Exception as exc:
                logger.warning("ManifestEvaluator: error checking req=%s: %s", req.id, exc)
                continue

            if passed:
                req.passed = True
                req.passed_at = datetime.now(timezone.utc).isoformat()
                req.evidence = evidence
                logger.info(
                    "ManifestEvaluator: req=%s PASSED (%s) mission=%s",
                    req.id, evidence, manifest.mission_id,
                )

        summary = manifest.summary()
        logger.info(
            "ManifestEvaluator: mission=%s manifest %d/%d passed complete=%s",
            manifest.mission_id, summary["passed"], summary["total"], summary["complete"],
        )
        return manifest

    # ── per-check-type dispatch ───────────────────────────────────────────────

    def _check(
        self,
        req: Requirement,
        metrics: dict,
        mission_dir: str,
        sandbox_ok: bool,
    ) -> tuple[bool, Optional[str]]:
        if req.check_type == "no_sandbox_error":
            return self._check_no_error(sandbox_ok)
        if req.check_type == "file_exists":
            return self._check_file_exists(req, mission_dir)
        if req.check_type == "metric_threshold":
            return self._check_metric(req, metrics)
        logger.warning("ManifestEvaluator: unknown check_type=%s", req.check_type)
        return False, None

    @staticmethod
    def _check_no_error(sandbox_ok: bool) -> tuple[bool, Optional[str]]:
        return sandbox_ok, "sandbox exited cleanly" if sandbox_ok else None

    @staticmethod
    def _check_file_exists(req: Requirement, mission_dir: str) -> tuple[bool, Optional[str]]:
        if not req.path_pattern:
            return False, None
        pattern = os.path.join(mission_dir, req.path_pattern)
        matches = glob.glob(pattern)
        if matches:
            return True, f"found {os.path.relpath(matches[0], mission_dir)}"
        return False, None

    @staticmethod
    def _check_metric(req: Requirement, metrics: dict) -> tuple[bool, Optional[str]]:
        if req.metric_name is None or req.threshold is None:
            return False, None

        value = ManifestEvaluator._resolve(req.metric_name, metrics)
        if value is None:
            return False, None

        ops = {
            ">=": value >= req.threshold,
            "<=": value <= req.threshold,
            ">":  value >  req.threshold,
            "<":  value <  req.threshold,
        }
        passed = ops.get(req.operator, False)
        evidence = f"{req.metric_name}={value:.4f} (threshold {req.operator} {req.threshold})"
        return passed, evidence if passed else None

    @staticmethod
    def _resolve(name: str, metrics: dict) -> Optional[float]:
        """Look up metric with suffix-match fallback (e.g. 'accuracy' matches 'validation_accuracy')."""
        if name in metrics:
            return metrics[name]
        for key, val in metrics.items():
            if key.endswith(f"_{name}") or key.startswith(f"{name}_"):
                return val
        return None
