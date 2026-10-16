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


def metric_meets_threshold(
    value: float,
    threshold: float,
    *,
    lower_is_better: bool = False,
    metric_name: str = "",
) -> bool:
    """Whether ``value`` satisfies ``threshold``.

    ``pass_rate`` is a discrete fraction (n/54 on the model-routed set) that
    the loop displays to 3 decimals. ``50/54 = 0.925925…`` must meet a target
    written as ``0.926`` — otherwise a real hit never completes and later
    pivots regress (mission ``6f4204c7``).
    """
    v = float(value)
    t = float(threshold)
    if metric_name == "pass_rate":
        v = round(v, 3)
        t = round(t, 3)
    if lower_is_better:
        return v <= t
    return v >= t


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
        # Support comma-separated alternatives inside {}: e.g. "checkpoints/*.{zip,pth}"
        pattern_str = req.path_pattern
        if "{" in pattern_str and "," in pattern_str:
            prefix, rest = pattern_str.split("{", 1)
            alts, suffix = rest.split("}", 1)
            patterns = [os.path.join(mission_dir, prefix + a + suffix) for a in alts.split(",")]
        else:
            patterns = [os.path.join(mission_dir, pattern_str)]
        for pattern in patterns:
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

        name = req.metric_name or ""
        if req.operator == ">=":
            passed = metric_meets_threshold(
                value, req.threshold, lower_is_better=False, metric_name=name,
            )
        elif req.operator == "<=":
            passed = metric_meets_threshold(
                value, req.threshold, lower_is_better=True, metric_name=name,
            )
        elif req.operator == ">":
            passed = value > req.threshold
        elif req.operator == "<":
            passed = value < req.threshold
        else:
            passed = False
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
