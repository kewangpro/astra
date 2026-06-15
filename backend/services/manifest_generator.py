"""
ManifestGenerator — Step 7.2.

Generates a RequirementManifest from a mission's goal, task_type,
and target_metric using rule-based logic.

Rules applied (in order):
  1. stability   — training must complete without sandbox errors (always added)
  2. artifact    — checkpoint file must be saved to disk (always added)
  3. performance — one metric_threshold requirement per entry in target_metric
"""
from __future__ import annotations

from backend.models.manifest import Requirement, RequirementManifest
from backend.logging_config import get_logger

logger = get_logger(__name__)

# Checkpoint glob patterns per task type
_CHECKPOINT_PATTERNS: dict = {
    "ml":  "checkpoints/model.*",
    "rl":  "checkpoints/*.zip",
    "sft": "checkpoints/*/",
}

# For metrics where lower is better, flip the operator
_LOWER_IS_BETTER = {"eval_loss", "loss", "perplexity"}


def generate_manifest(
    mission_id: str,
    goal: str,
    task_type: str,
    target_metric: dict,
) -> RequirementManifest:
    """Build a RequirementManifest from mission metadata."""
    reqs: list[Requirement] = []
    req_idx = 1

    def _req_id() -> str:
        nonlocal req_idx
        rid = f"req_{req_idx:03d}"
        req_idx += 1
        return rid

    # ── Stability: sandbox must exit cleanly ──────────────────────────────────
    reqs.append(Requirement(
        id=_req_id(),
        description="Training script completes without runtime errors",
        category="stability",
        check_type="no_sandbox_error",
    ))

    # ── Artifact: checkpoint saved to disk ────────────────────────────────────
    pattern = _CHECKPOINT_PATTERNS.get(task_type, "checkpoints/*")
    reqs.append(Requirement(
        id=_req_id(),
        description=f"Model checkpoint saved to {pattern}",
        category="artifact",
        check_type="file_exists",
        path_pattern=pattern,
    ))

    # ── Performance: one requirement per target_metric entry ─────────────────
    for metric_name, threshold in target_metric.items():
        operator = "<=" if metric_name in _LOWER_IS_BETTER else ">="
        op_word = "at most" if operator == "<=" else "at least"
        reqs.append(Requirement(
            id=_req_id(),
            description=f"{metric_name} {op_word} {threshold} on validation set",
            category="performance",
            check_type="metric_threshold",
            metric_name=metric_name,
            threshold=float(threshold),
            operator=operator,
        ))

    manifest = RequirementManifest(mission_id=mission_id, requirements=reqs)
    logger.info(
        "ManifestGenerator: generated %d requirements for mission=%s",
        len(reqs), mission_id,
    )
    return manifest
