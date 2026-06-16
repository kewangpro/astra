"""
SpecialistEvaluator — Step 3.4.

Independent evaluation agent that runs BenchmarkSuite + StressTester
and aggregates results into a final verdict for the Loop State Machine.
This runs after each training iteration and is the mandatory gatekeeper
for the Eval phase.
"""
from __future__ import annotations

import os
from typing import Optional

from backend.evaluator.benchmark import BenchmarkSuite
from backend.evaluator.stress_tester import StressTester
from backend.config import settings
from backend.logging_config import get_logger

logger = get_logger(__name__)


class SpecialistEvaluator:
    def __init__(self) -> None:
        pass

    async def evaluate(self, mission_id: str, plan: dict) -> dict:
        """
        Run the full evaluation suite for a completed training iteration.
        Returns a dict with metrics, benchmark results, and stress test results.
        """
        task_type = plan.get("task_type", "rl")
        domain = plan.get("domain", task_type)
        checkpoint_dir = os.path.join(settings.data_path, "missions", mission_id, "checkpoints")

        # Find the latest checkpoint
        checkpoint_path = self._latest_checkpoint(checkpoint_dir)
        if not checkpoint_path:
            logger.warning("SpecialistEvaluator: no checkpoint found for mission=%s", mission_id)
            return {"metrics": {}, "benchmark": {}, "stress": {}, "verdict": "no_checkpoint"}

        logger.info("SpecialistEvaluator: evaluating mission=%s checkpoint=%s", mission_id, checkpoint_path)

        # Run Golden Set benchmark
        benchmark = BenchmarkSuite(domain)
        benchmark_results = benchmark.run(checkpoint_path)

        # Run stress tests
        stress = StressTester(task_type)
        stress_results = stress.run(checkpoint_path)

        # Extract primary metrics from benchmark results
        metrics: dict = {}
        for r in benchmark_results.get("results", []):
            metrics.update(r.get("metrics", {}))

        verdict = "pass" if benchmark_results.get("failed", 1) == 0 else "fail"
        logger.info(
            "SpecialistEvaluator: mission=%s verdict=%s (passed=%d failed=%d)",
            mission_id, verdict,
            benchmark_results.get("passed", 0),
            benchmark_results.get("failed", 0),
        )

        return {
            "metrics": metrics,
            "benchmark": benchmark_results,
            "stress": stress_results,
            "verdict": verdict,
            "checkpoint_path": checkpoint_path,
        }

    @staticmethod
    def _latest_checkpoint(checkpoint_dir: str) -> Optional[str]:
        if not os.path.isdir(checkpoint_dir):
            return None
        # Prefer best_model.zip (saved at peak reward) over last_model.zip (final, often degraded)
        best = os.path.join(checkpoint_dir, "best_model.zip")
        if os.path.isfile(best):
            return best
        entries = [
            os.path.join(checkpoint_dir, f)
            for f in os.listdir(checkpoint_dir)
            if not f.startswith(".")
        ]
        if not entries:
            return None
        return max(entries, key=os.path.getmtime)
