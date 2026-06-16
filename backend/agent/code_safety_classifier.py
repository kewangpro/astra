"""
CodeSafetyClassifier — Phase 9.

LLM-based safety classifier for training scripts awaiting EXECUTE_CODE approval.
Returns a verdict (safe/unsafe) with a brief rationale so the HUD can
auto-approve low-risk scripts without blocking on human review.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

from backend.agent.inference.base import InferenceProvider, Message, GenerationConfig
from backend.logging_config import get_logger

logger = get_logger(__name__)

_SYSTEM = """\
You are ASTRA's Code Safety Classifier.
You are given a Python training script about to be executed in a sandboxed process.
Classify whether it is safe to auto-approve execution without explicit human review.

A script is SAFE if it:
- Only imports standard ML libraries (gymnasium, stable-baselines3, sklearn, torch, numpy, requests, logging, etc.)
- Only POSTs to localhost telemetry endpoints (127.0.0.1 / localhost)
- Only writes files inside the data/missions/ directory tree
- Has no shell execution (subprocess, os.system, eval, exec)
- Has no network calls to external hosts
- Has no file deletions or moves

A script is UNSAFE if it:
- Calls subprocess, os.system, os.popen, eval(), exec(), or __import__()
- Opens network connections to non-localhost hosts
- Reads, writes, or deletes files outside the project directory
- Contains obfuscated code or base64-decoded execution

Respond with JSON only:
{"safe": true, "reason": "one-sentence rationale"}
or
{"safe": false, "reason": "one-sentence explanation of the risk"}"""


@dataclass
class SafetyVerdict:
    safe: bool
    reason: str
    classifier: str = "llm"


class CodeSafetyClassifier:
    def __init__(self, provider: InferenceProvider) -> None:
        self._provider = provider

    async def classify(self, script: str) -> SafetyVerdict:
        """Classify a training script and return a SafetyVerdict."""
        # Fast-path static checks before calling LLM
        static = self._static_check(script)
        if not static.safe:
            return static

        messages = [
            Message(role="system", content=_SYSTEM),
            Message(role="user", content=f"Classify this script:\n\n{script[:6000]}"),
        ]
        try:
            raw = await self._provider.generate(
                messages, GenerationConfig(max_tokens=128, temperature=0.0)
            )
            raw = re.sub(r"```(?:json)?\s*|```", "", raw).strip()
            import json
            data = json.loads(raw)
            verdict = SafetyVerdict(
                safe=bool(data.get("safe", False)),
                reason=str(data.get("reason", "no reason given")),
            )
            logger.info("CodeSafetyClassifier: safe=%s reason=%s", verdict.safe, verdict.reason)
            return verdict
        except Exception as exc:
            logger.warning("CodeSafetyClassifier: LLM failed, defaulting to unsafe: %s", exc)
            return SafetyVerdict(safe=False, reason=f"Classification failed: {exc}")

    @staticmethod
    def _static_check(script: str) -> SafetyVerdict:
        """Deterministic pre-filter for obviously unsafe patterns."""
        danger_patterns = [
            (r"\bsubprocess\b", "uses subprocess"),
            (r"\bos\.system\b", "uses os.system"),
            (r"\bos\.popen\b", "uses os.popen"),
            (r"\beval\s*\(", "uses eval()"),
            (r"\bexec\s*\(", "uses exec()"),
            (r'__import__\s*\(', "uses __import__()"),
            (r'requests\.(get|post|put|delete|patch)\s*\(\s*["\']https?://(?!127\.0\.0\.1|localhost)',
             "makes external HTTP request"),
        ]
        for pattern, reason in danger_patterns:
            if re.search(pattern, script):
                return SafetyVerdict(safe=False, reason=f"Static check failed: {reason}", classifier="static")
        return SafetyVerdict(safe=True, reason="passed static checks", classifier="static")
