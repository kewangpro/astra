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
- Reads, writes, or deletes FILES outside the project directory (os.remove, shutil.rmtree, etc.)
- Contains obfuscated code or base64-decoded execution

Important clarifications:
- `del variable` is Python object deletion (freeing memory), NOT a file operation — it is SAFE
- `requests.post(...)` to 127.0.0.1 or localhost is SAFE telemetry, not an external network call
- Importing standard libraries (os, sys, json, logging, numpy, etc.) is SAFE
- `sys.path.insert(0, "/some/project/path")` is a Python import path modification, NOT a file operation — it is SAFE
- Writing files to absolute paths inside the project directory (e.g. /Users/.../astra/data/missions/...) is SAFE

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
            Message(role="user", content=f"Classify this script:\n\n{script[:2000]}"),
        ]
        try:
            raw = await self._provider.generate(
                messages, GenerationConfig(max_tokens=64, temperature=0.0)
            )
            import json
            raw = re.sub(r"```(?:json)?\s*|```", "", raw).strip()
            # Extract first JSON object in case the LLM appends extra text
            m = re.search(r"\{[^}]*\}", raw, re.DOTALL)
            raw = m.group(0) if m else raw
            data = json.loads(raw)
            verdict = SafetyVerdict(
                safe=bool(data.get("safe", False)),
                reason=str(data.get("reason", "no reason given")),
            )
            logger.info("CodeSafetyClassifier: safe=%s reason=%s", verdict.safe, verdict.reason)
            return verdict
        except Exception as exc:
            logger.warning("CodeSafetyClassifier: LLM failed: %s", exc)
            return SafetyVerdict(safe=False, reason="LLM classifier unavailable — manual review required", classifier="static")

    @staticmethod
    def _static_check(script: str) -> SafetyVerdict:
        """Deterministic pre-filter: fail fast on danger patterns, short-circuit safe on known-good scripts."""
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

        # Positive short-circuit: if every requests call in the script targets localhost,
        # this is a standard ASTRA training script — skip the LLM classifier entirely.
        all_requests = re.findall(
            r'requests\.(get|post|put|delete|patch)\s*\(\s*["\']([^"\']+)["\']',
            script,
        )
        if all_requests and all(
            "127.0.0.1" in url or "localhost" in url for _, url in all_requests
        ):
            return SafetyVerdict(
                safe=True,
                reason="all requests target localhost telemetry — auto-approved",
                classifier="static",
            )

        return SafetyVerdict(safe=True, reason="passed static checks", classifier="static")
