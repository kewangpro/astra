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

Important clarifications — these are ALL SAFE, do NOT flag them:
- `del variable` is Python object deletion (freeing memory), NOT a file operation
- `requests.post(...)` or `requests.post(VAR, ...)` where VAR resolves to 127.0.0.1/localhost is SAFE telemetry
- Importing standard libraries (os, sys, json, logging, numpy, etc.) is SAFE
- `sys.path.insert(0, "/any/absolute/path")` adds a directory to Python's import search path — it does NOT read files and is SAFE
- `_sys.path.insert(...)` is identical to `sys.path.insert(...)` — SAFE
- Writing or reading files under the project directory (e.g. /Users/.../astra/data/missions/...) is SAFE
- Absolute paths in strings are NOT inherently unsafe — only flag actual reads/writes OUTSIDE the project tree

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
        # Fast-path static checks before calling LLM.
        # classifier="static" means a definitive verdict (safe or unsafe) — skip LLM.
        # classifier="static_ambiguous" means no danger found but not confirmed safe — proceed to LLM.
        static = self._static_check(script)
        if static.classifier != "static_ambiguous":
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
        # Handle both literal URLs and variable-based URLs (e.g. TELEMETRY_URL = "http://127.0.0.1:...").
        localhost_url_vars: set[str] = set()
        external_url_vars: set[str] = set()
        for var, url in re.findall(r'(\w+)\s*=\s*["\']https?://([^"\']+)["\']', script):
            if url.startswith(("127.0.0.1", "localhost")):
                localhost_url_vars.add(var)
            else:
                external_url_vars.add(var)

        literal_requests = re.findall(
            r'requests\.\w+\s*\(\s*["\']([^"\']+)["\']', script
        )
        var_requests = re.findall(
            r'requests\.\w+\s*\(\s*([A-Z_][A-Z0-9_]*)\b', script
        )

        # Fail fast if a request variable resolves to a known external URL
        for var in var_requests:
            if var in external_url_vars:
                return SafetyVerdict(
                    safe=False,
                    reason=f"Static check failed: request variable {var} resolves to non-localhost URL",
                    classifier="static",
                )

        has_any_request = bool(literal_requests or var_requests)
        literals_safe = all("127.0.0.1" in u or "localhost" in u for u in literal_requests)
        vars_safe = all(v in localhost_url_vars for v in var_requests)

        if has_any_request and literals_safe and vars_safe:
            return SafetyVerdict(
                safe=True,
                reason="all requests target localhost telemetry — auto-approved",
                classifier="static",
            )

        return SafetyVerdict(safe=True, reason="passed static checks", classifier="static_ambiguous")
