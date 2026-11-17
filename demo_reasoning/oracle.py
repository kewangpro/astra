"""Exact-match CoT oracle for demo_reasoning (no MLX dependency)."""
from __future__ import annotations

import json
import re
from pathlib import Path

_CASES_DEFAULT = Path(__file__).resolve().parent / "cases.jsonl"
_ANSWER_RE = re.compile(r"(?:answer\s*[:=]\s*)?(-?\d+)\s*$", re.IGNORECASE | re.MULTILINE)


def load_cases(path: str | None = None) -> list[dict]:
    p = Path(path) if path else _CASES_DEFAULT
    rows = []
    with open(p) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def build_prompt(case: dict) -> str:
    return str(case.get("prompt") or case.get("query") or "").strip()


def _extract_answer(text: str) -> str:
    text = (text or "").strip()
    m = _ANSWER_RE.search(text)
    if m:
        return m.group(1)
    nums = re.findall(r"-?\d+", text)
    return nums[-1] if nums else ""


def score_completion(completion: str, case: dict, **_kw) -> float:
    got = _extract_answer(completion)
    want = str(case.get("answer") or "").strip()
    return 1.0 if got == want and want else 0.0


def rationalize_hint(case: dict) -> str:
    base = build_prompt(case)
    ans = str(case.get("answer") or "").strip()
    return (
        f"{base}\n\n[Expert Rationalization Guidance]\n"
        f"Verified answer: {ans}.\n"
        f"Step 1: Write concise reasoning inside <think>...</think>.\n"
        f"Step 2: End with the answer {ans} on its own line."
    )
