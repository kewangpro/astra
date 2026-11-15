#!/usr/bin/env python3
"""A/B: letter Choice (Fast :8081) vs RLCD skill-string readout (:8082).

Reports skill-argmax accuracy and abstain rate on closed routing cases.
Not bare_eval JSON — matches IMPROVEMENTS §167 Step 5 eval order.

Usage (from product repo root, both hosts up)::

    JEV_MLX_BASE_URL=http://mac-mini.local:8082/v1 \\
    JEV_BASE_MLX_BASE_URL=http://mac-mini.local:8081/v1 \\
    .venv/bin/python finetune/ab_rlcd_choice.py
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from core.decision_model import (  # noqa: E402
    RLCD_CLOSED_SKILLS,
    choice,
    filter_rlcd_options,
    rlcd_choice,
)


def load_cases() -> list[dict]:
    raw = yaml.safe_load((ROOT / "backend/core/eval_cases.yaml").read_text())
    cases = []
    for case in raw:
        if case.get("mode") != "routing":
            continue
        skill = case.get("expected_skill") or ""
        if not skill or str(skill).startswith("mcp:"):
            continue
        if skill not in RLCD_CLOSED_SKILLS:
            continue
        query = case.get("query") or case.get("input") or ""
        if not query:
            continue
        cases.append(case)
    return cases


def options_for(case: dict) -> dict[str, str]:
    # Offer the full closed set (same as production filter after registry).
    return {name: "" for name in sorted(RLCD_CLOSED_SKILLS)}


async def run() -> int:
    cases = load_cases()
    if not cases:
        print("no routing cases")
        return 1
    letter_url = (
        os.environ.get("JEV_BASE_MLX_BASE_URL", "").strip()
        or os.environ.get("FAST_MLX_BASE_URL", "").strip()
    )
    rlcd_url = os.environ.get("JEV_MLX_BASE_URL", "").strip()
    print(f"cases={len(cases)} letter={letter_url or '(unset)'} rlcd={rlcd_url or '(unset)'}")

    from collections import Counter

    letter_ok = letter_abs = 0
    rlcd_ok = rlcd_abs = 0
    rlcd_miss_pairs: Counter[str] = Counter()
    rlcd_miss_rows: list[tuple[str, str, str, str]] = []
    for i, case in enumerate(cases, 1):
        query = case.get("query") or case.get("input") or ""
        expected = case["expected_skill"]
        opts = options_for(case)
        # Letter page is capped at 10 — use first 10 closed skills always
        # including expected when possible.
        letter_keys = sorted(RLCD_CLOSED_SKILLS)
        if expected in letter_keys:
            letter_keys.remove(expected)
            letter_keys = [expected] + letter_keys
        letter_opts = {k: "" for k in letter_keys[:10]}

        letter = await choice(
            query, "Which one skill should handle this query?", letter_opts
        )
        if letter is None:
            letter_abs += 1
        elif letter.answer == expected:
            letter_ok += 1

        rlcd = await rlcd_choice(query, filter_rlcd_options(opts))
        if rlcd is None:
            rlcd_abs += 1
            got = "(abstain)"
        elif rlcd.answer == expected:
            rlcd_ok += 1
            got = expected
        else:
            got = rlcd.answer
        if got != expected:
            cid = str(case.get("id") or i)
            rlcd_miss_pairs[f"{expected}→{got}"] += 1
            rlcd_miss_rows.append((cid, expected, got, (query or "")[:90]))

        if i % 10 == 0 or i == len(cases):
            print(f"… {i}/{len(cases)}")

    n = len(cases)
    def fmt(ok, abstain):
        decided = n - abstain
        acc = ok / decided if decided else 0.0
        return f"acc={acc:.1%} ({ok}/{decided}) abstain={abstain}/{n} ({abstain/n:.1%})"

    print("letter ", fmt(letter_ok, letter_abs))
    print("rlcd   ", fmt(rlcd_ok, rlcd_abs))
    if rlcd_miss_rows:
        print(f"rlcd misses ({len(rlcd_miss_rows)}):")
        for pair, count in rlcd_miss_pairs.most_common():
            print(f"  {count:3d}  {pair}")
        for cid, expected, got, q in rlcd_miss_rows[:20]:
            print(f"  {cid}: {expected} → {got} | {q!r}")
        if len(rlcd_miss_rows) > 20:
            print(f"  … {len(rlcd_miss_rows) - 20} more")
    # Gate: RLCD should beat or match letter on decided accuracy, or raise
    # decided coverage without collapsing accuracy.
    letter_decided = n - letter_abs
    rlcd_decided = n - rlcd_abs
    letter_acc = letter_ok / letter_decided if letter_decided else 0.0
    rlcd_acc = rlcd_ok / rlcd_decided if rlcd_decided else 0.0
    if rlcd_acc + 1e-9 >= letter_acc and rlcd_decided >= letter_decided * 0.9:
        print("GATE pass: RLCD >= letter accuracy with similar coverage")
        return 0
    if rlcd_ok >= letter_ok and rlcd_abs <= letter_abs:
        print("GATE pass: RLCD absolute corrects >= letter, abstain <= letter")
        return 0
    print("GATE fail: RLCD did not beat/match letter on this set")
    return 2


if __name__ == "__main__":
    raise SystemExit(asyncio.run(run()))
