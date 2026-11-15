#!/usr/bin/env python3
"""A/B: letter Noul/Score/Choice vs RLCD generic readout (:8082).

Letter uses ``JEV_BASE_MLX_BASE_URL`` / ``FAST_MLX_BASE_URL`` (Fast :8081)
with ``RLCD_GENERIC_PROMPT`` off. RLCD uses ``JEV_MLX_BASE_URL`` (:8082) via
``rlcd_noul`` / ``rlcd_choice`` / ``rlcd_score`` (generic decision frame).

Usage (from product repo root)::

    JEV_MLX_BASE_URL=http://mac-mini.local:8082/v1 \\
    JEV_BASE_MLX_BASE_URL=http://mac-mini.local:8081/v1 \\
    .venv/bin/python finetune/ab_rlcd_decision.py
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from core.decision_model import (  # noqa: E402
    choice,
    noul,
    rlcd_choice,
    rlcd_noul,
    rlcd_score,
    score,
)
from core.decision_prompt import load_decision_jsonl  # noqa: E402

CASES = ROOT / "finetune/data/decision/foundation/foundation_mix.jsonl"


async def run() -> int:
    if not CASES.exists():
        print(f"missing {CASES}")
        return 1
    # Letter path must not divert into RLCD via noul()/score() wrappers.
    os.environ.pop("RLCD_PLANNER", None)

    cases = load_decision_jsonl(CASES)
    print(
        f"cases={len(cases)} "
        f"rlcd={os.environ.get('JEV_MLX_BASE_URL', '')!r} "
        f"letter={os.environ.get('JEV_BASE_MLX_BASE_URL', '')!r}"
    )
    by = {"noul": [0, 0, 0], "choice": [0, 0, 0], "score": [0, 0, 0]}
    for case in cases:
        prim = case["primitive"]
        opts = {o: "" for o in case["options"]}
        label = case["label"]
        ctx, q = case["context"], case["question"]
        # Letter half: generic prompt off so noul/score stay on letter tokens.
        os.environ.pop("RLCD_GENERIC_PROMPT", None)
        if prim == "noul":
            letter = await noul(ctx, q)
            os.environ["RLCD_GENERIC_PROMPT"] = "true"
            rlcd = await rlcd_noul(ctx, q)
            letter_ok = letter is not None and letter.answer.lower() == label.lower()
            rlcd_ok = rlcd is not None and rlcd.answer.lower() == label.lower()
        elif prim == "choice":
            letter_keys = list(opts.keys())[:10]
            letter = await choice(ctx, q, {k: "" for k in letter_keys})
            os.environ["RLCD_GENERIC_PROMPT"] = "true"
            rlcd = await rlcd_choice(ctx, opts)
            letter_ok = letter is not None and letter.answer == label
            rlcd_ok = rlcd is not None and rlcd.answer == label
        else:
            levels = case["options"]
            letter = await score(ctx, q, levels)
            os.environ["RLCD_GENERIC_PROMPT"] = "true"
            rlcd = await rlcd_score(ctx, q, levels)
            letter_ok = letter is not None and letter.answer == label
            rlcd_ok = rlcd is not None and rlcd.answer == label
        by[prim][2] += 1
        by[prim][0] += int(bool(letter_ok))
        by[prim][1] += int(bool(rlcd_ok))

    for prim, (letter_ok, rlcd_ok, n) in by.items():
        if n == 0:
            continue
        print(
            f"{prim}: letter={letter_ok}/{n} ({letter_ok / n:.1%}) "
            f"rlcd={rlcd_ok}/{n} ({rlcd_ok / n:.1%})"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(run()))
