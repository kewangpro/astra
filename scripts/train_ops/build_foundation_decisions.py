#!/usr/bin/env python3
"""Build foundation decision JSONL (BoolQ Noul + ARC-Easy Choice + Likert Score).

Always keeps checked-in smoke + Likert files under
``finetune/data/decision/foundation/``. When ``datasets`` is available and
``--download`` is set, appends larger BoolQ / ARC-Easy splits.

Usage::

    .venv/bin/python finetune/build_foundation_decisions.py --download
    .venv/bin/python finetune/build_foundation_decisions.py   # smoke+likert only
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from core.decision_prompt import NOUL_OPTIONS, dump_decision_jsonl, normalize_decision_case  # noqa: E402

FOUNDATION = ROOT / "finetune" / "data" / "decision" / "foundation"


def _load_checked_in() -> list[dict]:
    rows: list[dict] = []
    for name in ("smoke_noul_choice.jsonl", "likert_score.jsonl"):
        path = FOUNDATION / name
        if not path.exists():
            continue
        for line in path.read_text().splitlines():
            text = line.strip()
            if not text:
                continue
            rows.append(normalize_decision_case(json.loads(text)))
    return rows


def _from_boolq(limit: int) -> list[dict]:
    from datasets import load_dataset

    ds = load_dataset("google/boolq", split="train")
    out: list[dict] = []
    for i, row in enumerate(ds):
        if len(out) >= limit:
            break
        passage = str(row.get("passage") or "").strip()
        question = str(row.get("question") or "").strip()
        if not passage or not question:
            continue
        label = "Yes" if bool(row.get("answer")) else "No"
        out.append(normalize_decision_case({
            "id": f"boolq_{i}",
            "primitive": "noul",
            "context": f"Passage: {passage}\nQuestion: {question}",
            "question": "Is the answer to the question yes?",
            "options": list(NOUL_OPTIONS),
            "label": label,
            "source": "boolq",
        }))
    return out


def _from_arc(limit: int) -> list[dict]:
    from datasets import load_dataset

    ds = load_dataset("allenai/ai2_arc", "ARC-Easy", split="train")
    out: list[dict] = []
    for i, row in enumerate(ds):
        if len(out) >= limit:
            break
        stem = str(row.get("question") or "").strip()
        choices = row.get("choices") or {}
        texts = list(choices.get("text") or [])
        labels = list(choices.get("label") or [])
        key = str(row.get("answerKey") or "").strip()
        if not stem or len(texts) < 2 or not key:
            continue
        # Map letter key → answer text (foundation uses texts, not A/B).
        label_text = None
        for lab, text in zip(labels, texts):
            if str(lab).strip() == key:
                label_text = str(text).strip()
                break
        if not label_text or label_text not in texts:
            continue
        options = [str(t).strip() for t in texts if str(t).strip()]
        if label_text not in options or len(set(options)) != len(options):
            continue
        out.append(normalize_decision_case({
            "id": f"arc_easy_{i}",
            "primitive": "choice",
            "context": stem,
            "question": "Which option is correct?",
            "options": options,
            "label": label_text,
            "source": "arc_easy",
        }))
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--download", action="store_true", help="Fetch BoolQ + ARC-Easy via datasets")
    p.add_argument("--boolq-limit", type=int, default=2000)
    p.add_argument("--arc-limit", type=int, default=2000)
    p.add_argument(
        "--out",
        default=str(FOUNDATION / "foundation_mix.jsonl"),
        help="Combined mix written for --cases",
    )
    args = p.parse_args(argv)

    rows = _load_checked_in()
    if args.download:
        try:
            rows.extend(_from_boolq(args.boolq_limit))
            rows.extend(_from_arc(args.arc_limit))
        except Exception as exc:
            print(f"download failed ({type(exc).__name__}: {exc}); keeping checked-in only")
    by_id = {r["id"]: r for r in rows}
    merged = list(by_id.values())
    n = dump_decision_jsonl(merged, args.out)
    by: dict[str, int] = {}
    for r in merged:
        by[r["primitive"]] = by.get(r["primitive"], 0) + 1
    print(f"Wrote {n} foundation cases → {args.out} {by}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
