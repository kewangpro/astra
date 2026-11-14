#!/usr/bin/env python3
"""Rebuild decision_from_scratch_v1.jsonl (foundation + specialize + focus v8)."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent / "data" / "decision"
PARTS = [
    ROOT / "foundation" / "foundation_train.jsonl",
    ROOT / "ensemble" / "specialize.jsonl",
    ROOT / "ensemble" / "specialize_choice_focus_v8.jsonl",
]
OUT = ROOT / "ensemble" / "decision_from_scratch_v1.jsonl"


def main() -> None:
    by_id: dict[str, dict] = {}
    order: list[str] = []
    for path in PARTS:
        with path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                rid = row.get("id") or f"_anon_{len(order)}"
                if rid not in by_id:
                    order.append(rid)
                by_id[rid] = row  # later files win
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w") as f:
        for rid in order:
            f.write(json.dumps(by_id[rid], ensure_ascii=False) + "\n")
    print(f"wrote {OUT} ({len(order)} rows)")


if __name__ == "__main__":
    main()
