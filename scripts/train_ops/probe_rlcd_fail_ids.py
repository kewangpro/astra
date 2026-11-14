#!/usr/bin/env python3
"""Probe RLCD Choice on a fixed fail-id list (live eval misses).

Prints skill / confidence / abstain for each id. Exit 0 if every case is
correct or abstained (thin fail-open); exit 2 if any high-conf wrong remains.

Usage::

    JEV_MLX_BASE_URL=http://mac-mini.local:8082/v1 \\
    .venv/bin/python finetune/probe_rlcd_fail_ids.py \\
      --ids-file backend/data/eval_progress/99b91b5a-fail-ids.txt
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from core.decision_model import (  # noqa: E402
    RLCD_CLOSED_SKILLS,
    filter_rlcd_options,
    min_margin,
    rlcd_choice,
)

DEFAULT_FAIL_IDS = [
    "route_llm_data_analysis",
    "route_shell_git_status",
    "route_shell_ls_directory",
    "route_shell_grep",
    "route_programming_run_code",
    "route_csv_data_services_with_file",
    "route_cost_chart_to_business_reporting",
    "route_secretary_send_confirmation",
    "route_data_query_dataset",
    "route_data_calculate_stats",
    "route_calendar_schedule_meeting",
    "route_calendar_check_availability",
    "route_calendar_move_event",
    "route_shell_tail_log",
    "route_shell_find_large_files",
]


def load_cases(ids: list[str]) -> list[dict]:
    raw = yaml.safe_load((ROOT / "backend" / "core" / "eval_cases.yaml").read_text())
    by_id = {c["id"]: c for c in raw if c.get("id")}
    missing = [i for i in ids if i not in by_id]
    if missing:
        raise SystemExit(f"unknown case ids: {missing}")
    return [by_id[i] for i in ids]


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", nargs="*", default=None)
    ap.add_argument("--ids-file", type=Path, default=None)
    ap.add_argument("--high-conf", type=float, default=0.4)
    args = ap.parse_args()
    if args.ids_file:
        ids = [ln.strip() for ln in args.ids_file.read_text().splitlines() if ln.strip() and not ln.startswith("#")]
    else:
        ids = args.ids or DEFAULT_FAIL_IDS

    if not os.environ.get("JEV_MLX_BASE_URL", "").strip():
        print("JEV_MLX_BASE_URL unset", file=sys.stderr)
        return 1

    cases = load_cases(ids)
    opts = {name: "" for name in sorted(RLCD_CLOSED_SKILLS)}
    margin = min_margin()
    print(f"probe n={len(cases)} margin={margin} high_conf>={args.high_conf}")

    high_wrong = 0
    thin_wrong = 0
    ok = 0
    abstain = 0
    rows = []
    for case in cases:
        cid = case["id"]
        query = case.get("query") or case.get("input") or ""
        expected = case.get("expected_skill") or ""
        # Product intent when expected_skill is empty / tool-grader only.
        if not expected and cid == "route_csv_data_services_with_file":
            expected = "Data Services"
        if not expected and cid == "route_cost_chart_to_business_reporting":
            expected = "Business Reporting"
        decision = await rlcd_choice(query, filter_rlcd_options(opts))
        if decision is None or decision.confidence < margin:
            status = "abstain"
            abstain += 1
            got = None
            conf = None if decision is None else decision.confidence
        else:
            got = decision.answer
            conf = decision.confidence
            if got == expected:
                status = "ok"
                ok += 1
            elif conf >= args.high_conf:
                status = "high_wrong"
                high_wrong += 1
            else:
                status = "thin_wrong"
                thin_wrong += 1
        rows.append({"id": cid, "expected": expected, "got": got, "conf": conf, "status": status})
        print(f"{status:10} conf={conf!s:>8} exp={expected!r:22} got={got!r:22} {cid}")

    print(json.dumps({"ok": ok, "abstain": abstain, "thin_wrong": thin_wrong, "high_wrong": high_wrong}, indent=2))
    return 2 if high_wrong else 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
