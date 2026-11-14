#!/usr/bin/env python3
"""Export Ensemble scenarios into generic §167 decision JSONL.

Writes Choice (routing), Noul (cascade / grounding-shaped), and Score
(ordered rubric) rows that train under ``rlcd_train.py --cases``.

Usage (from Ensemble repo root)::

    .venv/bin/python finetune/export_ensemble_decisions.py \\
        --out finetune/data/decision/ensemble/specialize.jsonl
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(ROOT / "finetune"))

from core.decision_model import ordered_rubric_levels  # noqa: E402
from core.decision_prompt import (  # noqa: E402
    NOUL_OPTIONS,
    dump_decision_jsonl,
    normalize_decision_case,
    routing_case_to_decision,
)
from rlcd_train import (  # noqa: E402
    _case_expected_skill,
    closed_skill_set,
    load_routing_cases,
)

EVAL_CASES = ROOT / "backend" / "core" / "eval_cases.yaml"

# Heuristic cascade labels: multi-intent / multi-? need the default planner.
_CASCADE_YES_HINTS = (
    " and ",
    " then ",
    " also ",
    " as well",
    "?",
)
_CASCADE_NO_HINTS = (
    "git status",
    "ls ",
    "what time",
    "read my email",
    "check calendar",
)


def export_choice(include_dynamic_mcp: bool) -> list[dict]:
    routing = load_routing_cases(include_dynamic_mcp=include_dynamic_mcp)
    skills = closed_skill_set(routing)
    out: list[dict] = []
    for case in routing:
        mapped = routing_case_to_decision(case, skills)
        if mapped is not None:
            out.append(mapped)
    return out


def _cascade_label(query: str) -> str | None:
    q = (query or "").lower()
    if not q:
        return None
    # Prefer explicit single-skill shortcuts as No (fast tier OK).
    for hint in _CASCADE_NO_HINTS:
        if hint in q and q.count("?") <= 1 and " and " not in q:
            return "No"
    multi_q = q.count("?") >= 2
    multi_and = " and " in q and any(
        w in q for w in ("email", "calendar", "stock", "chart", "report", "code")
    )
    if multi_q or multi_and:
        return "Yes"
    if any(h in q for h in _CASCADE_YES_HINTS) and len(q.split()) > 12:
        return "Yes"
    return "No"


def export_cascade_noul() -> list[dict]:
    raw = yaml.safe_load(EVAL_CASES.read_text()) or []
    out: list[dict] = []
    for case in raw:
        if case.get("mode") != "routing":
            continue
        query = str(case.get("query") or case.get("input") or "").strip()
        case_id = str(case.get("id") or "").strip()
        if not query or not case_id:
            continue
        label = _cascade_label(query)
        if label is None:
            continue
        out.append(normalize_decision_case({
            "id": f"noul_cascade_{case_id}",
            "primitive": "noul",
            "context": query,
            "question": (
                "Does this query need the stronger default planner "
                "instead of a fast single-skill plan?"
            ),
            "options": list(NOUL_OPTIONS),
            "label": label,
            "source": "ensemble_cascade",
        }))
    return out


def export_grounding_noul() -> list[dict]:
    """Synthetic supported / unsupported pairs for grounding Noul."""
    pairs = [
        (
            "ground_supported_price",
            "Tool output:\nAAPL last close 190.12\n\nClaim:\nAAPL closed at 190.12.",
            "Yes",
        ),
        (
            "ground_unsupported_price",
            "Tool output:\nAAPL last close 190.12\n\nClaim:\nAAPL closed at 250.00.",
            "No",
        ),
        (
            "ground_supported_email",
            "Tool output:\nFrom: bob@ex.com Subject: Invoice\n\nClaim:\nBob sent an email about an invoice.",
            "Yes",
        ),
        (
            "ground_unsupported_email",
            "Tool output:\nFrom: bob@ex.com Subject: Invoice\n\nClaim:\nAlice cancelled the meeting.",
            "No",
        ),
        (
            "ground_supported_flight",
            "Tool output:\nUA100 SFO→JFK departs 09:00\n\nClaim:\nThere is a morning UA100 from SFO to JFK.",
            "Yes",
        ),
        (
            "ground_unsupported_flight",
            "Tool output:\nUA100 SFO→JFK departs 09:00\n\nClaim:\nThe flight leaves at midnight for LAX.",
            "No",
        ),
    ]
    out = []
    for case_id, context, label in pairs:
        out.append(normalize_decision_case({
            "id": case_id,
            "primitive": "noul",
            "context": context,
            "question": "Is the claim supported by the tool output?",
            "options": list(NOUL_OPTIONS),
            "label": label,
            "source": "ensemble_grounding",
        }))
    return out


def export_score() -> list[dict]:
    raw = yaml.safe_load(EVAL_CASES.read_text()) or []
    out: list[dict] = []
    for case in raw:
        criteria = case.get("scoring_criteria") or case.get("acceptance_criteria") or ""
        if isinstance(criteria, list):
            criteria = "; ".join(str(x) for x in criteria)
        levels = ordered_rubric_levels(str(criteria))
        if not levels:
            continue
        case_id = str(case.get("id") or "").strip()
        query = str(case.get("query") or case.get("input") or "").strip()
        if not case_id:
            continue
        # Prefer an explicit gold level if present; else mid level as weak prior.
        gold = case.get("expected_level") or case.get("rubric_level")
        if isinstance(gold, int) and 1 <= gold <= len(levels):
            label = levels[gold - 1]
        elif isinstance(gold, str) and gold in levels:
            label = gold
        else:
            label = levels[len(levels) // 2]
        skill = _case_expected_skill(case) or case.get("skill") or "General Chat"
        context = (
            f"Task: {case.get('title') or case_id}\n"
            f"Skill: {skill}\n"
            f"Query: {query}\n"
            f"Output:\n{(case.get('reference') or case.get('mock_llm_response') or '(no sample output)')[:1500]}"
        )
        out.append(normalize_decision_case({
            "id": f"score_{case_id}",
            "primitive": "score",
            "context": context,
            "question": "Which rubric level fits this output?",
            "options": levels,
            "label": label,
            "source": "ensemble_score",
        }))
    # Always include a small hand-built rubric set so Score trains even if YAML
    # has few numbered rubrics.
    hand = [
        {
            "id": "score_hand_poor",
            "primitive": "score",
            "context": "Task: summarize\nSkill: Research & Search\nOutput:\nI don't know.",
            "question": "Which rubric level fits this output?",
            "options": ["poor", "ok", "good"],
            "label": "poor",
            "source": "ensemble_score",
        },
        {
            "id": "score_hand_ok",
            "primitive": "score",
            "context": (
                "Task: summarize\nSkill: Research & Search\n"
                "Output:\nThe article covers Q3 revenue growth briefly."
            ),
            "question": "Which rubric level fits this output?",
            "options": ["poor", "ok", "good"],
            "label": "ok",
            "source": "ensemble_score",
        },
        {
            "id": "score_hand_good",
            "primitive": "score",
            "context": (
                "Task: summarize\nSkill: Research & Search\n"
                "Output:\nQ3 revenue rose 12% YoY to $4.2B, citing the 10-Q filing."
            ),
            "question": "Which rubric level fits this output?",
            "options": ["poor", "ok", "good"],
            "label": "good",
            "source": "ensemble_score",
        },
    ]
    out.extend(normalize_decision_case(c) for c in hand)
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--out",
        default=str(ROOT / "finetune/data/decision/ensemble/specialize.jsonl"),
    )
    p.add_argument("--include-dynamic-mcp", action="store_true")
    p.add_argument("--choice-only", action="store_true")
    args = p.parse_args(argv)

    rows = export_choice(include_dynamic_mcp=args.include_dynamic_mcp)
    if not args.choice_only:
        rows.extend(export_cascade_noul())
        rows.extend(export_grounding_noul())
        rows.extend(export_score())

    n = dump_decision_jsonl(rows, args.out)
    by: dict[str, int] = {}
    for r in rows:
        by[r["primitive"]] = by.get(r["primitive"], 0) + 1
    print(f"Wrote {n} cases → {args.out} {by}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
