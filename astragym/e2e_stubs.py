"""Curated e2e stub cases for Agent GRPO (``gym_id=e2e_stubs``).

Shipped with the pip ``astragym`` package so Mini does not need Ensemble
``finetune/`` on ``sys.path`` for stub loading. Trajectory scoring lives in
``astragym.agent_rewards`` (re-exported by Ensemble ``trajectory_reward``).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Union


def load_e2e_stub_cases(path: Union[str, Path]) -> list[dict]:
    path = Path(path)
    if not path.is_file():
        return []
    text = path.read_text()
    if path.suffix in (".yaml", ".yml"):
        import yaml

        data = yaml.safe_load(text) or []
    else:
        data = json.loads(text)
    if isinstance(data, dict):
        data = data.get("cases") or data.get("eval_cases") or []
    return list(data)


DEFAULT_E2E_STUBS: list[dict[str, Any]] = [
    {
        "id": "e2e_stub_email_inbox",
        "query": "List unread emails in my inbox and summarize the latest one.",
        "expected_tools": ["query_database", "finish_task"],
        "require_finish": True,
        "tags": ["e2e", "stub", "email"],
    },
    {
        "id": "e2e_stub_calendar_week",
        "query": "Show my calendar events for this week.",
        "expected_tools": ["query_database", "finish_task"],
        "require_finish": True,
        "tags": ["e2e", "stub", "calendar"],
    },
    {
        "id": "e2e_stub_data_summary",
        "query": "Compute summary statistics for the latest sales table.",
        "expected_tools": ["query_database", "calculate", "finish_task"],
        "require_finish": True,
        "tags": ["e2e", "stub", "data"],
    },
    {
        "id": "e2e_stub_kb_refund",
        "query": "Look up the refund policy and compute an 80% refund on $150.",
        "expected_tools": ["search_kb", "calculate", "finish_task"],
        "require_finish": True,
        "tags": ["e2e", "stub", "kb"],
    },
    {
        "id": "e2e_stub_security_escalate",
        "query": "Search policy for unauthorized access and escalate to security.",
        "expected_tools": ["search_kb", "escalate_human", "finish_task"],
        "require_finish": True,
        "tags": ["e2e", "stub", "security"],
    },
    {
        "id": "e2e_stub_order_notify",
        "query": "Check order #8492 and email the customer if delayed.",
        "expected_tools": ["query_database", "send_email", "finish_task"],
        "require_finish": True,
        "tags": ["e2e", "stub", "order"],
    },
    {
        "id": "e2e_stub_clarify",
        "query": "Customer says check my recent item with no order id.",
        "expected_tools": ["clarify_query", "finish_task"],
        "require_finish": True,
        "tags": ["e2e", "stub", "clarify"],
    },
    {
        "id": "e2e_stub_full_triage",
        "query": "Look up return policy, query purchases, compute refund, email label.",
        "expected_tools": [
            "search_kb",
            "query_database",
            "calculate",
            "send_email",
            "finish_task",
        ],
        "require_finish": True,
        "tags": ["e2e", "stub", "triage"],
    },
]
