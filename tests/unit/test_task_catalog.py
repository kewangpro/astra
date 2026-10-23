"""The task/algorithm catalog is the worker list for the planner, schema, and docs."""
from __future__ import annotations

from pathlib import Path

from backend.agent.lead_agent import _PLANNING_SYSTEM, _PLAN_SCHEMA
from backend.agent.task_catalog import TASK_GROUPS, format_task_pairs, task_type_enum

_ROOT = Path(__file__).resolve().parents[2]


def test_catalog_matches_the_task_algorithm_table():
    assert [(g.task, [(a.name, a.task_type) for a in g.algorithms]) for g in TASK_GROUPS] == [
        ("Environment RL", [
            ("PPO", "rl"), ("DQN", "rl"), ("A2C", "rl"), ("SAC", "rl"), ("TD3", "rl"),
        ]),
        ("Reinforcement fine-tuning", [("GRPO", "grpo"), ("DPO", "dpo")]),
        ("Supervised fine-tuning", [
            ("SFT", "sft"), ("SFT", "mlx_lora"), ("distillation", "distill"),
            ("rejection sampling", "rft"), ("STaR", "star"),
        ]),
        ("On-policy distillation", [("OPD", "opd")]),
        ("Calibrated decision", [("RLCD", "rlcd")]),
        ("Prompt edit", [("rule append", "prompt")]),
        ("Classical ML", [("random forest", "ml")]),
        ("Retrieval", [("embedding", "embed"), ("reranker", "rerank")]),
    ]


def test_plan_schema_enum_is_the_catalog():
    enum = _PLAN_SCHEMA["properties"]["plan"]["properties"]["task_type"]["enum"]
    assert enum == task_type_enum()
    assert "star" in enum


def test_planning_prompt_states_each_task():
    rendered = format_task_pairs()
    assert rendered in _PLANNING_SYSTEM
    for group in TASK_GROUPS:
        assert group.task in _PLANNING_SYSTEM
    assert "Rejection sampling is supervised fine-tuning" in _PLANNING_SYSTEM
    assert "GRPO and DPO are reinforcement fine-tuning" in _PLANNING_SYSTEM


def test_docs_and_goal_bar_use_the_same_task_names():
    design = (_ROOT / "docs" / "DESIGN.md").read_text()
    readme = (_ROOT / "README.md").read_text()
    goal = (_ROOT / "frontend" / "src" / "components" / "command-center" / "GoalInput.tsx").read_text()
    docs_page = (_ROOT / "frontend" / "src" / "app" / "docs" / "page.tsx").read_text()
    for group in TASK_GROUPS:
        assert group.task in design
        assert group.task in readme
        assert group.task in docs_page
        assert f'label: "{group.task}"' in goal or f"label: '{group.task}'" in goal
        for algo in group.algorithms:
            assert f'value: "{algo.task_type}"' in goal
            assert f'taskType: "{algo.task_type}"' in docs_page
