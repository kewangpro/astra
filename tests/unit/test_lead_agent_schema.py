"""Unit tests for LeadAgent's plan schema — dpo/grpo/distill task_type support.

Without these in _PLAN_SCHEMA's enum, the LLM could never return those values,
and LoopStateMachine would silently overwrite a mission's task_type away from
"dpo"/"grpo"/"distill" to whatever the schema-constrained plan picked instead
(see LoopStateMachine's task_type reconciliation at iteration 0)."""
from __future__ import annotations

from backend.agent.lead_agent import _PLAN_SCHEMA, _PLANNING_SYSTEM


def test_plan_schema_task_type_enum_includes_finetune_types():
    enum = _PLAN_SCHEMA["properties"]["plan"]["properties"]["task_type"]["enum"]
    assert "dpo" in enum
    assert "grpo" in enum
    assert "distill" in enum


def test_plan_schema_task_type_enum_unchanged_for_existing_types():
    enum = _PLAN_SCHEMA["properties"]["plan"]["properties"]["task_type"]["enum"]
    assert set(enum) == {"rl", "sft", "ml", "mlx_lora", "dpo", "grpo", "distill"}


def test_planning_system_prompt_instructs_empty_hyperparameters_for_finetune():
    """The LLM must not guess adapter paths/num_layers for dpo/grpo/distill —
    those come entirely from the recipe and a wrong guess crashes the run."""
    assert "dpo/grpo/distill" in _PLANNING_SYSTEM
    assert 'leave "hyperparameters"' in _PLANNING_SYSTEM
    assert "empty object {}" in _PLANNING_SYSTEM
