"""Per-learner hook attribute contracts (derived from platform ``run_*``).

``@runtime_checkable`` Protocols only check *methods* at runtime; train loops
use attribute bags (``SimpleNamespace``). The frozensets below are the real
contract enforced by :func:`astra_plugin.validate.validate_hooks`.
"""
from __future__ import annotations

from typing import Any, Protocol

# ── Required attributes (must exist on hooks passed to run_*) ─────────────────

GRPO_HOOK_ATTRS: frozenset[str] = frozenset({
    "AUTO_CASES_PATH",
    "EMAIL_CASES",
    "GOLDEN_CASES_PATH",
    "MIN_CASES_FOR_HOLDOUT",
    "RoutingRewardPlugin",
    "apply_lora",
    "auto_focus_ids",
    "build_case_pool",
    "build_prompt_str",
    "case_pool_stats",
    "held_out_golden_routing",
    "is_dynamic_mcp_case",
    "load_training_cases",
    "mission_seed",
    "mlx_load",
    "pfd",
    "run_routing_eval",
    "save_checkpoint",
    "save_final_adapter",
    "score_completion",
    "tokenize_completion",
    "tokenize_prompt",
    "trainable_param_counts",
    "warm_start_lora",
    "write_adapter_config",
})

AGRPO_HOOK_ATTRS: frozenset[str] = frozenset({
    "DEFAULT_E2E_STUBS",
    "SCENARIOS",
    "eval_task_success",
    "load_e2e_stub_cases",
    "make_reward_fn",
    "mlx_load",
    "rollout_agent_gym",
    "rollout_e2e_stub",
    "save_checkpoint",
    "save_final_adapter",
    "write_adapter_config",
})

OPD_HOOK_ATTRS: frozenset[str] = frozenset({
    "EVAL_CASES_PATH",
    "MIN_CASES_FOR_HOLDOUT",
    "apply_lora",
    "assert_compatible_tokenizers",
    "build_prompt_str",
    "generate_k_completions",
    "is_dynamic_mcp_case",
    "is_pre_llm_case",
    "mission_seed",
    "mlx_load",
    "pfd",
    "run_routing_eval",
    "save_final_adapter",
    "score_completion",
    "split_cases",
    "tokenize_completion",
    "tokenize_prompt",
    "warm_start_lora",
    "write_adapter_config",
})

DPO_HOOK_ATTRS: frozenset[str] = frozenset({
    "DPOPair",
    "MIN_CASES_FOR_HOLDOUT",
    "cache_ref_log_probs",
    "case_pool_stats",
    "collect_pairs",
    "held_out_golden_routing",
    "is_dynamic_mcp_case",
    "load_training_cases",
    "mission_seed",
    "mlx_load",
    "pfd",
    "run_routing_eval",
    "save_checkpoint",
    "save_final_adapter",
    "tokenize_completion",
    "tokenize_prompt",
    "write_adapter_config",
})

RLCD_HOOK_ATTRS: frozenset[str] = frozenset({
    "ANSWER_MARKER",
    "apply_adapter_config",
    "load_train_pool",
    "mission_seed",
    "mlx_load",
    "pfd",
    "report_eval",
    "save_final_adapter",
    "snapshot_trainable",
    "split_cases",
    "train_stage",
    "write_adapter_config",
})

HOOK_ATTRS_BY_COMMAND: dict[str, frozenset[str]] = {
    "grpo": GRPO_HOOK_ATTRS,
    "agrpo": AGRPO_HOOK_ATTRS,
    "agent_grpo": AGRPO_HOOK_ATTRS,
    "traj": AGRPO_HOOK_ATTRS,
    "opd": OPD_HOOK_ATTRS,
    "dpo": DPO_HOOK_ATTRS,
    "rlcd": RLCD_HOOK_ATTRS,
}


# ── Typing Protocols (documentation / static checkers; attrs not runtime-checked) ─


class GrpoHooks(Protocol):
    """Hooks for ``astrarl.train_grpo_routing.run_grpo_routing_train``."""

    mlx_load: Any
    apply_lora: Any
    warm_start_lora: Any
    write_adapter_config: Any
    save_final_adapter: Any
    save_checkpoint: Any
    RoutingRewardPlugin: Any
    score_completion: Any
    build_prompt_str: Any
    load_training_cases: Any
    case_pool_stats: Any
    is_dynamic_mcp_case: Any
    held_out_golden_routing: Any
    mission_seed: Any
    build_case_pool: Any
    run_routing_eval: Any
    tokenize_prompt: Any
    tokenize_completion: Any
    trainable_param_counts: Any
    auto_focus_ids: Any
    pfd: Any
    EMAIL_CASES: Any
    GOLDEN_CASES_PATH: Any
    AUTO_CASES_PATH: Any
    MIN_CASES_FOR_HOLDOUT: Any


class AgrpoHooks(Protocol):
    """Hooks for ``astrarl.train_grpo_traj.run_grpo_traj_train``."""

    mlx_load: Any
    write_adapter_config: Any
    save_final_adapter: Any
    save_checkpoint: Any
    make_reward_fn: Any
    load_e2e_stub_cases: Any
    DEFAULT_E2E_STUBS: Any
    SCENARIOS: Any
    rollout_e2e_stub: Any
    rollout_agent_gym: Any
    eval_task_success: Any


class OpdHooks(Protocol):
    """Hooks for ``astrarl.train_opd.run_opd_train``."""

    mlx_load: Any
    apply_lora: Any
    warm_start_lora: Any
    write_adapter_config: Any
    save_final_adapter: Any
    assert_compatible_tokenizers: Any
    build_prompt_str: Any
    generate_k_completions: Any
    score_completion: Any
    run_routing_eval: Any
    tokenize_prompt: Any
    tokenize_completion: Any
    split_cases: Any
    mission_seed: Any
    is_dynamic_mcp_case: Any
    is_pre_llm_case: Any
    pfd: Any
    EVAL_CASES_PATH: Any
    MIN_CASES_FOR_HOLDOUT: Any


class DpoHooks(Protocol):
    """Hooks for ``astratune.train_dpo.run_dpo_train``."""

    mlx_load: Any
    pfd: Any
    DPOPair: Any
    collect_pairs: Any
    cache_ref_log_probs: Any
    load_training_cases: Any
    case_pool_stats: Any
    is_dynamic_mcp_case: Any
    held_out_golden_routing: Any
    mission_seed: Any
    run_routing_eval: Any
    tokenize_prompt: Any
    tokenize_completion: Any
    save_checkpoint: Any
    save_final_adapter: Any
    write_adapter_config: Any
    MIN_CASES_FOR_HOLDOUT: Any


class RlcdHooks(Protocol):
    """Hooks for ``astratune.train_rlcd.run_rlcd_train``."""

    mlx_load: Any
    pfd: Any
    apply_adapter_config: Any
    snapshot_trainable: Any
    load_train_pool: Any
    train_stage: Any
    report_eval: Any
    save_final_adapter: Any
    write_adapter_config: Any
    split_cases: Any
    mission_seed: Any
    ANSWER_MARKER: Any
