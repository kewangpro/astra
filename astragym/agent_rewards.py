"""Pluggable trajectory rewards for Agent GRPO (pip ``astragym``).

Stage 2 (default): AgentGym milestone / task_success return.
Stage 3: e2e stub graders on curated cases (no live network).

Stub case loading: ``astragym.e2e_stubs``. Product ``trajectory_reward``
re-exports this module for ``agent_grpo_train``.
"""
from __future__ import annotations

from typing import Callable


RewardFn = Callable[[dict, list[dict], list[dict], float, float], float]


def score_trajectory_agent_gym(
    case: dict,
    messages: list[dict],
    tool_log: list[dict],
    episode_return: float,
    task_success: float,
) -> float:
    """Dense env return with a terminal success bonus already in episode_return.

    Prefer binary task_success when available so pass-rate and reward align.
    """
    if task_success >= 1.0:
        return 1.0
    # Partial credit from shaped return (typical successful episode ~2.5–4.0)
    if episode_return <= 0:
        return max(-0.5, episode_return / 4.0)
    return min(0.9, episode_return / 5.0)


def _tool_names(tool_log: list[dict]) -> list[str]:
    names = []
    for t in tool_log:
        name = (t.get("tool") or t.get("name") or "").strip().lower()
        if name:
            names.append(name)
    return names


def score_trajectory_e2e(
    case: dict,
    messages: list[dict],
    tool_log: list[dict],
    episode_return: float = 0.0,
    task_success: float = 0.0,
) -> float:
    """Sparse outcome reward for curated stub e2e cases.

    Case schema (YAML/JSON):
      id, query, expected_tools: [str, ...], optional require_finish: bool,
      optional forbidden_tools: [str, ...], graders: optional keyword checks
      on the final assistant text.
    """
    tools = _tool_names(tool_log)
    expected = [str(t).lower() for t in (case.get("expected_tools") or [])]
    forbidden = {str(t).lower() for t in (case.get("forbidden_tools") or [])}

    if not tools:
        return -0.5

    if forbidden and any(t in forbidden for t in tools):
        return 0.0

    # Prefix match on expected tool sequence (ignore finish_task unless required)
    require_finish = bool(case.get("require_finish", True))
    cmp_tools = [t for t in tools if t != "finish_task"]
    exp_core = [t for t in expected if t != "finish_task"]
    if exp_core:
        if cmp_tools[: len(exp_core)] != exp_core:
            # partial: fraction of matching prefix
            matched = 0
            for a, b in zip(cmp_tools, exp_core):
                if a == b:
                    matched += 1
                else:
                    break
            return 0.3 * (matched / max(1, len(exp_core)))

    if require_finish and "finish_task" not in tools:
        return 0.3

    # Optional keyword graders on last assistant content
    for g in case.get("graders") or []:
        if g.get("type") != "contains":
            continue
        needle = str(g.get("text") or "").lower()
        hay = " ".join(
            str(m.get("content") or m.get("result") or "") for m in messages
        ).lower()
        if needle and needle not in hay:
            return 0.5

    return 1.0


def make_reward_fn(mode: str) -> RewardFn:
    mode = (mode or "agent_gym").lower()
    if mode in ("e2e", "ensemble_e2e"):
        return score_trajectory_e2e
    return score_trajectory_agent_gym


__all__ = [
    "RewardFn",
    "score_trajectory_agent_gym",
    "score_trajectory_e2e",
    "make_reward_fn",
]
