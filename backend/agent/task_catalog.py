"""Task is the objective. Algorithm is the update rule. task_type is the worker key.

Only Environment RL has more than one switchable algorithm on a single worker.
Every other row is one algorithm, one worker, fixed for the life of the mission.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Algorithm:
    name: str
    task_type: str


@dataclass(frozen=True)
class TaskGroup:
    task: str
    algorithms: tuple[Algorithm, ...]


TASK_GROUPS: tuple[TaskGroup, ...] = (
    TaskGroup(
        "Environment RL",
        (
            Algorithm("PPO", "rl"),
            Algorithm("DQN", "rl"),
            Algorithm("A2C", "rl"),
            Algorithm("SAC", "rl"),
            Algorithm("TD3", "rl"),
        ),
    ),
    TaskGroup(
        "Reinforcement fine-tuning",
        (
            Algorithm("GRPO", "grpo"),
            Algorithm("DPO", "dpo"),
        ),
    ),
    TaskGroup(
        "Supervised fine-tuning",
        (
            Algorithm("SFT", "sft"),
            Algorithm("SFT", "mlx_lora"),
            Algorithm("distillation", "distill"),
            Algorithm("rejection sampling", "rft"),
            Algorithm("STaR", "star"),
        ),
    ),
    TaskGroup(
        "On-policy distillation",
        (Algorithm("OPD", "opd"),),
    ),
    TaskGroup(
        "Calibrated decision",
        (Algorithm("RLCD", "rlcd"),),
    ),
    TaskGroup(
        "Prompt edit",
        (Algorithm("rule append", "prompt"),),
    ),
    TaskGroup(
        "Classical ML",
        (Algorithm("random forest", "ml"),),
    ),
    TaskGroup(
        "Retrieval",
        (
            Algorithm("embedding", "embed"),
            Algorithm("reranker", "rerank"),
        ),
    ),
    TaskGroup(
        "Generative recommendation",
        (Algorithm("HSTU", "hstu"),),
    ),
)


def task_type_enum() -> list[str]:
    """Worker keys, in catalog order, with duplicates removed."""
    seen: list[str] = []
    for group in TASK_GROUPS:
        for algo in group.algorithms:
            if algo.task_type not in seen:
                seen.append(algo.task_type)
    return seen


def format_task_pairs() -> str:
    """One line per task, for the planner prompt."""
    lines = []
    for group in TASK_GROUPS:
        types = {algo.task_type for algo in group.algorithms}
        if len(types) == 1 and len(group.algorithms) > 1:
            names = ", ".join(algo.name for algo in group.algorithms)
            lines.append(
                f'- {group.task}: {names} → task_type "{group.algorithms[0].task_type}"'
            )
            continue
        parts = [f'{algo.name} → "{algo.task_type}"' for algo in group.algorithms]
        lines.append(f"- {group.task}: " + "; ".join(parts))
    return "\n".join(lines)
