"""Platform StarHooks contract for ``task_type: star`` (STaR).

AstraTune owns the CE train loop (``ce_loop``) and stdout contract. Products
own harvest oracles via :class:`astra_plugin.UseCasePlugin.get_runner` (or a
future hooks bag). This module documents the **logical** seam so Ensemble
routing and non-Ensemble reasoners stay out of AstraTune core.

Required product responsibilities (implemented in the product runner today)
---------------------------------------------------------------------------
* ``load_cases(path|default) -> list[dict]`` — training/eval case list
* ``build_prompt(case) -> str`` — unhinted user prompt
* ``score_completion(completion, case, **kw) -> float`` — oracle (≥1.0 keeps)
* ``rationalize_hint(case) -> str`` — hinted prompt for failed direct rollouts
* ``format_training_completion(completion, case) -> str`` — optional; default
  keep verified trace as-is (often already contains ``<think>…</think>``)

Platform must **not** import product ``eval_cases.yaml`` or skill oracles.
"""
from __future__ import annotations

from typing import Any, Callable, List, Optional, Protocol, runtime_checkable

# Attribute names products should expose on a future StarHooks bag (mirrors
# GRPO_HOOK_ATTRS style). CE runners still use get_runner("star") today.
STAR_HOOK_ATTRS: frozenset = frozenset({
    "load_cases",
    "build_prompt",
    "score_completion",
    "rationalize_hint",
})


@runtime_checkable
class StarHooks(Protocol):
    """Optional typed view of product STaR harvest helpers."""

    def load_cases(self, path=None):
        # type: (Optional[str]) -> List[dict]
        ...

    def build_prompt(self, case):
        # type: (dict) -> str
        ...

    def score_completion(self, completion, case, **kwargs):
        # type: (str, dict, **Any) -> float
        ...

    def rationalize_hint(self, case):
        # type: (dict) -> str
        ...


# Callables used by thin demo runners (same names, free functions OK).
LoadCases = Callable[..., List[dict]]
BuildPrompt = Callable[[dict], str]
ScoreCompletion = Callable[..., float]
RationalizeHint = Callable[[dict], str]

__all__ = [
    "STAR_HOOK_ATTRS",
    "StarHooks",
    "LoadCases",
    "BuildPrompt",
    "ScoreCompletion",
    "RationalizeHint",
]
