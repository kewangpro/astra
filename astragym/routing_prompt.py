"""Routing prompt gym — single-turn case + oracle score (GRPO / OPD).

Heavy scoring lives in Ensemble (`score_completion` / case pool). This module
exposes a stable ``gym_id`` and a light facade so AstraRL can depend on
AstraGym without importing Ensemble at package import time.
"""
from __future__ import annotations

from typing import Any, Callable, Optional


class RoutingPromptGym:
    gym_id = "routing_prompt"

    def __init__(
        self,
        *,
        cases_path: str = "",
        score_fn: Optional[Callable[..., float]] = None,
        **_: Any,
    ) -> None:
        self.cases_path = cases_path
        self._score_fn = score_fn
        self._cases: list[Any] = []
        self._idx = 0

    def load_cases(self) -> list[Any]:
        """Lazy-load cases via Ensemble helpers when available."""
        if self._cases:
            return self._cases
        if not self.cases_path:
            self._cases = []
            return self._cases
        try:
            from case_pool import load_cases  # Ensemble finetune/

            self._cases = list(load_cases(self.cases_path))
        except Exception:
            # Minimal JSONL fallback
            import json
            from pathlib import Path

            path = Path(self.cases_path)
            if path.is_file():
                rows = []
                for line in path.read_text().splitlines():
                    line = line.strip()
                    if line:
                        rows.append(json.loads(line))
                self._cases = rows
            else:
                self._cases = []
        return self._cases

    def reset(self, *, seed: Optional[int] = None) -> tuple[Any, dict]:
        cases = self.load_cases()
        if seed is not None and cases:
            self._idx = int(seed) % len(cases)
        elif cases:
            self._idx = 0
        case = cases[self._idx] if cases else None
        return case, {"index": self._idx, "n": len(cases)}

    def step(self, action: Any) -> tuple[Any, float, bool, bool, dict]:
        """``action`` is a completion string; returns oracle reward."""
        case, _ = self.reset()  # current case
        reward = 0.0
        info: dict[str, Any] = {}
        if self._score_fn is not None and case is not None:
            reward = float(self._score_fn(case, action))
            info["scored"] = True
        else:
            info["scored"] = False
            info["detail"] = "no score_fn bound — bind Ensemble oracle in learner"
        # advance
        cases = self.load_cases()
        if cases:
            self._idx = (self._idx + 1) % len(cases)
        return None, reward, True, False, info

    def close(self) -> None:
        return None

    @property
    def observation_space(self) -> Any:
        return None

    @property
    def action_space(self) -> Any:
        return None


def make_routing_prompt(**kwargs: Any) -> RoutingPromptGym:
    return RoutingPromptGym(**kwargs)
