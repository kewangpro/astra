"""RLCD strictly proper scores and readout gradients (platform-owned)."""
from __future__ import annotations

import math


def proper_score(u: list[float], y: int, score: str = "brier") -> float:
    """Strictly proper score, higher is better. Brier J = 2 u_y − ||u||²."""
    if score == "log":
        return math.log(max(u[y], 1e-12))
    return 2.0 * u[y] - sum(p * p for p in u)


def standard_brier(u: list[float], y: int) -> float:
    """||u − e_y||². Lower is better. Equals 1 − J for the Brier score above."""
    return sum((p - (1.0 if i == y else 0.0)) ** 2 for i, p in enumerate(u))


def readout_logit_grads(u: list[float], y: int, score: str = "brier") -> list[float]:
    """∂J/∂z for u = softmax(z). Sums to 0 (softmax is shift-invariant).

    Pathwise stop-grad coefficient for per-option readout backprop (see
    ``rlcd_train`` stage-1 loop).
    """
    g = [0.0] * len(u)
    if score == "log":
        g[y] = 1.0 / max(u[y], 1e-12)
    else:
        for i, p in enumerate(u):
            g[i] = -2.0 * p
        g[y] += 2.0
    dot = sum(gi * pi for gi, pi in zip(g, u))
    return [pi * (gi - dot) for gi, pi in zip(g, u)]