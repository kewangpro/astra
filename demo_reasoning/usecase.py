"""demo_reasoning UseCasePlugin — general CoT STaR scaffold (not Ensemble)."""
from __future__ import annotations

import argparse
from typing import Any, Callable


def make_hooks(command: str) -> Any:
    """STaR uses get_runner; no AstraRL hook bag required."""
    raise KeyError(f"demo_reasoning has no hooks for command {command!r} (use get_runner)")


def get_runner(command: str) -> Callable[[Any], None]:
    cmd = (command or "").lower()
    if cmd == "star":
        from demo_reasoning.star_runner import train
        return train
    raise KeyError(f"no runner for command {command!r}")


def parse_args(command: str, rest: list[str]) -> argparse.Namespace:
    cmd = (command or "").lower()
    if cmd != "star":
        raise KeyError(command)
    from demo_reasoning import star_runner
    import sys
    old = sys.argv
    try:
        sys.argv = ["star", *rest]
        return star_runner.parse_args()
    finally:
        sys.argv = old


def fill_defaults(command: str, args: argparse.Namespace, rest: list[str]) -> argparse.Namespace:
    try:
        return parse_args(command, rest)
    except Exception:
        return args
