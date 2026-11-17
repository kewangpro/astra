"""Thin non-Ensemble STaR demo use-case (platform demo product).

Import as ``--use-case demo_reasoning``. Owns JSONL cases + exact-match oracle;
reuses AstraTune ``ce_loop`` via ``get_runner("star")``. Not Ensemble routing.
"""
from .usecase import fill_defaults, get_runner, make_hooks, parse_args

__all__ = ["make_hooks", "parse_args", "get_runner", "fill_defaults"]
