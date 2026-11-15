"""Top-level use-case plugin Protocol (product implements this)."""
from __future__ import annotations

import argparse
from typing import Any, Callable, Optional, Protocol, runtime_checkable


@runtime_checkable
class UseCasePlugin(Protocol):
    """Product adapter loaded by ``--use-case`` / :func:`astra_plugin.load_use_case`.

    Required
    --------
    * ``make_hooks(command)`` — attribute bag for platform ``run_*``
    * ``parse_args(command, rest)`` — learner CLI flags → ``Namespace``

    Optional
    --------
    * ``get_runner(command)`` — CE paths (distill/rft/star) until platform owns them
    * ``fill_defaults(command, args, rest)`` — fallback when parse_args is absent
    """

    def make_hooks(self, command: str) -> Any:
        """Return hooks object for ``command`` (``grpo``, ``dpo``, …)."""
        ...

    def parse_args(self, command: str, rest: list[str]) -> argparse.Namespace:
        """Parse learner argv ``rest`` for ``command``."""
        ...

    # Optional — not part of runtime_checkable method set if missing on impl;
    # loader/CLIs use getattr.
    # def get_runner(self, command: str) -> Callable[[Any], None]: ...
    # def fill_defaults(...): ...
