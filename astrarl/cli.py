"""AstraRL training center CLI — ``python -m astrarl.cli <grpo|agrpo|opd>``.

Loads an Ensemble (or other) use-case plugin for hooks, then runs platform
``run_*`` loops. Mission facades ``os.execv`` into this module.
"""
from __future__ import annotations

import argparse
import os
import sys
from types import SimpleNamespace
from typing import Any, Callable


def _ensure_workspace_on_path(workspace: str | None) -> None:
    if not workspace:
        return
    ws = os.path.abspath(workspace)
    # Repo root (parent of finetune/) so ``import finetune`` works.
    parent = os.path.dirname(ws)
    for p in (parent, ws, os.path.join(ws, "finetune")):
        if os.path.isdir(p) and p not in sys.path:
            sys.path.insert(0, p)


def load_use_case(name: str) -> Any:
    """Import use-case plugin. Default ``ensemble`` → Ensemble ``finetune`` package."""
    name = (name or "ensemble").strip().lower()
    if name in ("ensemble", "finetune", "ensemble_usecase"):
        try:
            import finetune as plugin  # type: ignore
            if hasattr(plugin, "make_hooks"):
                return plugin
        except ImportError:
            pass
        # Mini-flat: ``finetune/`` directory itself is on ``sys.path``.
        import usecase as plugin  # type: ignore
        return plugin
    return __import__(name, fromlist=["*"])


def _ns_from_args(args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(**vars(args))


def cmd_grpo(args: argparse.Namespace) -> None:
    from astrarl.train_grpo_routing import run_grpo_routing_train

    plugin = load_use_case(args.use_case)
    hooks = plugin.make_hooks("grpo")
    run_grpo_routing_train(_ns_from_args(args), hooks)


def cmd_agrpo(args: argparse.Namespace) -> None:
    from astrarl.train_grpo_traj import run_grpo_traj_train

    plugin = load_use_case(args.use_case)
    hooks = plugin.make_hooks("agrpo")
    run_grpo_traj_train(_ns_from_args(args), hooks)


def cmd_opd(args: argparse.Namespace) -> None:
    from astrarl.train_opd import run_opd_train

    plugin = load_use_case(args.use_case)
    hooks = plugin.make_hooks("opd")
    run_opd_train(_ns_from_args(args), hooks)


_COMMANDS: dict[str, Callable[[argparse.Namespace], None]] = {
    "grpo": cmd_grpo,
    "agrpo": cmd_agrpo,
    "opd": cmd_opd,
}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m astrarl.cli", description="AstraRL train CLI")
    p.add_argument("command", choices=sorted(_COMMANDS), help="Learner subcommand")
    p.add_argument("--use-case", default="ensemble", help="Use-case plugin (default: ensemble)")
    p.add_argument(
        "--workspace",
        default=os.environ.get("ASTRA_WORKSPACE") or os.environ.get("ASTRA_FINETUNE_DIR") or "",
        help="Cwd / data workspace (alias: recipe finetune_dir)",
    )
    # Pass-through bag: unknown flags go to the learner via parse_known_args in main
    return p


def main(argv: list[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    # Split: known CLI flags vs learner flags
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("command", nargs="?", default=None)
    pre.add_argument("--use-case", default="ensemble")
    pre.add_argument("--workspace", default="")
    pre.add_argument("--finetune-dir", default="", dest="finetune_dir")
    known, rest = pre.parse_known_args(argv)

    if not known.command or known.command not in _COMMANDS:
        build_parser().print_help()
        sys.exit(2 if not known.command else 0)

    # Subcommand --help: parse via plugin only if importable; else show stub help.
    if any(a in ("-h", "--help") for a in rest):
        workspace = known.workspace or known.finetune_dir or ""
        if workspace:
            _ensure_workspace_on_path(workspace)
        try:
            plugin = load_use_case(known.use_case)
            parse = getattr(plugin, "parse_args", None)
            if callable(parse):
                parse(known.command, rest)  # argparse exits
        except Exception:
            print(f"usage: python -m astrarl.cli {known.command} [learner flags…]")
            print("(install Ensemble use-case on PYTHONPATH for full --help)")
            sys.exit(0)

    workspace = known.workspace or known.finetune_dir or ""
    if workspace:
        os.chdir(workspace)
        _ensure_workspace_on_path(workspace)

    # Re-parse with plugin-provided argparse if available
    plugin = load_use_case(known.use_case)
    parse = getattr(plugin, "parse_args", None)
    if callable(parse):
        args = parse(known.command, rest)
    else:
        # Minimal namespace: command + use_case + workspace + key=value from rest
        args = argparse.Namespace(
            command=known.command,
            use_case=known.use_case,
            workspace=workspace,
            finetune_dir=workspace,
        )
        # Attach raw rest as extra for hooks that rebuild argparse
        args._rest = rest  # noqa: SLF001
        # Best-effort: let plugin fill defaults
        fill = getattr(plugin, "fill_defaults", None)
        if callable(fill):
            args = fill(known.command, args, rest)

    args.use_case = known.use_case
    args.workspace = workspace
    if not getattr(args, "finetune_dir", None):
        args.finetune_dir = workspace

    _COMMANDS[known.command](args)


if __name__ == "__main__":
    main()
