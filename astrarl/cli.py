"""AstraRL training center CLI — ``python -m astrarl.cli <grpo|agrpo|opd>``.

Loads a use-case plugin via :mod:`astra_plugin`, validates hooks, then runs
platform ``run_*`` loops. Mission facades ``os.execv`` into this module.
"""
from __future__ import annotations

import argparse
import os
import sys
from types import SimpleNamespace
from typing import Any, Callable

from astra_plugin import ensure_workspace_on_path, load_use_case, validate_hooks


def _ns_from_args(args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(**vars(args))


def cmd_grpo(args: argparse.Namespace) -> None:
    from astrarl.train_grpo_routing import run_grpo_routing_train

    plugin = load_use_case(args.use_case)
    hooks = validate_hooks("grpo", plugin.make_hooks("grpo"))
    run_grpo_routing_train(_ns_from_args(args), hooks)


def cmd_agrpo(args: argparse.Namespace) -> None:
    from astrarl.train_grpo_traj import run_grpo_traj_train

    plugin = load_use_case(args.use_case)
    hooks = validate_hooks("agrpo", plugin.make_hooks("agrpo"))
    run_grpo_traj_train(_ns_from_args(args), hooks)


def cmd_opd(args: argparse.Namespace) -> None:
    from astrarl.train_opd import run_opd_train

    plugin = load_use_case(args.use_case)
    hooks = validate_hooks("opd", plugin.make_hooks("opd"))
    run_opd_train(_ns_from_args(args), hooks)


_COMMANDS: dict[str, Callable[[argparse.Namespace], None]] = {
    "grpo": cmd_grpo,
    "agrpo": cmd_agrpo,
    "opd": cmd_opd,
}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m astrarl.cli", description="AstraRL train CLI")
    p.add_argument("command", choices=sorted(_COMMANDS), help="Learner subcommand")
    p.add_argument("--use-case", default="ensemble", help="Use-case plugin (default: ensemble → finetune)")
    p.add_argument(
        "--workspace",
        default=os.environ.get("ASTRA_WORKSPACE") or os.environ.get("ASTRA_FINETUNE_DIR") or "",
        help="Cwd / data workspace (alias: recipe finetune_dir)",
    )
    return p


def main(argv: list[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("command", nargs="?", default=None)
    pre.add_argument("--use-case", default="ensemble")
    pre.add_argument("--workspace", default="")
    pre.add_argument("--finetune-dir", default="", dest="finetune_dir")
    known, rest = pre.parse_known_args(argv)

    if not known.command or known.command not in _COMMANDS:
        build_parser().print_help()
        sys.exit(2 if not known.command else 0)

    if any(a in ("-h", "--help") for a in rest):
        workspace = known.workspace or known.finetune_dir or ""
        if workspace:
            ensure_workspace_on_path(workspace)
        try:
            plugin = load_use_case(known.use_case)
            parse = getattr(plugin, "parse_args", None)
            if callable(parse):
                parse(known.command, rest)
        except Exception:
            print(f"usage: python -m astrarl.cli {known.command} [learner flags…]")
            print("(install use-case on PYTHONPATH for full --help; see astra_plugin)")
            sys.exit(0)

    workspace = known.workspace or known.finetune_dir or ""
    if workspace:
        os.chdir(workspace)
        ensure_workspace_on_path(workspace)

    plugin = load_use_case(known.use_case)
    parse = getattr(plugin, "parse_args", None)
    if callable(parse):
        args = parse(known.command, rest)
    else:
        args = argparse.Namespace(
            command=known.command,
            use_case=known.use_case,
            workspace=workspace,
            finetune_dir=workspace,
            _rest=rest,
        )
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
