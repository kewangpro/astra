"""AstraTune training center CLI — ``python -m astratune.cli <sft|dpo|...>``.

Loads a use-case plugin via :mod:`astra_plugin`, validates hooks where
applicable, then runs platform ``run_*`` (or CE ``get_runner``).
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


def cmd_sft(args: argparse.Namespace) -> None:
    from astratune.train_sft import run_sft_train

    run_sft_train(args)


def cmd_dpo(args: argparse.Namespace) -> None:
    from astratune.train_dpo import run_dpo_train

    plugin = load_use_case(args.use_case)
    hooks = validate_hooks("dpo", plugin.make_hooks("dpo"))
    run_dpo_train(_ns_from_args(args), hooks)


def cmd_rlcd(args: argparse.Namespace) -> None:
    from astratune.train_rlcd import run_rlcd_train

    plugin = load_use_case(args.use_case)
    hooks = validate_hooks("rlcd", plugin.make_hooks("rlcd"))
    run_rlcd_train(_ns_from_args(args), hooks)


def cmd_distill(args: argparse.Namespace) -> None:
    plugin = load_use_case(args.use_case)
    runner = plugin.get_runner("distill")
    runner(args)


def cmd_rft(args: argparse.Namespace) -> None:
    plugin = load_use_case(args.use_case)
    runner = plugin.get_runner("rft")
    runner(args)


def cmd_star(args: argparse.Namespace) -> None:
    plugin = load_use_case(args.use_case)
    runner = plugin.get_runner("star")
    runner(args)


_COMMANDS: dict[str, Callable[[argparse.Namespace], None]] = {
    "sft": cmd_sft,
    "dpo": cmd_dpo,
    "rlcd": cmd_rlcd,
    "distill": cmd_distill,
    "rft": cmd_rft,
    "star": cmd_star,
}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m astratune.cli", description="AstraTune train CLI")
    p.add_argument("command", choices=sorted(_COMMANDS), help="Learner subcommand")
    p.add_argument("--use-case", default="ensemble", help="Use-case plugin (default: ensemble → finetune)")
    p.add_argument("--workspace", default="")
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
            print(f"usage: python -m astratune.cli {known.command} [learner flags…]")
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
