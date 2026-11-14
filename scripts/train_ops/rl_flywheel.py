"""Online-RL data flywheel status and train-command helper.

Closes the gap between §RSI3 ``auto_eval_cases.yaml`` (production misses) and
Method E GRPO sampling. Does **not** auto-train or swap production adapters —
those stay manual / suite-gated.

Usage:
  .venv/bin/python finetune/rl_flywheel.py status
  .venv/bin/python finetune/rl_flywheel.py focus-ids
  .venv/bin/python finetune/rl_flywheel.py train-cmd
  .venv/bin/python finetune/rl_flywheel.py train-cmd --save-dir finetune/adapters/grpo_flywheel_1
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from case_pool import (  # noqa: E402
    AUTO_CASES_PATH,
    GOLDEN_CASES_PATH,
    auto_focus_ids,
    case_pool_stats,
    is_dynamic_mcp_skill,
    load_training_cases,
)


def cmd_status(_: argparse.Namespace) -> int:
    cases = load_training_cases(include_auto=True)
    stats = case_pool_stats(cases, is_dynamic_mcp=is_dynamic_mcp_skill)
    focus = sorted(auto_focus_ids(cases, is_dynamic_mcp=is_dynamic_mcp_skill))

    print("Online RL data flywheel")
    print(f"  golden:     {GOLDEN_CASES_PATH}  ({stats['golden']} cases)")
    print(f"  auto:       {AUTO_CASES_PATH}  ({stats['auto']} cases"
          f"{'' if AUTO_CASES_PATH.is_file() else ' — missing'})")
    print(f"  merged:     {stats['total']}  (golden wins on id/query)")
    print(f"  routing+skill: {stats['routing_with_skill']}")
    print(f"  auto trainable (non-dynamic-MCP): {len(stats['auto_trainable_ids'])}")
    print(f"  auto skipped dynamic MCP:        {stats['auto_skipped_dynamic_mcp']}")
    if focus:
        print("  focus seeds:")
        for cid in focus:
            print(f"    - {cid}")
    else:
        print("  focus seeds: (none — no gradeable auto routing cases)")
    print()
    print("Loop: production miss → auto_eval_cases → grpo_train (default includes auto)")
    print("      → suite gate → manual adapter promote / serve")
    print("Escape hatch: grpo_train.py --no-auto-cases")
    return 0


def cmd_focus_ids(args: argparse.Namespace) -> int:
    cases = load_training_cases(include_auto=True)
    focus = sorted(auto_focus_ids(cases, is_dynamic_mcp=is_dynamic_mcp_skill))
    if args.json:
        print(json.dumps({"focus_ids": focus}, indent=2))
    else:
        for cid in focus:
            print(cid)
    if args.out:
        path = Path(args.out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(focus) + ("\n" if focus else ""), encoding="utf-8")
        print(f"# wrote {len(focus)} id(s) → {path}", file=sys.stderr)
    return 0


def cmd_train_cmd(args: argparse.Namespace) -> int:
    cases = load_training_cases(include_auto=True)
    stats = case_pool_stats(cases, is_dynamic_mcp=is_dynamic_mcp_skill)
    n_auto = len(stats["auto_trainable_ids"])
    save_dir = args.save_dir
    adapter = args.adapter
    lines = [
        "# Online-RL flywheel GRPO command (auto cases included by default)",
        f"# auto trainable cases in pool: {n_auto}",
        "~/finetune-env/bin/python finetune/grpo_train.py \\",
        f"  --adapter {adapter} \\",
        "  --prompt-template backend/prompts/conductor_min.md \\",
        f"  --save-dir {save_dir} \\",
        "  --routing-only \\",
        "  --reward-schema v2 \\",
        "  --num-generations 4 \\",
        "  --iters 100",
        "",
        "# Golden-only (repro historical champion runs):",
        "#   ... same flags ... --no-auto-cases",
        "# After suite gate: point serve_adapter / Makefile mlx at the new best/",
    ]
    print("\n".join(lines))
    return 0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Online RL data flywheel helpers")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("status", help="Show golden/auto merge and trainable focus seeds")

    p_focus = sub.add_parser("focus-ids", help="List auto case ids eligible for GRPO focus")
    p_focus.add_argument("--json", action="store_true")
    p_focus.add_argument("--out", default="", help="Optional path to write one id per line")

    p_cmd = sub.add_parser("train-cmd", help="Print a recommended grpo_train invocation")
    p_cmd.add_argument(
        "--adapter",
        default="finetune/adapters/grpo_v9_min/best",
        help="Warm-start adapter path",
    )
    p_cmd.add_argument(
        "--save-dir",
        default="finetune/adapters/grpo_flywheel_1",
        help="Output adapter directory",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if args.cmd == "status":
        return cmd_status(args)
    if args.cmd == "focus-ids":
        return cmd_focus_ids(args)
    if args.cmd == "train-cmd":
        return cmd_train_cmd(args)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
