"""RLCD training loop — platform-owned (astratune)."""
from __future__ import annotations
import json, random, sys, time
from pathlib import Path
from typing import Any
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import yaml
from mlx.utils import tree_flatten, tree_map
from astratune.rlcd_math import (
    fit_prompt, leave_one_out_advantages, pathwise_option_loss, proper_score,
    readout_logit_grads, score_function_loss, softmax, standard_brier,
)
from astratune.lora_io import apply_lora, trainable_param_counts, warm_start_lora

def run_rlcd_train(args, hooks) -> None:
    # conductor_min is only needed if something still imports routing_prompt;
    # decision cases use decision_prompt.build_decision_prompt exclusively.
    if args.prompt_template:
        tmpl = Path(args.prompt_template)
        if tmpl.exists():
            hooks.pfd.CONDUCTOR_PROMPT = tmpl
            hooks.pfd.ROUTING_SYSTEM_TEMPLATE = hooks.pfd._build_routing_system()
            print(f"Prompt template (legacy routing helpers): {args.prompt_template}")
        else:
            print(f"Prompt template not found (ok for --cases): {tmpl}")

    print(f"Loading model: {args.model}")
    model, tokenizer = hooks.mlx_load(args.model)

    if not args.no_adapter:
        adapter_path = Path(args.adapter)
        if not adapter_path.exists():
            raise FileNotFoundError(f"Adapter not found: {adapter_path}")
        hooks.apply_adapter_config(args, adapter_path)

    apply_lora(
        model,
        args.num_layers,
        rank=args.lora_rank,
        dropout=args.lora_dropout,
        scale=args.lora_scale,
    )
    print(f"LoRA: num_layers={args.num_layers} rank={args.lora_rank} scale={args.lora_scale}")

    if not args.no_adapter:
        loaded = warm_start_lora(model, args.adapter, strict=False, allow_missing=False)
        print(f"Warm-start from: {loaded}")
    else:
        print("Cold start (no adapter loaded)")

    n_trainable, total = trainable_param_counts(model)
    print(f"Trainable params: {n_trainable:,} / {total:,} ({100 * n_trainable / total:.3f}%)")

    pool = hooks.load_train_pool(args)
    by_prim: dict[str, int] = {}
    for c in pool:
        by_prim[c["primitive"]] = by_prim.get(c["primitive"], 0) + 1
    print(f"Readout pool: {len(pool)} cases {by_prim}, score={args.score}, marker={hooks.ANSWER_MARKER!r}")
    if getattr(args, "empty_rationale", False):
        print("Empty-rationale mode: train/eval match live POST /v1/rlcd/readout")

    focus_ids = {c.strip() for c in (args.focus_cases or "").split(",") if c.strip()}
    _held_in, held_out = hooks.split_cases(pool, seed=hooks.mission_seed(Path(args.save_dir)))
    held_ids = {c["id"] for c in held_out}
    # Focus misses stay in the train pool even if the holdout drew them.
    if focus_ids and held_out:
        keep_valid, moved = [], []
        for c in held_out:
            if c["id"] in focus_ids:
                moved.append(c["id"])
            else:
                keep_valid.append(c)
        held_out = keep_valid
        held_ids = {c["id"] for c in held_out}
        if moved:
            print(f"Focus: moved {sorted(moved)} from held-out into train")
    train_cases = [c for c in pool if c["id"] not in held_ids] or pool
    if held_out:
        print(f"Held out from the training pool: {len(held_out)} — {sorted(held_ids)}")
    del _held_in

    # Up-weight focus ids in the sampling pool (keep the rest of the catalog).
    if focus_ids:
        weight = max(1, int(args.focus_weight))
        focus_rows = [c for c in train_cases if c["id"] in focus_ids]
        missing = sorted(focus_ids - {c["id"] for c in focus_rows})
        if missing:
            print(f"Focus WARNING: not in decision pool: {missing}")
        if focus_rows and weight > 1:
            train_cases = list(train_cases) + focus_rows * (weight - 1)
        print(
            f"Focus scope: {len(focus_rows)} case(s) ×{weight} — "
            f"{sorted(c['id'] for c in focus_rows)}; train pool size={len(train_cases)}"
        )

    random.seed(hooks.mission_seed(Path(args.save_dir)))
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    hooks.write_adapter_config(save_dir, args)

    optimizer = optim.Adam(learning_rate=args.learning_rate)
    best = {"brier": float("inf")}

    # Eval on held-out (or a capped train sample) — full foundation pools are
    # thousands of cases; scoring every option string each step is prohibitive.
    eval_cases = held_out if held_out else train_cases
    eval_cap = int(getattr(args, "eval_max_cases", 0) or 0)
    if eval_cap > 0 and len(eval_cases) > eval_cap:
        eval_cases = eval_cases[:eval_cap]
        print(f"Eval capped to {len(eval_cases)} cases (--eval-max-cases {eval_cap})")
    else:
        print(f"Eval set: {len(eval_cases)} cases")

    print("\n--- Baseline readout (step 0) ---")
    model.eval()
    base_acc, base_brier = hooks.report_eval(
        model, tokenizer, eval_cases, args, "baseline",
    )
    print(f"Baseline readout accuracy {base_acc:.1%}, Brier {base_brier:.4f}")

    stages = [1, 2] if args.stage == "both" else [int(args.stage)]
    ref_snap = None
    for stage in stages:
        if stage == 2:
            ref_snap = hooks.snapshot_trainable(model)
            stage1_dir = save_dir / "stage1"
            hooks.save_final_adapter(model, stage1_dir, args)
            print(f"Stage-1 anchor saved → {stage1_dir}")
        hooks.train_stage(
            args, model, tokenizer, optimizer,
            train_cases, eval_cases, stage, ref_snap, best,
        )

    hooks.save_final_adapter(model, save_dir, args)
    print("\n=== Final readout ===")
    model.eval()
    hooks.report_eval(model, tokenizer, eval_cases, args, "final")
    if best["brier"] < float("inf"):
        print(f"Best Brier during training: {best['brier']:.4f}")
    print("Done.")


