"""STaR harvest + platform ce_loop for demo_reasoning (exact-match oracle)."""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

from demo_reasoning.oracle import (
    build_prompt,
    load_cases,
    rationalize_hint,
    score_completion,
)


def _generate(model, tokenizer, prompt: str, *, max_tokens: int, temp: float) -> str:
    from mlx_lm import generate
    return generate(
        model, tokenizer, prompt=prompt, max_tokens=max_tokens, temp=temp, verbose=False
    )


def train(args) -> None:
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
    from mlx_lm import load as mlx_load

    from astratune.ce_loop import run_ce_lora_steps
    from astratune.distill_math import distill_masked_loss as _masked_sft_loss
    from astratune.lora_io import apply_lora, save_final_adapter, warm_start_lora

    cases = load_cases(getattr(args, "cases", None) or None)
    if getattr(args, "focus_cases", None):
        ids = {c.strip() for c in args.focus_cases.split(",") if c.strip()}
        cases = [c for c in cases if c.get("id") in ids]
        print(f"Focus scope: {len(cases)} case(s)")
    if not cases:
        print("ERROR: no cases loaded")
        sys.exit(1)

    print(f"Loading model: {args.model}")
    model, tokenizer = mlx_load(args.model)
    apply_lora(
        model,
        args.num_layers,
        rank=args.lora_rank,
        dropout=args.lora_dropout,
        scale=args.lora_scale,
    )
    if args.no_adapter:
        print("Cold start: NO warm-start adapter\n")
    else:
        loaded = warm_start_lora(model, args.adapter, strict=False, allow_missing=False)
        print(f"Warm-start: {loaded}")

    def report_eval(label: str = "Pass rate") -> tuple[float, list[str], None]:
        fails = []
        for case in cases:
            prompt = build_prompt(case)
            out = _generate(model, tokenizer, prompt, max_tokens=args.eval_max_tokens, temp=0.0)
            if score_completion(out, case) < 1.0:
                fails.append(case["id"])
        n_pass = len(cases) - len(fails)
        rate = n_pass / len(cases) if cases else 0.0
        print(f"{label}: {rate:.1%} ({n_pass}/{len(cases)})")
        return rate, fails, None

    model.eval()
    base_rate, base_fails, _ = report_eval("Baseline")
    if base_fails:
        print(f"Baseline failures ({len(base_fails)}): {base_fails}")
    print()

    print(f"STaR harvest K={args.k_samples} temp={args.temp} "
          f"rationalize_temp={args.rationalize_temp}…")
    t0 = time.time()
    completions: dict[str, str] = {}
    direct = rat = 0
    for i, case in enumerate(cases, 1):
        prompt = build_prompt(case)
        winners = []
        for _ in range(args.k_samples):
            cand = _generate(model, tokenizer, prompt, max_tokens=args.max_tokens, temp=args.temp)
            if score_completion(cand, case) >= 1.0:
                winners.append(cand)
        if winners:
            direct += 1
            completions[case["id"]] = winners[0]
        else:
            hint = rationalize_hint(case)
            rw = []
            for _ in range(max(2, args.k_samples // 4)):
                cand = _generate(
                    model, tokenizer, hint,
                    max_tokens=args.max_tokens + 64, temp=args.rationalize_temp,
                )
                if score_completion(cand, case) >= 1.0 and ("<think>" in cand or "</think>" in cand):
                    rw.append(cand)
            if rw:
                rat += 1
                completions[case["id"]] = rw[0]
        mx.clear_cache()
        if i % 2 == 0 or i == len(cases):
            print(f"  [{i}/{len(cases)}] harvested {len(completions)} "
                  f"(direct={direct}, rationalized={rat}) ({time.time()-t0:.0f}s)")

    if not completions:
        print("ERROR: No training examples harvested via STaR.")
        sys.exit(1)
    print(f"\nSTaR Harvesting Complete: {len(completions)}/{len(cases)}\n")

    tokenised: list[tuple[list[int], list[int]]] = []
    for case in cases:
        cid = case["id"]
        if cid not in completions:
            continue
        prompt = build_prompt(case)
        comp = completions[cid]
        p_ids = tokenizer.encode(prompt)
        c_ids = tokenizer.encode(comp)
        if args.max_seq_len and len(p_ids) + len(c_ids) > args.max_seq_len:
            continue
        if c_ids:
            tokenised.append((p_ids, c_ids))
    if not tokenised:
        print("ERROR: All examples empty after tokenisation.")
        sys.exit(1)
    print(f"{len(tokenised)} examples ready for training\n")

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    def _save(m, path: Path) -> None:
        save_final_adapter(
            m, path,
            model_id=args.model,
            num_layers=args.num_layers,
            rank=args.lora_rank,
            dropout=args.lora_dropout,
            scale=args.lora_scale,
        )

    _save(model, save_dir)
    _save(model, save_dir / "best")

    optimizer = optim.Adam(learning_rate=args.learning_rate)
    loss_and_grad_fn = nn.value_and_grad(model, _masked_sft_loss)
    total_steps = args.iters
    print(f"total_steps={total_steps}")
    print(f"Eval every {args.steps_per_eval} steps, save every {args.save_every} steps\n")
    t_train = time.time()

    best_rate, _ = run_ce_lora_steps(
        model,
        tokenised=tokenised,
        loss_and_grad_fn=loss_and_grad_fn,
        optimizer=optimizer,
        total_steps=total_steps,
        batch_size=args.batch_size,
        max_grad_norm=args.max_grad_norm,
        steps_per_eval=args.steps_per_eval,
        save_every=args.save_every,
        save_dir=save_dir,
        report_eval=report_eval,
        save_adapter=_save,
        initial_best_pass_rate=base_rate,
    )

    print("\n=== Final Eval ===")
    model.eval()
    pass_rate, failures, _ = report_eval("Pass rate")
    if failures:
        print(f"Failures: {failures}")
    if pass_rate >= best_rate or not (save_dir / "best").exists():
        best_rate = pass_rate
        _save(model, save_dir / "best")
        print(f"Best adapter ensured at {save_dir / 'best'}")
    print(f"Best pass rate during training: {best_rate:.1%}")
    print(f"Total time: {(time.time() - t_train) / 60:.1f} min")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="demo_reasoning STaR (exact-match CoT)")
    p.add_argument("--model", default="mlx-community/gemma-3-1b-it-4bit")
    p.add_argument("--adapter", default="")
    p.add_argument("--no-adapter", action="store_true")
    p.add_argument("--save-dir", required=True)
    p.add_argument("--cases", default="")
    p.add_argument("--focus-cases", default="")
    p.add_argument("--k-samples", type=int, default=4)
    p.add_argument("--temp", type=float, default=1.0)
    p.add_argument("--rationalize-temp", type=float, default=0.8)
    p.add_argument("--max-tokens", type=int, default=128)
    p.add_argument("--eval-max-tokens", type=int, default=128)
    p.add_argument("--num-layers", type=int, default=4)
    p.add_argument("--lora-rank", type=int, default=8)
    p.add_argument("--lora-scale", type=float, default=20.0)
    p.add_argument("--lora-dropout", type=float, default=0.0)
    p.add_argument("--iters", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--steps-per-eval", type=int, default=10)
    p.add_argument("--save-every", type=int, default=10)
    p.add_argument("--max-seq-len", type=int, default=1024)
    p.add_argument("--prompt-template", default="")
    p.add_argument("--reward-schema", default="v1")
    p.add_argument("--routing-only", action="store_true", default=False)
    p.add_argument("--val-split", type=float, default=0.0)
    args = p.parse_args()
    if not args.no_adapter and not args.adapter:
        p.error("one of --adapter or --no-adapter is required")
    if args.no_adapter and args.adapter:
        p.error("--adapter and --no-adapter are mutually exclusive")
    return args
