"""DPO training loop — platform-owned (astratune)."""
from __future__ import annotations
import json, random, sys, time
from pathlib import Path
from typing import Any
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import yaml
from mlx.utils import tree_flatten, tree_map
from astratune.dpo_math import dpo_loss
from astratune.lora_io import apply_lora, warm_start_lora

def run_dpo_train(args, hooks) -> None:
    # Apply prompt template before loading so ROUTING_SYSTEM_TEMPLATE is correct.
    if args.prompt_template:
        import prepare_finetune_data as pfd
        tmpl_path = Path(args.prompt_template)
        if not tmpl_path.exists():
            raise FileNotFoundError(f"Prompt template not found: {tmpl_path}")
        hooks.pfd.CONDUCTOR_PROMPT = tmpl_path
        hooks.pfd.ROUTING_SYSTEM_TEMPLATE = hooks.pfd._build_routing_system()
        print(f"Prompt template: {args.prompt_template}")

    print(f"Loading model: {args.model}")
    model, tokenizer = hooks.mlx_load(args.model)

    adapter_path = Path(args.adapter)
    if not adapter_path.exists():
        raise FileNotFoundError(f"Adapter not found: {adapter_path}")
    cfg_file = adapter_path / "adapter_config.json"
    if cfg_file.exists():
        try:
            with open(cfg_file) as f:
                acfg = json.load(f)
            if "num_layers" in acfg and args.num_layers == 8:
                args.num_layers = int(acfg["num_layers"])
            lp = acfg.get("lora_parameters", {})
            if "rank" in lp and args.lora_rank == 8:
                args.lora_rank = int(lp["rank"])
            if "scale" in lp and args.lora_scale == 20.0:
                args.lora_scale = float(lp["scale"])
            if "dropout" in lp and args.lora_dropout == 0.0:
                args.lora_dropout = float(lp["dropout"])
            print(f"Auto-detected LoRA config from {cfg_file}")
        except Exception as e:
            print(f"Warning: could not parse {cfg_file}: {e}")

    apply_lora(
        model,
        args.num_layers,
        rank=args.lora_rank,
        dropout=args.lora_dropout,
        scale=args.lora_scale,
    )
    print(f"LoRA: num_layers={args.num_layers}  rank={args.lora_rank}  scale={args.lora_scale}  dropout={args.lora_dropout}")

    loaded = warm_start_lora(model, adapter_path, strict=False, allow_missing=False)
    print(f"Warm-start (reference): {loaded}\n")

    raw_cases     = hooks.load_training_cases(include_auto=not args.no_auto_cases)
    stats = hooks.case_pool_stats(raw_cases, is_dynamic_mcp=is_dynamic_mcp_case)
    print(
        f"Case sources: golden={stats['golden']} auto={stats['auto']} "
        f"(auto trainable: {len(stats['auto_trainable_ids'])})"
    )
    exclude_mcp   = not args.include_dynamic_mcp
    routing_cases = [
        c for c in raw_cases
        if c.get("mode") == "routing" and c.get("expected_skill")
        and not (exclude_mcp and hooks.is_dynamic_mcp_case(c))
    ]
    if exclude_mcp:
        n_mcp = sum(1 for c in raw_cases
                    if c.get("mode") == "routing" and hooks.is_dynamic_mcp_case(c))
        print(f"Dynamic MCP routing cases excluded from fine-tune pass-rate: {n_mcp} "
              f"(runtime-only skills absent from the training prompt — tracked via "
              f"bare_eval.py). Pass --include-dynamic-mcp to override.\n")

    # Held-out slice: golden routing only. Auto/production cases always stay in
    # the pool (flywheel). Informational only — does not drive best selection.
    held_out_cases = hooks.held_out_golden_routing(
        routing_cases, seed=hooks.mission_seed(Path(args.save_dir)),
    )
    held_out_ids = {c["id"] for c in held_out_cases}
    pool_cases = [c for c in raw_cases if not (c.get("mode") == "routing" and c["id"] in held_out_ids)]
    if held_out_cases:
        print(f"Held out from pair collection (golden only; auto stays in pool): "
              f"{len(held_out_cases)} case(s) — {sorted(held_out_ids)}\n")
    else:
        print(f"Too few golden routing cases (<{hooks.MIN_CASES_FOR_HOLDOUT}) for a held-out signal — skipped\n")

    # ── Phase 1: Collect or load preference pairs ──────────────────────────────
    if args.load_pairs:
        print(f"Loading pairs from {args.load_pairs}...")
        with open(args.load_pairs) as f:
            pairs = [DPOPair(**json.loads(line)) for line in f if line.strip()]
        if exclude_mcp:
            mcp_ids = {c["id"] for c in raw_cases
                       if c.get("mode") == "routing" and hooks.is_dynamic_mcp_case(c)}
            before = len(pairs)
            pairs = [p for p in pairs if p.case_id not in mcp_ids]
            if before != len(pairs):
                print(f"  dropped {before - len(pairs)} dynamic-MCP pairs from loaded set")
        if held_out_ids:
            before = len(pairs)
            pairs = [p for p in pairs if p.case_id not in held_out_ids]
            if before != len(pairs):
                print(f"  dropped {before - len(pairs)} held-out-case pairs from loaded set")
        print(f"Loaded {len(pairs)} pairs\n")
    else:
        model.eval()
        pairs = collect_pairs(
            model, tokenizer, pool_cases,
            K            = args.K_collect,
            max_tokens   = args.max_tokens,
            temp         = args.temp,
            routing_only = args.routing_only,
            email_weight = args.email_weight,
            exclude_dynamic_mcp = exclude_mcp,
        )
        if not pairs:
            print("ERROR: No preference pairs collected.")
            print("The model may be too deterministic at this temperature.")
            print("Try increasing --temp or --K-collect.")
            sys.exit(1)

    if args.save_pairs:
        out_path = Path(args.save_pairs)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            for p in pairs:
                f.write(json.dumps(asdict(p)) + "\n")
        print(f"Pairs saved to {out_path}")

    # ── Phase 2: Cache reference log-probs ────────────────────────────────────
    # Model weights are still at warm-start here — this is the reference policy.
    ref_lps = cache_ref_log_probs(model, tokenizer, pairs)

    # ── Tokenise all pairs up-front (amortises cost across epochs) ────────────
    print("Tokenising pairs...")
    tokenised: list[tuple[list[int], list[int], list[int]]] = []
    ref_lps_filtered: list[tuple[float, float]] = []

    for pair, (ref_c_lp, ref_r_lp) in zip(pairs, ref_lps):
        prompt_ids   = hooks.tokenize_prompt(tokenizer, pair.prompt_str)
        chosen_ids   = hooks.tokenize_completion(tokenizer, pair.chosen)
        rejected_ids = hooks.tokenize_completion(tokenizer, pair.rejected)
        if not chosen_ids or not rejected_ids:
            continue  # skip malformed completions
        tokenised.append((prompt_ids, chosen_ids, rejected_ids))
        ref_lps_filtered.append((ref_c_lp, ref_r_lp))

    print(f"{len(tokenised)} pairs ready for training\n")

    if not tokenised:
        print("ERROR: All pairs were malformed after tokenisation.")
        sys.exit(1)

    # ── Training setup ─────────────────────────────────────────────────────────
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    hooks.write_adapter_config(save_dir, args)

    optimizer        = optim.Adam(learning_rate=args.learning_rate)
    loss_and_grad_fn = nn.value_and_grad(model, dpo_loss)

    model.train()
    best_pass_rate = 0.0
    global_step    = 0
    t_start        = time.time()

    total_steps = args.epochs * len(tokenised)
    print(f"DPO training: epochs={args.epochs}  pairs={len(tokenised)}  "
          f"total_steps={total_steps}  beta={args.beta}  lr={args.learning_rate}")
    print(f"Eval every {args.steps_per_eval} steps\n")

    # Baseline (warm-start, before any DPO update)
    print("--- Baseline eval (step 0) ---")
    model.eval()
    base_pass_rate, base_failures = hooks.run_routing_eval(
        model, tokenizer, routing_cases, args.eval_max_tokens
    )
    n_routing = len(routing_cases)
    n_pass    = round(base_pass_rate * n_routing)
    print(f"Baseline: {base_pass_rate:.1%} ({n_pass}/{n_routing})")
    print(
        "Note: trainer pass rate is routing-oracle on this pool "
        "(≠ live hermetic suite / docs promotion bars)."
    )
    if base_failures:
        print(f"Failures ({len(base_failures)}): {base_failures}")
    if held_out_cases:
        held_out_pass_rate, _ = hooks.run_routing_eval(model, tokenizer, held_out_cases, args.eval_max_tokens)
        n_ho_pass = round(held_out_pass_rate * len(held_out_cases))
        print(f"Held-out pass rate: {held_out_pass_rate:.1%} ({n_ho_pass}/{len(held_out_cases)})")
    print()
    model.train()

    # ── Training epochs ────────────────────────────────────────────────────────
    for epoch in range(1, args.epochs + 1):
        indices = list(range(len(tokenised)))
        random.shuffle(indices)
        epoch_loss = 0.0

        for idx in indices:
            global_step += 1
            prompt_ids, chosen_ids, rejected_ids = tokenised[idx]
            ref_c_lp,   ref_r_lp                = ref_lps_filtered[idx]

            loss_val, grads = loss_and_grad_fn(
                model,
                prompt_ids, chosen_ids, rejected_ids,
                ref_c_lp, ref_r_lp,
                args.beta,
            )
            mx.eval(loss_val, grads)

            grads, _ = optim.clip_grad_norm(grads, max_norm=args.max_grad_norm)
            optimizer.update(model, grads)
            mx.eval(model.parameters(), optimizer.state)
            mx.clear_cache()

            loss_f      = float(loss_val)
            epoch_loss += loss_f
            elapsed     = time.time() - t_start

            print(f"Step {global_step:4d}/{total_steps}  epoch={epoch}  "
                  f"loss={loss_f:.4f}  {elapsed:.0f}s")

            if global_step % args.steps_per_eval == 0:
                print(f"\n--- Eval at step {global_step} ---")
                model.eval()
                pass_rate, failures = hooks.run_routing_eval(
                    model, tokenizer, routing_cases, args.eval_max_tokens
                )
                n_pass = round(pass_rate * n_routing)
                print(f"Pass rate: {pass_rate:.1%} ({n_pass}/{n_routing})")
                if failures:
                    print(f"Failures ({len(failures)}): {failures}")
                else:
                    print("All routing cases PASS ✓")
                if held_out_cases:
                    held_out_pass_rate, _ = hooks.run_routing_eval(model, tokenizer, held_out_cases, args.eval_max_tokens)
                    n_ho_pass = round(held_out_pass_rate * len(held_out_cases))
                    print(f"Held-out pass rate: {held_out_pass_rate:.1%} ({n_ho_pass}/{len(held_out_cases)})")

                if pass_rate > best_pass_rate:
                    best_pass_rate = pass_rate
                    best_path = save_dir / "best"
                    hooks.save_final_adapter(model, best_path, args)
                    print(f"New best: {pass_rate:.1%} → saved to {best_path}")
                print()
                model.train()

        avg_epoch_loss = epoch_loss / len(tokenised)
        print(f"\n=== Epoch {epoch}/{args.epochs} done  avg_loss={avg_epoch_loss:.4f} ===\n")
        save_checkpoint(model, save_dir, global_step)

    # ── Final save and eval ────────────────────────────────────────────────────
    hooks.save_final_adapter(model, save_dir, args)

    print("\n=== Final Eval ===")
    model.eval()
    pass_rate, failures = hooks.run_routing_eval(
        model, tokenizer, routing_cases, args.eval_max_tokens
    )
    n_pass = round(pass_rate * n_routing)
    print(f"Pass rate: {pass_rate:.1%} ({n_pass}/{n_routing})")
    if failures:
        print(f"Failures: {failures}")
    else:
        print("100% — all routing cases pass ✓")
    if held_out_cases:
        held_out_pass_rate, _ = hooks.run_routing_eval(model, tokenizer, held_out_cases, args.eval_max_tokens)
        n_ho_pass = round(held_out_pass_rate * len(held_out_cases))
        print(f"Held-out pass rate: {held_out_pass_rate:.1%} ({n_ho_pass}/{len(held_out_cases)})")

    if pass_rate >= best_pass_rate or not (save_dir / "best").exists():
        best_pass_rate = pass_rate
        best_path = save_dir / "best"
        hooks.save_final_adapter(model, best_path, args)
        print(f"Best adapter ensured at {best_path}")

    from datetime import datetime, timezone
    meta = {
        "pass_rate": best_pass_rate,
        "base_model": args.model,
        "num_layers": args.num_layers,
        "lora_rank": args.lora_rank,
        "lora_scale": args.lora_scale,
        "lora_dropout": args.lora_dropout,
        "saved_at": datetime.now(timezone.utc).isoformat(),
    }
    with open(save_dir / "checkpoint_metadata.json", "w") as f:
        json.dump(meta, f, indent=2)
    if (save_dir / "best").exists():
        with open(save_dir / "best" / "checkpoint_metadata.json", "w") as f:
            json.dump(meta, f, indent=2)

    print(f"Best pass rate during training: {best_pass_rate:.1%}")
    print(f"Total time: {(time.time() - t_start) / 60:.1f} min")


# ── CLI ───────────────────────────────────────────────────────────────────────

