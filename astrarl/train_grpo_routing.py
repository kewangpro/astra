"""MLX routing GRPO training loop — platform-owned.

Missions execv ``python -m astrarl.cli grpo``. Ensemble ``finetune.grpo``
supplies hooks (oracle, case pool, adapter I/O wrappers) via ``--use-case``.
"""
from __future__ import annotations

import random
import time
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten, tree_map

from astrarl.grpo_math import grpo_loss_single, ref_log_prob
from astrarl.mlx_grpo_step import collect_group
from astrarl.rollout import make_rollout_backend, resolve_clip_epsilons


def run_grpo_routing_train(args, hooks) -> None:
    # Apply --prompt-template before anything else so ROUTING_SYSTEM_TEMPLATE is rebuilt
    # with the correct prompt. CONDUCTOR_PROMPT env var is NOT read here — must use this flag.
    if args.prompt_template:
        pfd = hooks.pfd
        tmpl_path = Path(args.prompt_template)
        if not tmpl_path.exists():
            raise FileNotFoundError(f"prompt template not found: {tmpl_path}")
        hooks.pfd.CONDUCTOR_PROMPT = tmpl_path
        hooks.pfd.ROUTING_SYSTEM_TEMPLATE = hooks.pfd._build_routing_system()
        print(f"Prompt template: {args.prompt_template}")

    print(f"Loading model: {args.model}")
    model, tokenizer = hooks.mlx_load(args.model)

    # Apply LoRA layers + optional warm-start (platform helper).
    hooks.apply_lora(
        model,
        args.num_layers,
        rank=args.lora_rank,
        dropout=args.lora_dropout,
        scale=args.lora_scale,
    )
    print(f"LoRA: num_layers={args.num_layers} rank={args.lora_rank} scale={args.lora_scale}")

    # Warm-start: load matching LoRA weights from SFT adapter (strict=False so
    # layers not present in the checkpoint initialise at zero — equivalent to
    # LoRA default init for any newly added layers).
    if not args.no_warmstart and Path(args.adapter).exists():
        loaded = hooks.warm_start_lora(model, args.adapter, strict=False)
        print(f"Warm-start from: {loaded}")
    else:
        print("Cold start (no adapter loaded)")

    model.train()

    # rollout / DAPO helpers imported at module level from astrarl

    rollout_kind = getattr(args, "rollout_backend", "mlx") or "mlx"
    rollout = make_rollout_backend(
        rollout_kind,
        model=model,
        tokenizer=tokenizer,
        base_url=getattr(args, "rollout_base_url", "") or "",
        api_key=getattr(args, "rollout_api_key", "") or "",
        openai_model=getattr(args, "rollout_model", "") or "default",
    )
    print(f"Rollout backend: {rollout_kind}"
          + (f" @ {args.rollout_base_url}" if rollout_kind != "mlx" else ""))

    eps_low, eps_high = resolve_clip_epsilons(
        args.clip_epsilon,
        getattr(args, "clip_epsilon_low", None),
        getattr(args, "clip_epsilon_high", None),
    )
    print(f"Clip epsilons: low={eps_low} high={eps_high} "
          f"(symmetric={eps_low == eps_high})")

    reward_plugin = hooks.RoutingRewardPlugin(
        score_fn=hooks.score_completion,
        build_prompt_fn=hooks.build_prompt_str,
        reward_schema=args.reward_schema,
    )

    # Count trainable parameters
    n_trainable, _n_total = hooks.trainable_param_counts(model)
    total = sum(p.size for _, p in tree_flatten(model.parameters()))
    print(f"Trainable params: {n_trainable:,} / {total:,} ({100 * n_trainable / total:.3f}%)")

    # Load eval cases (golden + §RSI3 auto_eval_cases unless --no-auto-cases)
    raw_cases = hooks.load_training_cases(include_auto=not args.no_auto_cases)
    stats = hooks.case_pool_stats(raw_cases, is_dynamic_mcp=hooks.is_dynamic_mcp_case)
    print(
        f"Case sources: golden={stats['golden']} auto={stats['auto']} "
        f"(auto trainable after MCP filter: {len(stats['auto_trainable_ids'])}; "
        f"skipped dynamic MCP: {stats['auto_skipped_dynamic_mcp']})"
    )
    if args.no_auto_cases:
        print(f"Auto cases excluded (--no-auto-cases). Golden only: {hooks.GOLDEN_CASES_PATH}")
    elif hooks.AUTO_CASES_PATH.is_file():
        print(f"Auto cases included: {hooks.AUTO_CASES_PATH}")
    # Must match run_routing_eval's internal filter — cases without expected_skill
    # (multi-grader) are skipped by the oracle, and counting them inflates the
    # displayed denominator (the old "(72/97)" logs; the real total is 65).
    exclude_mcp = not args.include_dynamic_mcp
    routing_cases = [
        c for c in raw_cases
        if c.get("mode") == "routing" and c.get("expected_skill")
        and not (exclude_mcp and hooks.is_dynamic_mcp_case(c))
    ]
    if exclude_mcp:
        n_mcp = sum(1 for c in raw_cases
                    if c.get("mode") == "routing" and hooks.is_dynamic_mcp_case(c))
        print(f"Dynamic MCP routing cases excluded from pass-rate: {n_mcp} "
              f"(runtime-only skills absent from the training prompt). "
              f"Pass --include-dynamic-mcp to override.")
    # Held-out slice: golden routing only. Auto/production cases always stay in
    # the pool (flywheel). Informational "Held-out pass rate" only — does NOT
    # replace Pass rate or affect best-checkpoint selection. Seeded from
    # --save-dir so every astra mission iteration scores the same held-out set.
    held_out_cases = hooks.held_out_golden_routing(
        routing_cases, seed=hooks.mission_seed(Path(args.save_dir)),
    )
    held_out_ids = {c["id"] for c in held_out_cases}
    pool_cases = [c for c in raw_cases if not (c.get("mode") == "routing" and c["id"] in held_out_ids)]
    if held_out_cases:
        print(f"Held out from training pool (golden only; auto stays in pool): "
              f"{len(held_out_cases)} case(s) — {sorted(held_out_ids)}")
    else:
        print(f"Too few golden routing cases (<{hooks.MIN_CASES_FOR_HOLDOUT}) for a held-out signal — skipped")

    # Initial pool without focus weighting — used until step-0 baseline identifies failures
    case_pool = hooks.build_case_pool(pool_cases, args.email_weight, args.routing_only,
                                exclude_dynamic_mcp=exclude_mcp)
    print(f"Case pool: {len(case_pool)} entries ({len(routing_cases)} unique routing cases)")
    print(f"Email cases upweighted {args.email_weight}×: {sorted(hooks.EMAIL_CASES)}\n")

    optimizer = optim.Adam(learning_rate=args.learning_rate)
    loss_and_grad_fn = nn.value_and_grad(model, grpo_loss_single)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    hooks.write_adapter_config(save_dir, args)  # checkpoints are servable even if the run dies mid-training

    best_pass_rate = 0.0
    step_losses: list[float] = []

    print(f"Starting GRPO training: iters={args.iters} K={args.num_generations} "
          f"temp={args.temp} lr={args.learning_rate}")
    print(f"Eval every {args.steps_per_eval} steps, save every {args.save_every} steps\n")

    # ── Step-0 baseline (before any weight update) ───────────────────────────
    print("--- Baseline eval (step 0) ---")
    model.eval()
    base_pass_rate, base_failures = hooks.run_routing_eval(
        model, tokenizer, routing_cases, args.eval_max_tokens, schema=args.reward_schema
    )
    n_routing = len(routing_cases)
    n_pass    = round(base_pass_rate * n_routing)
    print(f"Baseline pass rate: {base_pass_rate:.1%} ({n_pass}/{n_routing})")
    print(
        "Note: trainer pass rate is routing-oracle on this pool "
        "(≠ live hermetic suite / docs promotion bars)."
    )
    if base_failures:
        print(f"Baseline failures ({len(base_failures)}): {base_failures}")
    if held_out_cases:
        held_out_pass_rate, held_out_failures = hooks.run_routing_eval(
            model, tokenizer, held_out_cases, args.eval_max_tokens, schema=args.reward_schema
        )
        n_ho_pass = round(held_out_pass_rate * len(held_out_cases))
        print(f"Held-out pass rate: {held_out_pass_rate:.1%} ({n_ho_pass}/{len(held_out_cases)})")
    focus_ids = set(base_failures) - hooks.EMAIL_CASES
    # Seed focus with §RSI3 auto cases (production misses) unless disabled —
    # they are the point of the online-RL data flywheel.
    if not args.no_auto_cases and not args.no_auto_focus:
        prod_focus = hooks.auto_focus_ids(pool_cases, is_dynamic_mcp=hooks.is_dynamic_mcp_case if exclude_mcp else None)
        prod_focus -= hooks.EMAIL_CASES
        if prod_focus:
            print(f"Auto/production focus seeds ({len(prod_focus)}): {sorted(prod_focus)}")
            focus_ids |= prod_focus

    # Rebuild pool with 10× focus on baseline-failing + auto cases so training
    # steps spend time where reward variance / production signal exists.
    case_pool = hooks.build_case_pool(
        pool_cases, args.email_weight, args.routing_only,
        focus_ids=focus_ids, focus_weight=args.focus_weight,
        exclude_dynamic_mcp=exclude_mcp,
    )
    focused = len(focus_ids)
    print(f"Focused pool: {len(case_pool)} entries ({focused} failing cases × {args.focus_weight}× weight)\n")

    t_start = time.time()
    # Tracks consecutive all-zero or all-one reward steps per case_id.
    # All-zero streak → impenetrable prior, demote from focus.
    # All-one streak  → fully mastered, also demote (zero gradient, wasted steps).
    zero_streak: dict[str, int] = {}
    one_streak:  dict[str, int] = {}

    # Running mean baseline (EMA, α=0.1) — used instead of within-batch mean so
    # all-fail batches still produce a negative advantage signal.
    reward_ema: float = 0.0
    ema_alpha:  float = 0.1
    dynamic_resamples = 0
    flat_groups_skipped = 0
    dynamic_retries = int(getattr(args, "dynamic_sample_retries", 0) or 0)

    for step in range(1, args.iters + 1):
        case = random.choice(case_pool)
        prompt_str = hooks.build_prompt_str(case)

        # ── 1–2. Platform group collect (rollout + plugin score + DAPO sample) ─
        model.eval()
        group, reward_ema = collect_group(
            rollout=rollout,
            plugin=reward_plugin,
            case=case,
            k=args.num_generations,
            temperature=args.temp,
            max_tokens=args.max_tokens,
            reward_ema=reward_ema,
            ema_alpha=ema_alpha,
            dynamic_sample_retries=dynamic_retries,
        )
        dynamic_resamples += group.dynamic_resamples
        if group.skipped:
            flat_groups_skipped += 1
            print(f"Step {step:4d} | {case['id'][:40]:<40} | SKIP (flat rewards after "
                  f"{dynamic_retries} dynamic-sample retries)")
            continue
        completions = group.completions
        rewards = group.rewards
        advantages = group.advantages

        # Symmetric pool pruning: demote cases that are impenetrable (all-zero
        # streak) or fully mastered (all-one streak) — both produce zero gradient.
        cid = case["id"]
        all_zero = all(r <= 0.0 for r in rewards)
        all_one  = all(r >= 1.0 for r in rewards)

        if all_zero:
            zero_streak[cid] = zero_streak.get(cid, 0) + 1
            one_streak[cid]  = 0
            if zero_streak[cid] == args.max_zero_steps:
                focus_ids.discard(cid)
                case_pool = hooks.build_case_pool(
                    pool_cases, args.email_weight, args.routing_only,
                    focus_ids=focus_ids, focus_weight=args.focus_weight,
                )
                print(f"  [prune-zero] {cid} demoted after {args.max_zero_steps} "
                      f"all-zero steps (pool {len(case_pool)}, focused {len(focus_ids)})")
        elif all_one:
            one_streak[cid]  = one_streak.get(cid, 0) + 1
            zero_streak[cid] = 0
            if one_streak[cid] == args.max_one_steps:
                focus_ids.discard(cid)
                case_pool = hooks.build_case_pool(
                    pool_cases, args.email_weight, args.routing_only,
                    focus_ids=focus_ids, focus_weight=args.focus_weight,
                )
                print(f"  [prune-one]  {cid} demoted after {args.max_one_steps} "
                      f"all-one steps (pool {len(case_pool)}, focused {len(focus_ids)})")
        else:
            zero_streak[cid] = 0
            one_streak[cid]  = 0

        # ── 3. Tokenise + compute reference log-probs (no grad) ─────────────────
        prompt_ids = hooks.tokenize_prompt(tokenizer, prompt_str)
        all_comp_ids = [hooks.tokenize_completion(tokenizer, c) for c in completions]

        # Skip step if all completions are malformed (nothing to learn from)
        if all(len(ids) == 0 for ids in all_comp_ids):
            print(f"Step {step:4d} | {case['id'][:40]:<40} | SKIP (all completions malformed)")
            continue

        # Reference log-probs captured before any weight update this step.
        # Used by PPO clip to bound policy movement.
        model.eval()
        old_log_probs = []
        for cids in all_comp_ids:
            lp = ref_log_prob(model, prompt_ids, cids) if len(cids) > 0 else 0.0
            old_log_probs.append(lp)
            mx.clear_cache()  # flush dead Metal buffers after each forward pass

        # ── 4. Per-completion gradient accumulation ───────────────────────────
        # Each completion is differentiated independently so MLX only holds
        # one activation graph at a time — avoids K× memory blow-up.
        model.train()
        K = len(all_comp_ids)
        accumulated_grads = None
        total_loss_float  = 0.0
        total_ntoks       = 0

        for comp_ids, adv, old_lp in zip(all_comp_ids, advantages, old_log_probs):
            (loss_i, ntoks_i), grads_i = loss_and_grad_fn(
                model, prompt_ids, comp_ids, float(adv), old_lp, eps_low, eps_high,
            )
            mx.eval(loss_i, grads_i)          # materialise + free activations

            total_loss_float += float(loss_i) / K
            total_ntoks      += ntoks_i

            if accumulated_grads is None:
                accumulated_grads = tree_map(lambda g: g / K, grads_i)
            else:
                accumulated_grads = tree_map(
                    lambda a, g: a + g / K, accumulated_grads, grads_i
                )
            mx.clear_cache()  # flush dead activation buffers after each backward pass

        # Gradient clipping — clip_grad_norm returns (clipped_grads, norm)
        accumulated_grads, _ = optim.clip_grad_norm(
            accumulated_grads, max_norm=args.max_grad_norm
        )

        # ── 5. Update LoRA weights ────────────────────────────────────────────
        optimizer.update(model, accumulated_grads)
        mx.eval(model.parameters(), optimizer.state)
        mx.clear_cache()  # flush optimizer state buffers after weight update

        loss_float = total_loss_float
        step_losses.append(loss_float)

        # ── Logging ───────────────────────────────────────────────────────────
        if step % args.steps_per_report == 0:
            avg_loss = sum(step_losses[-args.steps_per_report:]) / args.steps_per_report
            elapsed  = time.time() - t_start
            r_str    = " ".join(f"{r:.1f}" for r in rewards)
            print(f"Step {step:4d}/{args.iters} | loss={avg_loss:.4f} | baseline={reward_ema:.3f} | "
                  f"rewards=[{r_str}] | {case['id'][:35]:<35} | {elapsed:.0f}s")

        # ── Eval ──────────────────────────────────────────────────────────────
        if step % args.steps_per_eval == 0:
            print(f"\n--- Eval at step {step} ---")
            model.eval()
            pass_rate, failures = hooks.run_routing_eval(
                model, tokenizer, routing_cases, args.eval_max_tokens, schema=args.reward_schema
            )
            n_routing = len(routing_cases)
            n_pass    = round(pass_rate * n_routing)
            print(f"Pass rate: {pass_rate:.1%} ({n_pass}/{n_routing})")
            if failures:
                print(f"Failures ({len(failures)}): {failures}")
            else:
                print("All routing cases PASS ✓")
            if held_out_cases:
                held_out_pass_rate, held_out_failures = hooks.run_routing_eval(
                    model, tokenizer, held_out_cases, args.eval_max_tokens, schema=args.reward_schema
                )
                n_ho_pass = round(held_out_pass_rate * len(held_out_cases))
                print(f"Held-out pass rate: {held_out_pass_rate:.1%} ({n_ho_pass}/{len(held_out_cases)})")

            if pass_rate > best_pass_rate:
                best_pass_rate = pass_rate
                best_path = save_dir / "best"
                hooks.save_final_adapter(model, best_path, args)
                print(f"New best: {pass_rate:.1%} → saved to {best_path}")
            print()
            model.train()

        # ── Checkpoint ────────────────────────────────────────────────────────
        if step % args.save_every == 0:
            hooks.save_checkpoint(model, save_dir, step)

    # ── Final save and eval ───────────────────────────────────────────────────
    hooks.save_final_adapter(model, save_dir, args)

    print("\n=== Final Eval ===")
    model.eval()
    pass_rate, failures = hooks.run_routing_eval(model, tokenizer, routing_cases, args.eval_max_tokens, schema=args.reward_schema)
    n_routing = len(routing_cases)
    n_pass    = round(pass_rate * n_routing)
    print(f"Pass rate: {pass_rate:.1%} ({n_pass}/{n_routing})")
    if failures:
        print(f"Failures: {failures}")
    else:
        print("100% — all routing cases pass ✓")
    if held_out_cases:
        held_out_pass_rate, held_out_failures = hooks.run_routing_eval(
            model, tokenizer, held_out_cases, args.eval_max_tokens, schema=args.reward_schema
        )
        n_ho_pass = round(held_out_pass_rate * len(held_out_cases))
        print(f"Held-out pass rate: {held_out_pass_rate:.1%} ({n_ho_pass}/{len(held_out_cases)})")
    print(f"Best pass rate during training: {best_pass_rate:.1%}")
    print(f"Dynamic resamples: {dynamic_resamples} | flat groups skipped: {flat_groups_skipped}")
    print(f"Total time: {(time.time() - t_start) / 60:.1f} min")


