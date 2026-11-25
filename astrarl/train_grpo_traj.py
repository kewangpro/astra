"""Agent / trajectory GRPO training loop — platform-owned (astrarl)."""
from __future__ import annotations
import json, random, sys, time
from pathlib import Path
from typing import Any
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten, tree_map
from astrarl.grpo_math import compute_advantages, grpo_loss_single, ref_log_prob
from astratune.lora_io import apply_lora, warm_start_lora

def run_grpo_traj_train(args, hooks) -> None:
    print(f"Loading model: {args.model}")
    model, tokenizer = hooks.mlx_load(args.model)

    apply_lora(
        model,
        args.num_layers,
        rank=args.lora_rank,
        dropout=args.lora_dropout,
        scale=args.lora_scale,
    )
    print(f"LoRA: num_layers={args.num_layers} rank={args.lora_rank} scale={args.lora_scale}")

    if not args.no_warmstart and args.adapter and Path(args.adapter).exists():
        loaded = warm_start_lora(model, args.adapter, strict=False)
        print(f"Warm-start from: {loaded}")
    else:
        print("Cold start (no adapter loaded)")

    model.train()
    n_trainable = sum(p.size for _, p in tree_flatten(model.trainable_parameters()))
    total = sum(p.size for _, p in tree_flatten(model.parameters()))
    print(f"Trainable params: {n_trainable:,} / {total:,} ({100 * n_trainable / total:.3f}%)")

    reward_fn = hooks.make_reward_fn(args.reward_mode)
    e2e_cases = []
    if args.reward_mode in ("e2e", "ensemble_e2e"):
        e2e_cases = hooks.load_e2e_stub_cases(args.e2e_cases) if args.e2e_cases else []
        if not e2e_cases:
            e2e_cases = list(hooks.DEFAULT_E2E_STUBS)
        print(f"E2E stub cases: {len(e2e_cases)}")
    else:
        print(f"AgentGym scenarios: {len(hooks.SCENARIOS)}")

    from astrarl.rollout import resolve_clip_epsilons

    eps_low, eps_high = resolve_clip_epsilons(
        args.clip_epsilon,
        getattr(args, "clip_epsilon_low", None),
        getattr(args, "clip_epsilon_high", None),
    )
    optimizer = optim.Adam(learning_rate=args.learning_rate)
    loss_and_grad_fn = nn.value_and_grad(model, grpo_loss_single)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    hooks.write_adapter_config(save_dir, args)

    best_pass_rate = -1.0
    reward_ema = 0.0
    ema_alpha = 0.1
    t_start = time.time()

    print("--- Baseline eval (step 0) ---")
    model.eval()
    base_rate, n_pass, n_total = hooks.eval_task_success(
        model, tokenizer, len(hooks.SCENARIOS), args.max_turns, args.max_tokens,
    )
    print(f"Pass rate: {base_rate:.1%} ({n_pass}/{n_total})")
    print(
        "Note: agent GRPO pass rate is AgentGym task_success "
        "(≠ product routing hermetic suite)."
    )
    best_pass_rate = base_rate
    best_path = save_dir / "best"
    hooks.save_final_adapter(model, best_path, args)

    print(
        f"Starting Agent GRPO: iters={args.iters} K={args.num_generations} "
        f"temp={args.temp} reward_mode={args.reward_mode}\n"
    )

    for step in range(1, args.iters + 1):
        # ── K independent trajectories ───────────────────────────────────────
        model.eval()
        trajectories = []
        for _ in range(args.num_generations):
            if e2e_cases:
                case = random.choice(e2e_cases)
                traj = hooks.rollout_e2e_stub(
                    model, tokenizer, case, args.max_turns, args.max_tokens, args.temp,
                )
            else:
                scenario_idx = random.randrange(len(hooks.SCENARIOS))
                traj = hooks.rollout_agent_gym(
                    model, tokenizer, scenario_idx, args.max_turns,
                    args.max_tokens, args.temp, args.tool_failure_prob,
                )
            trajectories.append(traj)

        rewards = [
            reward_fn(
                t["case"], t["messages"], t["tool_log"],
                t["episode_return"], t["task_success"],
            )
            for t in trajectories
        ]
        batch_mean = sum(rewards) / len(rewards)
        reward_ema = (1 - ema_alpha) * reward_ema + ema_alpha * batch_mean
        advantages = compute_advantages(rewards, baseline=reward_ema)

        # Reference log-probs per turn
        model.eval()
        old_lps: list[list[float]] = []
        for traj in trajectories:
            turn_lps = []
            for turn in traj["turns"]:
                if not turn["comp_ids"]:
                    turn_lps.append(0.0)
                    continue
                turn_lps.append(ref_log_prob(model, turn["prompt_ids"], turn["comp_ids"]))
                mx.clear_cache()
            old_lps.append(turn_lps)

        # Gradient accumulation across all turns of all trajectories
        model.train()
        K = len(trajectories)
        accumulated_grads = None
        total_loss_float = 0.0
        n_updates = 0

        for traj, adv, turn_old in zip(trajectories, advantages, old_lps):
            if not traj["turns"]:
                continue
            for turn, old_lp in zip(traj["turns"], turn_old):
                if not turn["comp_ids"]:
                    continue
                (loss_i, _ntoks), grads_i = loss_and_grad_fn(
                    model, turn["prompt_ids"], turn["comp_ids"],
                    float(adv), old_lp, eps_low, eps_high,
                )
                mx.eval(loss_i, grads_i)
                total_loss_float += float(loss_i)
                n_updates += 1
                if accumulated_grads is None:
                    accumulated_grads = tree_map(lambda g: g / max(1, K), grads_i)
                else:
                    accumulated_grads = tree_map(
                        lambda a, g: a + g / max(1, K), accumulated_grads, grads_i,
                    )
                mx.clear_cache()

        if accumulated_grads is None or n_updates == 0:
            print(f"Step {step:4d} | SKIP (no valid turns)")
            continue

        # Average loss over turns for logging; grads already /K per traj
        total_loss_float /= max(1, n_updates)

        accumulated_grads, _ = optim.clip_grad_norm(
            accumulated_grads, max_norm=args.max_grad_norm
        )
        optimizer.update(model, accumulated_grads)
        mx.eval(model.parameters(), optimizer.state)
        mx.clear_cache()

        if step % args.steps_per_report == 0:
            r_str = " ".join(f"{r:.2f}" for r in rewards)
            case_id = trajectories[0]["case"].get("id", "?")
            elapsed = time.time() - t_start
            print(
                f"Step {step:4d}/{args.iters} | loss={total_loss_float:.4f} | "
                f"baseline={reward_ema:.3f} | rewards=[{r_str}] | "
                f"{case_id[:35]:<35} | {elapsed:.0f}s"
            )

        if step % args.steps_per_eval == 0:
            print(f"\n--- Eval at step {step} ---")
            model.eval()
            pass_rate, n_pass, n_total = hooks.eval_task_success(
                model, tokenizer, len(hooks.SCENARIOS), args.max_turns, args.max_tokens,
            )
            print(f"Pass rate: {pass_rate:.1%} ({n_pass}/{n_total})")
            if pass_rate > best_pass_rate:
                best_pass_rate = pass_rate
                hooks.save_final_adapter(model, save_dir / "best", args)
                print(f"  New best → {save_dir / 'best'}")

        if step % args.save_every == 0:
            hooks.save_checkpoint(model, save_dir, step)

    hooks.save_final_adapter(model, save_dir, args)
    print(f"\nDone. Best pass rate: {best_pass_rate:.1%}")
    # Final pass-rate line for Astra telemetry parsers
    model.eval()
    pass_rate, n_pass, n_total = hooks.eval_task_success(
        model, tokenizer, len(hooks.SCENARIOS), args.max_turns, args.max_tokens,
    )
    print(f"Pass rate: {pass_rate:.1%} ({n_pass}/{n_total})")


