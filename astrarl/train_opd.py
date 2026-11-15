"""OPD training loop — platform-owned (astrarl).

Missions execv ``python -m astrarl.cli opd``. Use-case hooks for routing oracle / case helpers via ``--use-case``.
"""
from __future__ import annotations

import random
import sys
import time
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import yaml
from mlx.utils import tree_flatten, tree_map

from astrarl.opd_math import (
    center_group_advantages,
    clip_advantage,
    kl_policy_advantage,
    mean_reverse_kl,
    mean_token_logprob,
    opd_sequence_advantage,
)

def run_opd_train(args, hooks) -> None:
    if args.prompt_template:
        tmpl_path = Path(args.prompt_template)
        if not tmpl_path.exists():
            raise FileNotFoundError(f"Prompt template not found: {tmpl_path}")
        hooks.pfd.CONDUCTOR_PROMPT = tmpl_path
        hooks.pfd.ROUTING_SYSTEM_TEMPLATE = hooks.pfd._build_routing_system()
        print(f"Prompt template: {args.prompt_template}")

    with open(hooks.EVAL_CASES_PATH) as f:
        cases_raw = yaml.safe_load(f)
    if isinstance(cases_raw, dict):
        cases_raw = cases_raw.get("cases", [])
    raw_cases = [c for c in cases_raw if c.get("mode") == "routing"]
    for c in raw_cases:
        if not c.get("expected_skill") and c.get("graders"):
            for g in c.get("graders"):
                if isinstance(g, dict) and g.get("type") == "routing" and g.get("expected_skill"):
                    c["expected_skill"] = g["expected_skill"]
                    break
    from prepare_finetune_data import case_plan_json as _case_plan_json
    raw_cases = [c for c in raw_cases
                 if c.get("expected_skill") or _case_plan_json(c) or c.get("mock_llm_response")]

    if not args.include_dynamic_mcp:
        before = len(raw_cases)
        raw_cases = [c for c in raw_cases if not hooks.is_dynamic_mcp_case(c)]
        excluded = before - len(raw_cases)
        if excluded:
            print(f"Dynamic MCP routing cases excluded from fine-tune pass-rate: {excluded} "
                  f"(--include-dynamic-mcp to override)")

    focus_ids = None
    if args.focus_cases:
        focus_ids = {c.strip() for c in args.focus_cases.split(",") if c.strip()}
        raw_cases = [c for c in raw_cases if c["id"] in focus_ids]
        print(f"Focus scope: {len(raw_cases)} case(s) — {sorted(focus_ids)}")

    train_cases, valid_cases = hooks.split_cases(raw_cases, seed=hooks.mission_seed(Path(args.save_dir)))
    if valid_cases:
        print(f"Case split: {len(train_cases)} train / {len(valid_cases)} held-out valid")
    else:
        print(f"Case split: {len(train_cases)} case(s) — below {hooks.MIN_CASES_FOR_HOLDOUT}, "
              f"no held-out split; eval will reuse training cases")

    if not train_cases:
        print("ERROR: no training cases after filtering.")
        sys.exit(1)

    # Student
    print(f"Loading student: {args.model}")
    model, tokenizer = hooks.mlx_load(args.model)
    hooks.apply_lora(
        model,
        args.num_layers,
        rank=args.lora_rank,
        dropout=args.lora_dropout,
        scale=args.lora_scale,
    )
    print(f"LoRA: num_layers={args.num_layers}  rank={args.lora_rank}  scale={args.lora_scale}")

    if args.no_adapter:
        print("Cold start: NO warm-start adapter — LoRA weights are freshly initialised "
              "and every non-LoRA weight is at base-model values.")
        print("  There is no prior checkpoint to beat; this run establishes a baseline "
              "rather than testing against one.\n")
    else:
        loaded = hooks.warm_start_lora(model, args.adapter, strict=False, allow_missing=False)
        print(f"Warm-start: {loaded}")

    n_trainable = sum(p.size for _, p in tree_flatten(model.trainable_parameters()))
    total = sum(p.size for _, p in tree_flatten(model.parameters()))
    print(f"Trainable params: {n_trainable:,} / {total:,} ({100 * n_trainable / total:.3f}%)")

    # Teacher — frozen MLX model, no LoRA. Same-family tokenizer assumed so
    # student token IDs are valid under the teacher.
    print(f"Loading teacher (MLX, frozen): {args.teacher_model}")
    try:
        teacher, teacher_tok = hooks.mlx_load(args.teacher_model)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load teacher model {args.teacher_model!r}. "
            f"OPD requires an MLX checkpoint that can score student token IDs "
            f"(not an Ollama chat tag). Original error: {exc}"
        ) from exc
    # Prefer student tokenizer for encoding rollouts; teacher must accept those IDs.
    hooks.assert_compatible_tokenizers(
        tokenizer, teacher_tok, allow_mismatch=args.allow_mismatched_tokenizer
    )
    teacher.eval()
    print(f"Teacher ready (no LoRA). kl_coef={args.kl_coef}  outcome_bonus={args.outcome_bonus}\n")

    def _static(pool):
        return [c for c in pool
                if c.get("mode") == "routing" and c.get("expected_skill")
                and not hooks.is_dynamic_mcp_case(c)]

    def _mcp(pool):
        return [c for c in pool
                if c.get("mode") == "routing" and c.get("expected_skill")
                and hooks.is_dynamic_mcp_case(c)]

    def _model_routed(pool):
        return [c for c in _static(pool) if not hooks.is_pre_llm_case(c)]

    eval_pool = valid_cases if valid_cases else train_cases

    def report_eval(label: str = "Pass rate") -> tuple[float, list[str], tuple[int, int] | None]:
        static_cases = _static(eval_pool)
        mcp_cases = _mcp(eval_pool)
        routed_cases = _model_routed(eval_pool)
        rate, fails = hooks.run_routing_eval(
            model, tokenizer, static_cases, args.eval_max_tokens, schema=args.reward_schema
        )
        n_pass = len(static_cases) - len(fails)

        _routed_ids = {c["id"] for c in routed_cases}
        _routed_fails = [f for f in fails if f in _routed_ids]
        n_routed_pass = len(routed_cases) - len(_routed_fails)
        routed_rate = (n_routed_pass / len(routed_cases)) if routed_cases else rate
        _n_prellm = len(static_cases) - len(routed_cases)

        print(f"{label}: {routed_rate:.1%} ({n_routed_pass}/{len(routed_cases)})"
              f"  ← model-routed; drives selection ({_n_prellm} pre-LLM excluded)")
        print(f"  incl. pre-LLM: {n_pass}/{len(static_cases)} ({rate:.1%})"
              f"  ← historical 'static' denominator, kept for continuity")

        if mcp_cases:
            mcp_rate, mcp_fails = hooks.run_routing_eval(
                model, tokenizer, mcp_cases, args.eval_max_tokens, schema=args.reward_schema
            )
            n_mcp_pass = len(mcp_cases) - len(mcp_fails)
            blended_pass = n_pass + n_mcp_pass
            blended_total = len(static_cases) + len(mcp_cases)
            print(f"  static:        {n_pass}/{len(static_cases)} ({rate:.1%})"
                  f"  ← learnable subset incl. pre-LLM; NOT the selection metric")
            print(f"  dynamic MCP:   {n_mcp_pass}/{len(mcp_cases)} ({mcp_rate:.1%})"
                  f"  ← runtime-only skills, absent from the training prompt")
            print(f"  blended:       {blended_pass}/{blended_total} "
                  f"({blended_pass / blended_total:.1%})"
                  f"  ← overall capability; quote THIS as the checkpoint's pass rate")
            if mcp_fails:
                print(f"  MCP failures ({len(mcp_fails)}): {mcp_fails}")
            return routed_rate, fails, (blended_pass, blended_total)
        return routed_rate, fails, None

    # Baseline before any update
    model.eval()
    base_pass_rate, base_failures, _ = report_eval("Baseline")
    if base_failures:
        print(f"Failures ({len(base_failures)}): {base_failures}")
    print()

    print(f"{len(train_cases)} examples ready for training\n")

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    hooks.write_adapter_config(save_dir, args)
    hooks.save_final_adapter(model, save_dir, args)
    hooks.save_final_adapter(model, save_dir / "best", args)

    optimizer = optim.Adam(learning_rate=args.learning_rate)
    loss_and_grad_fn = nn.value_and_grad(model, grpo_loss_single)

    total_steps = args.iters
    print(f"total_steps={total_steps}")
    print(f"OPD: iters={total_steps}  K={args.num_generations}  temp={args.temp}  "
          f"lr={args.learning_rate}  kl_coef={args.kl_coef}  "
          f"outcome_bonus={args.outcome_bonus}  clip_eps={args.clip_epsilon}  "
          f"adv_clip={args.adv_clip}")
    print(f"Eval every {args.steps_per_eval} steps, save every {args.save_every} steps\n")

    best_pass_rate = 0.0
    best_blended: tuple[int, int] | None = None
    t_start = time.time()
    total_rollouts = 0

    for step in range(1, total_steps + 1):
        # Sample batch_size cases (default 1); accumulate grads across their K rollouts.
        batch_cases = [random.choice(train_cases) for _ in range(max(1, args.batch_size))]

        model.eval()
        # (prompt_ids, comp_ids, advantage, old_log_prob, raw_rkl)
        triples: list[tuple[list[int], list[int], float, float, float]] = []

        for case in batch_cases:
            prompt_str = hooks.build_prompt_str(case)
            completions = hooks.generate_k_completions(
                model, tokenizer, prompt_str,
                K=args.num_generations,
                max_tokens=args.max_tokens,
                temp=args.temp,
            )
            total_rollouts += len(completions)
            prompt_ids = hooks.tokenize_prompt(tokenizer, prompt_str)

            # KL advantage is ABSOLUTE (no group centering). Outcome bonus — if
            # any — is centered within the group so relative routing reward
            # drives that PPO term without cancelling the teacher direction.
            case_raw: list[tuple[list[int], list[int], float, float, float, float]] = []
            for text in completions:
                comp_ids = hooks.tokenize_completion(tokenizer, text)
                if not comp_ids:
                    continue
                if args.max_seq_len and len(prompt_ids) + len(comp_ids) > args.max_seq_len:
                    continue

                reward = hooks.score_completion(text, case, schema=args.reward_schema)
                t_lps = teacher_completion_logps(teacher, prompt_ids, comp_ids)
                s_lps = student_completion_logps_detached(model, prompt_ids, comp_ids)
                rkl = mean_reverse_kl(t_lps, s_lps)
                if rkl is None:
                    continue
                old_lp = mean_token_logprob(s_lps)
                kl_adv = kl_policy_advantage(rkl, args.kl_coef)
                outcome_raw = float(args.outcome_bonus) * float(reward)
                case_raw.append(
                    (prompt_ids, comp_ids, kl_adv, outcome_raw, old_lp, float(rkl))
                )
                mx.clear_cache()

            if case_raw:
                if args.outcome_bonus != 0.0:
                    centered_outcome = center_group_advantages(
                        [row[3] for row in case_raw]
                    )
                else:
                    centered_outcome = [0.0] * len(case_raw)
                for row, out_adv in zip(case_raw, centered_outcome):
                    prompt_ids, comp_ids, kl_adv, _out_raw, old_lp, rkl = row
                    adv = clip_advantage(kl_adv + float(out_adv), args.adv_clip)
                    triples.append((prompt_ids, comp_ids, adv, old_lp, rkl))

        # Progress line for astra HUD (matches _OPD_PROGRESS_RE)
        elapsed_s = time.time() - t_start
        print(f"  [{step}/{total_steps}]  {total_rollouts} rollouts  ({elapsed_s:.0f}s)")

        if not triples:
            print(f"Step {step}/{total_steps} SKIP (no usable completions)")
            continue

        # PPO update with reverse-KL (+ optional outcome) advantages
        model.train()
        K = len(triples)
        accumulated_grads = None
        step_loss = 0.0
        step_rkl = 0.0
        step_adv = 0.0
        for prompt_ids, comp_ids, adv, old_lp, rkl in triples:
            (loss_i, _ntoks), grads_i = loss_and_grad_fn(
                model, prompt_ids, comp_ids, float(adv), float(old_lp), args.clip_epsilon
            )
            mx.eval(loss_i, grads_i)
            step_loss += float(loss_i) / K
            step_rkl += float(rkl) / K
            step_adv += float(adv) / K
            if accumulated_grads is None:
                accumulated_grads = tree_map(lambda g: g / K, grads_i)
            else:
                accumulated_grads = tree_map(
                    lambda a, g: a + g / K, accumulated_grads, grads_i
                )
            mx.clear_cache()

        accumulated_grads, _ = optim.clip_grad_norm(
            accumulated_grads, max_norm=args.max_grad_norm
        )
        optimizer.update(model, accumulated_grads)
        mx.eval(model.parameters(), optimizer.state)
        mx.clear_cache()

        elapsed = time.time() - t_start
        # loss= PPO surrogate (astra tails it). rkl=/adv= diagnostics.
        print(
            f"Step {step}/{total_steps} loss={step_loss:.4f} "
            f"rkl={step_rkl:.4f} adv={step_adv:.4f}  {elapsed:.0f}s"
        )

        if step % args.steps_per_eval == 0:
            print(f"\n--- Eval at step {step} ---")
            model.eval()
            pass_rate, failures, blended = report_eval("Pass rate")
            if failures:
                print(f"Failures ({len(failures)}): {failures}")
            else:
                print("All routing cases PASS ✓")

            _blended_rate = (blended[0] / blended[1]) if blended else None
            _best_rate = (best_blended[0] / best_blended[1]) if best_blended else None
            _is_best = pass_rate > best_pass_rate or (
                pass_rate == best_pass_rate
                and _blended_rate is not None
                and _best_rate is not None
                and _blended_rate > _best_rate
            )
            if _is_best:
                _tie = pass_rate == best_pass_rate
                best_pass_rate = pass_rate
                best_blended = blended
                best_path = save_dir / "best"
                hooks.save_final_adapter(model, best_path, args)
                _why = " (model-routed tie, higher blended)" if _tie else ""
                print(f"New best: {pass_rate:.1%}{_why} → saved to {best_path}")
            print()
            model.train()

        if step % args.save_every == 0:
            hooks.save_final_adapter(model, save_dir, args)

    hooks.save_final_adapter(model, save_dir, args)

    print("\n=== Final Eval ===")
    model.eval()
    pass_rate, failures, blended = report_eval("Pass rate")
    if failures:
        print(f"Failures ({len(failures)}): {failures}")
    else:
        print("All routing cases PASS ✓")
    print(f"Best static pass rate during training: {best_pass_rate:.1%}")
    if best_blended:
        _bp, _bt = best_blended
        print(f"Best blended pass rate during training: {_bp / _bt:.1%} ({_bp}/{_bt})")
    print(f"Total time: {(time.time() - t_start) / 60:.1f} min")
    print(f"Total rollouts: {total_rollouts}")


# ── CLI ───────────────────────────────────────────────────────────────────────

