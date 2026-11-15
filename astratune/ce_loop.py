"""Shared masked-CE LoRA microbatch loop (distill / rft / star).

Platform-owned train step + best-checkpoint selection. Use-case runners prepare
tokenised ``(prompt_ids, comp_ids)`` pairs and supply ``report_eval`` /
adapter save callbacks (routing oracle stays use-case).
"""
from __future__ import annotations

import random
import time
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

from astratune.stdout_contract import format_step_loss, is_better_checkpoint

# report_eval(label) -> (static_rate, failures, blended_counts|None)
ReportEval = Callable[[str], tuple[float, list, Optional[tuple[int, int]]]]
SaveAdapter = Callable[[Any, Path], None]


def run_ce_lora_steps(
    model: Any,
    *,
    tokenised: Sequence[tuple[list[int], list[int]]],
    loss_and_grad_fn: Any,
    optimizer: Any,
    total_steps: int,
    batch_size: int,
    max_grad_norm: float,
    steps_per_eval: int,
    save_every: int,
    save_dir: Path,
    report_eval: ReportEval,
    save_adapter: SaveAdapter,
    initial_best_pass_rate: float = 0.0,
) -> tuple[float, Optional[tuple[int, int]]]:
    """Run the CE LoRA loop. Returns ``(best_pass_rate, best_blended)``.

    Expects baseline eval already printed by the caller. Prints
    ``Step … loss=`` and eval ``Pass rate:`` lines per stdout_contract.
    """
    import mlx.core as mx
    import mlx.optimizers as optim
    from mlx.utils import tree_map

    best_pass_rate = float(initial_best_pass_rate)
    best_blended: Optional[tuple[int, int]] = None
    t_start = time.time()
    indices = list(range(len(tokenised)))
    random.shuffle(indices)
    cursor = 0

    model.train()
    for step in range(1, total_steps + 1):
        batch = []
        for _ in range(batch_size):
            if cursor >= len(indices):
                random.shuffle(indices)
                cursor = 0
            batch.append(tokenised[indices[cursor]])
            cursor += 1

        accumulated_grads = None
        step_loss = 0.0
        for prompt_ids, comp_ids in batch:
            loss_i, grads_i = loss_and_grad_fn(model, prompt_ids, comp_ids)
            mx.eval(loss_i, grads_i)
            step_loss += float(loss_i) / len(batch)
            if accumulated_grads is None:
                accumulated_grads = tree_map(lambda g: g / len(batch), grads_i)
            else:
                accumulated_grads = tree_map(
                    lambda a, g: a + g / len(batch), accumulated_grads, grads_i,
                )
            mx.clear_cache()

        accumulated_grads, _ = optim.clip_grad_norm(
            accumulated_grads, max_norm=max_grad_norm,
        )
        optimizer.update(model, accumulated_grads)
        mx.eval(model.parameters(), optimizer.state)
        mx.clear_cache()

        print(format_step_loss(step, total_steps, step_loss, time.time() - t_start))

        if steps_per_eval and step % steps_per_eval == 0:
            print(f"\n--- Eval at step {step} ---")
            model.eval()
            pass_rate, failures, blended = report_eval("Pass rate")
            if failures:
                print(f"Failures ({len(failures)}): {failures}")
            else:
                print("All routing cases PASS ✓")

            if is_better_checkpoint(pass_rate, best_pass_rate, blended, best_blended):
                tie = pass_rate == best_pass_rate
                best_pass_rate = pass_rate
                best_blended = blended
                best_path = Path(save_dir) / "best"
                save_adapter(model, best_path)
                why = " (static tie, higher blended)" if tie else ""
                print(f"New best: {pass_rate:.1%}{why} → saved to {best_path}")
            print()
            model.train()

        if save_every and step % save_every == 0:
            save_adapter(model, Path(save_dir))

    save_adapter(model, Path(save_dir))
    return best_pass_rate, best_blended


__all__ = ["run_ce_lora_steps", "ReportEval", "SaveAdapter"]
