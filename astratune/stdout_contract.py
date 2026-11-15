"""Astra Orchestrator SSH-log contracts for Mini MLX trainers.

``LoopStateMachine`` / HUD parsers expect these literal prefixes on stdout.
Platform ``run_*`` loops (and use-case runners they call) must preserve them.

Contracts
---------
* ``Baseline: X.X% (n/total)`` — pre-update static routing pass rate
* ``Pass rate: X.X% (n/total)`` — post-step / final static routing pass rate
* ``Step {i}/{n} loss={float}`` — per-step CE / distill / rft / star / DPO loss
  (DPO may insert ``epoch=…`` between step and ``loss=``; HUD ``.*?loss=`` still matches)
* ``New best: X.X%`` — DPO checkpoint improved (HUD status only)
* ``Iter {i}: Train loss {float}`` — mlx_lm.lora / SFT style (`astratune.cli sft`)
* ``Case split:`` — train/valid case counts (informational)

Blended / MCP breakdown lines are indented follow-ups and must not replace the
primary ``Pass rate:`` / ``Baseline:`` line (static numbers drive selection).
"""
from __future__ import annotations

# Documented markers — importable for tests / greps.
BASELINE_PREFIX = "Baseline:"
PASS_RATE_PREFIX = "Pass rate:"
STEP_LOSS_RE = r"Step \d+/\d+ loss="
ITER_TRAIN_LOSS_RE = r"Iter \d+: Train loss "


def format_pass_rate(label: str, rate: float, n_pass: int, n_total: int) -> str:
    """``label`` is ``Baseline`` or ``Pass rate`` (without colon)."""
    return f"{label}: {rate:.1%} ({n_pass}/{n_total})"


def format_step_loss(step: int, total_steps: int, loss: float, elapsed_s: float) -> str:
    return f"Step {step}/{total_steps} loss={loss:.4f}  {elapsed_s:.0f}s"


def is_better_checkpoint(
    pass_rate: float,
    best_pass_rate: float,
    blended: tuple[int, int] | None,
    best_blended: tuple[int, int] | None,
) -> bool:
    """Static rate selects; blended breaks ties (mission 594322b2)."""
    if pass_rate > best_pass_rate:
        return True
    if pass_rate != best_pass_rate:
        return False
    if blended is None or best_blended is None:
        return False
    blended_rate = blended[0] / blended[1]
    best_rate = best_blended[0] / best_blended[1]
    return blended_rate > best_rate
