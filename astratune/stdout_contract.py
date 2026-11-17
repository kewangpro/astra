"""Astra Orchestrator SSH-log contracts for Mini MLX trainers.

``LoopStateMachine`` tails remote ``sandbox.log`` and:

* **Mirrors** noise-filtered lines as ``type=log`` (Event Stream narrative).
  Banners do **not** need a status regex to be visible in the HUD.
* **Extracts typed metrics** (``emit_metric``) only where charts / the goal
  loop need numbers — those prefixes must stay stable.

Platform ``run_*`` loops (and use-case runners they call) must preserve the
metric lines below. Recommended banners are still useful for humans reading
raw logs; they are no longer a HUD visibility gate.

Metric contracts (required for charts / goal)
---------------------------------------------
* ``Baseline pass rate: X.X% (n/total)`` — AstraRL GRPO routing baseline
  (``train_grpo_routing``; HUD ``_PASS_RATE_RE`` is line-anchored with optional
  ``Baseline `` — must not match ``Held-out pass rate:``)
* ``Pass rate: X.X% (n/total)`` — mid-step / final static routing pass rate
  (goal / live ``pass_rate`` metric; also mirrored as ``log``)
* ``Step {i}/{n} loss={float}`` — per-step CE / distill / rft / star / DPO loss
  (DPO may insert ``epoch=…`` between step and ``loss=``; HUD ``.*?loss=`` still matches;
  loss may be negative for OPD reverse-KL)
* ``Step {i}/{n} | loss={float} | …`` — GRPO / RLCD / agrpo pipe form
  (``_GRPO_LOSS_RE``; loss may be negative sentinel, e.g. ``-1.0000``)
* ``=== Epoch N/M done  avg_loss={float} ===`` — DPO epoch summary (``avg_loss`` series)
* ``Iter {i}: Train loss {float}`` / ``Val loss`` — mlx_lm.lora / SFT style

Recommended banners (mirrored as log; not a visibility gate)
------------------------------------------------------------
* ``Baseline: X.X% (n/total)`` — DPO / distill-style pre-update static rate
* ``Held-out pass rate: X.X% (n/total)`` — informational split (never goal metric)
* ``--- Baseline eval (step N) ---`` / ``--- Eval at step N ---`` /
  ``=== Final Eval ===``
* ``Case pool:`` / ``Starting GRPO training:`` / ``Focused pool:`` /
  ``DPO training:`` / ``N pairs ready for training`` / ``Loading model:`` /
  ``Warm-start from:`` / ``Warm-start (reference):`` /
  ``Caching reference log-probs…`` / collect ``[i/n]  k pairs  (Ts)``
* ``New best: X.X%`` — checkpoint improved mid-train
* ``Best adapter ensured at {path}`` — DPO / GRPO / STaR / RFT final ensure of ``<save-dir>/best/``
  (required for end-only ``steps_per_eval`` smokes that never hit mid-train eval)
* ``Eval every N steps`` — authoritative SPE for Astra's reliability gate
  (``_dpo_run_diagnostics`` prefers this over plan/recipe HP)
* ``N examples ready for training`` — distill / rft / star reliability gate
* ``Case split:`` — train/valid case counts

Blended / MCP breakdown lines are indented follow-ups and must not replace the
primary ``Pass rate:`` / ``Baseline:`` / ``Baseline pass rate:`` line
(static numbers drive selection).
"""
from __future__ import annotations

# Documented markers — importable for tests / greps.
BASELINE_PREFIX = "Baseline:"
BASELINE_PASS_RATE_PREFIX = "Baseline pass rate:"
PASS_RATE_PREFIX = "Pass rate:"
STEP_LOSS_RE = r"Step \d+/\d+ .*?loss=-?[\d.]+"
GRPO_STEP_LOSS_RE = r"Step \s*\d+/\d+\s*\|\s*loss=-?[\d.]+"
ITER_TRAIN_LOSS_RE = r"Iter \d+: Train loss "


def format_pass_rate(label: str, rate: float, n_pass: int, n_total: int) -> str:
    """``label`` is ``Baseline``, ``Baseline pass rate``, or ``Pass rate`` (no trailing colon)."""
    return f"{label}: {rate:.1%} ({n_pass}/{n_total})"


def format_step_loss(step: int, total_steps: int, loss: float, elapsed_s: float) -> str:
    return f"Step {step}/{total_steps} loss={loss:.4f}  {elapsed_s:.0f}s"


def format_grpo_step_loss(
    step: int, total_steps: int, loss: float, *, baseline: float | None = None
) -> str:
    """GRPO/RLCD pipe contract used by ``_GRPO_LOSS_RE``."""
    base = f"Step {step}/{total_steps} | loss={loss:.4f}"
    if baseline is not None:
        base += f" | baseline={baseline:.3f}"
    return base


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
