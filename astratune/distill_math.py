"""Knowledge-distillation step loss — same masked CE as SFT on teacher completions."""
from __future__ import annotations

from typing import Any

from astratune.sft_math import full_sequence_ce_loss_mlx, masked_ce_loss_mlx


def distill_masked_loss(model: Any, prompt_ids: list[int], comp_ids: list[int]) -> Any:
    """Teacher-completion CE with prompt tokens masked (default distill behaviour)."""
    return masked_ce_loss_mlx(model, prompt_ids, comp_ids)


def distill_full_sequence_loss(model: Any, prompt_ids: list[int], comp_ids: list[int]) -> Any:
    """Teacher-completion CE over the full tokenised sequence."""
    return full_sequence_ce_loss_mlx(model, prompt_ids, comp_ids)