"""Shared MLX LoRA apply / warm-start / save helpers (AstraTune platform).

Used by AstraRL/AstraTune ``run_*`` loops and use-case hooks so adapter
I/O is not duplicated. Requires ``mlx`` / ``mlx_lm`` at call time (Mini finetune-env).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Optional, Union


def lora_config_dict(rank: int, dropout: float, scale: float) -> dict[str, float]:
    return {"rank": int(rank), "dropout": float(dropout), "scale": float(scale)}


def apply_lora(
    model: Any,
    num_layers: int,
    *,
    rank: int,
    dropout: float = 0.0,
    scale: float = 20.0,
) -> dict[str, float]:
    """Apply LoRA via ``mlx_lm.tuner.utils.linear_to_lora_layers``. Returns config."""
    from mlx_lm.tuner.utils import linear_to_lora_layers

    cfg = lora_config_dict(rank, dropout, scale)
    linear_to_lora_layers(model, int(num_layers), cfg)
    return cfg


def resolve_adapter_weights_path(adapter: Union[str, Path]) -> Path:
    """Accept a directory (…/best) or a direct ``.safetensors`` path."""
    p = Path(adapter)
    if p.is_dir():
        return p / "adapters.safetensors"
    return p


def warm_start_lora(
    model: Any,
    adapter: Union[str, Path, None],
    *,
    strict: bool = False,
    allow_missing: bool = True,
) -> Optional[Path]:
    """Load LoRA weights with ``strict=False`` (shape-tolerant warm-start).

    Returns the weights path loaded, or None if skipped (missing / cold start).
    When ``allow_missing`` is False, raises FileNotFoundError if weights absent.
    """
    if adapter is None or adapter == "" or str(adapter) in ("None", "none"):
        return None
    weights_path = resolve_adapter_weights_path(adapter)
    if not weights_path.exists():
        if allow_missing:
            return None
        raise FileNotFoundError(f"adapter weights not found: {weights_path}")
    model.load_weights(str(weights_path), strict=strict)
    return weights_path


def trainable_param_counts(model: Any) -> tuple[int, int]:
    """Return ``(n_trainable, n_total)`` parameter element counts."""
    from mlx.utils import tree_flatten

    n_trainable = sum(p.size for _, p in tree_flatten(model.trainable_parameters()))
    n_total = sum(p.size for _, p in tree_flatten(model.parameters()))
    return int(n_trainable), int(n_total)


def save_lora_checkpoint(
    model: Any,
    save_dir: Union[str, Path],
    step: Optional[int] = None,
) -> Path:
    """Save trainable LoRA tensors. If ``step`` set, use ``{step:07d}_adapters.safetensors``."""
    import mlx.core as mx
    from mlx.utils import tree_flatten

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    name = (
        f"{int(step):07d}_adapters.safetensors"
        if step is not None
        else "adapters.safetensors"
    )
    out = save_dir / name
    flat = dict(tree_flatten(model.trainable_parameters()))
    mx.save_safetensors(str(out), flat)
    return out


def write_adapter_config(
    save_dir: Union[str, Path],
    *,
    model: str,
    num_layers: int,
    rank: int,
    dropout: float,
    scale: float,
    extra: Optional[Mapping[str, Any]] = None,
) -> Path:
    """Write ``adapter_config.json`` for ``mlx_lm.server`` / ``load``."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    config: dict[str, Any] = {
        "fine_tune_type": "lora",
        "model": model,
        "num_layers": int(num_layers),
        "lora_parameters": {
            "rank": int(rank),
            "dropout": float(dropout),
            "scale": float(scale),
        },
    }
    if extra:
        config.update(dict(extra))
    path = save_dir / "adapter_config.json"
    with open(path, "w") as f:
        json.dump(config, f, indent=2)
    return path


def save_final_adapter(
    model: Any,
    save_dir: Union[str, Path],
    *,
    model_id: str,
    num_layers: int,
    rank: int,
    dropout: float,
    scale: float,
) -> Path:
    """Save ``adapters.safetensors`` + ``adapter_config.json``."""
    out = save_lora_checkpoint(model, save_dir, step=None)
    write_adapter_config(
        save_dir,
        model=model_id,
        num_layers=num_layers,
        rank=rank,
        dropout=dropout,
        scale=scale,
    )
    return out
