"""SFT training loop — platform-owned (astratune); wraps mlx_lm.lora."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

def resolve_data_path(raw_path: str) -> str:
    """Resolve data path across common Mac Mini locations."""
    if os.path.exists(raw_path):
        return os.path.abspath(raw_path)
    candidates = [
        os.path.join("/tmp/astra", raw_path),
        os.path.join("/tmp/astra/data", raw_path),
        os.path.join("/Users/kewang/finetune", raw_path),
        os.path.join("/Users/kewang/finetune/data_ft", raw_path),
    ]
    for c in candidates:
        if os.path.exists(c):
            return os.path.abspath(c)
    return os.path.abspath(raw_path)


def prepare_dataset_dir(
    data_path: str,
    save_dir: str,
    val_split: float,
    seed: int,
    preserve_reasoning: bool,
    batch_size: int = 4,
) -> tuple[str, int]:
    """Ensure data is in a directory containing train.jsonl and valid.jsonl for mlx_lm.
    Returns (data_dir, effective_batch_size) ensuring effective_batch_size <= min(train_size, val_size).
    """
    if os.path.isdir(data_path):
        train_file = os.path.join(data_path, "train.jsonl")
        val_file = os.path.join(data_path, "valid.jsonl")
        if os.path.exists(train_file):
            n_train = sum(1 for line in open(train_file, "r", encoding="utf-8") if line.strip())
            n_val = sum(1 for line in open(val_file, "r", encoding="utf-8") if line.strip()) if os.path.exists(val_file) else n_train
            eff_bs = max(1, min(batch_size, n_train, n_val))
            return data_path, eff_bs

    # data_path is a file
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset file not found: {data_path}")

    records = []
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    if not records:
        raise ValueError(f"Dataset is empty: {data_path}")

    formatted = []
    for rec in records:
        if "messages" in rec:
            formatted.append({"messages": rec["messages"]})
        elif "text" in rec:
            formatted.append({"text": rec["text"]})
        else:
            prompt = rec.get("prompt", "")
            reasoning = rec.get("reasoning") or rec.get("thinking")
            response = rec.get("response") or rec.get("output", rec.get("completion", ""))
            if preserve_reasoning and reasoning:
                completion = f"<think>\n{str(reasoning).strip()}\n</think>\n{str(response).strip()}"
            else:
                completion = str(response).strip()
            formatted.append({"prompt": str(prompt).strip(), "completion": completion})

    # Deterministic split
    rng = random.Random(seed)
    shuffled = list(formatted)
    rng.shuffle(shuffled)

    n_total = len(shuffled)
    val_count = int(n_total * val_split) if val_split > 0.0 else 0
    if val_split > 0.0 and n_total >= 2:
        if n_total >= batch_size * 2:
            val_count = max(batch_size, val_count)
        else:
            val_count = max(1, val_count)

    if val_count >= n_total:
        val_count = max(0, n_total - 1)

    val_records = shuffled[:val_count]
    train_records = shuffled[val_count:]

    # Compute effective batch size: mlx_lm requires len(dataset) >= batch_size
    eff_batch_size = batch_size
    min_split = len(train_records)
    if val_records:
        min_split = min(min_split, len(val_records))
    if eff_batch_size > min_split:
        eff_batch_size = max(1, min_split)
        print(f"Adjusting batch_size from {batch_size} to {eff_batch_size} to match dataset split sizes (train={len(train_records)}, val={len(val_records)})")

    out_data_dir = os.path.join(save_dir, "_data")
    os.makedirs(out_data_dir, exist_ok=True)

    with open(os.path.join(out_data_dir, "train.jsonl"), "w", encoding="utf-8") as f:
        for r in train_records:
            f.write(json.dumps(r) + "\n")

    if val_records:
        with open(os.path.join(out_data_dir, "valid.jsonl"), "w", encoding="utf-8") as f:
            for r in val_records:
                f.write(json.dumps(r) + "\n")

    print(f"Prepared SFT dataset: {len(train_records)} train, {len(val_records)} valid in {out_data_dir} (effective batch_size={eff_batch_size})")
    return out_data_dir, eff_batch_size


def run_sft_train(args) -> None:
    os.makedirs(args.save_dir, exist_ok=True)

    # 1. Resolve and prepare dataset
    resolved_data = resolve_data_path(args.data)
    data_dir, effective_batch_size = prepare_dataset_dir(
        resolved_data,
        args.save_dir,
        val_split=args.val_split,
        seed=args.seed,
        preserve_reasoning=args.preserve_reasoning,
        batch_size=args.batch_size,
    )

    # 2. Run mlx_lm.lora
    import mlx_lm.lora as mlx_lora

    lora_args = argparse.Namespace(
        model=args.model,
        train=True,
        test=False,
        data=data_dir,
        adapter_path=args.save_dir,
        fine_tune_type="lora",
        optimizer="adam",
        optimizer_config={"adam": {}, "adamw": {}, "muon": {}, "sgd": {}, "adafactor": {}},
        batch_size=effective_batch_size,
        iters=args.iters,
        val_batches=args.val_batches,
        learning_rate=args.learning_rate,
        steps_per_report=args.steps_per_report,
        steps_per_eval=args.steps_per_eval,
        save_every=args.save_every,
        max_seq_length=args.max_seq_length,
        num_layers=args.num_layers,
        grad_checkpoint=args.grad_checkpoint,
        grad_accumulation_steps=1,
        clear_cache_threshold=0,
        lr_schedule=None,
        lora_parameters={
            "rank": args.lora_rank,
            "scale": args.lora_scale,
            "dropout": args.lora_dropout,
        },
        mask_prompt=args.mask_prompt,
        resume_adapter_file=args.resume_adapter_file,
        test_batches=25,
        config=None,
        report_to=None,
        project_name=None,
        seed=args.seed,
    )

    print(f"Starting MLX SFT training: model={args.model} iters={args.iters} lr={args.learning_rate} rank={args.lora_rank} batch_size={effective_batch_size}")
    sys.stdout.flush()
    mlx_lora.run(lora_args)

    # 3. Create best/ copy for Astra requirement manifest check (checkpoints/*/)
    best_dir = os.path.join(args.save_dir, "best")
    os.makedirs(best_dir, exist_ok=True)
    for fname in ("adapters.safetensors", "adapter_config.json"):
        src = os.path.join(args.save_dir, fname)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(best_dir, fname))

    # Write checkpoint_metadata.json for Astra evaluator / specialist
    meta = {
        "iteration": args.iters,
        "base_model": args.model,
        "val_split": args.val_split,
        "num_layers": args.num_layers,
        "lora_rank": args.lora_rank,
        "lora_scale": args.lora_scale,
        "lora_dropout": args.lora_dropout,
        "saved_at": datetime.now(timezone.utc).isoformat(),
    }
    meta_path = os.path.join(args.save_dir, "checkpoint_metadata.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    shutil.copy2(meta_path, os.path.join(best_dir, "checkpoint_metadata.json"))

    print(f"SFT fine-tuning complete. Best adapter saved to {best_dir}")
    sys.stdout.flush()


