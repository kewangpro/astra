"""Unit tests for SFTTrainer (backend/trainers/sft_trainer.py)."""
from __future__ import annotations

import json
import math
import os
import pytest

from backend.trainers.base import TrainerConfig
from backend.trainers.sft_trainer import (
    SFTTrainer,
    extract_reasoning_trace,
    strip_reasoning_trace,
    format_reasoning_prompt,
    load_and_split_dataset,
)


# ── Reasoning / CoT helper tests ──────────────────────────────────────────────

def test_extract_reasoning_trace():
    raw = "<think>\nStep 1: calculate x\nStep 2: verify\n</think>\nFinal answer: 42"
    reasoning, response = extract_reasoning_trace(raw)
    assert reasoning == "Step 1: calculate x\nStep 2: verify"
    assert response == "Final answer: 42"


def test_extract_reasoning_trace_no_tags():
    raw = "Direct response without thinking block."
    reasoning, response = extract_reasoning_trace(raw)
    assert reasoning is None
    assert response == "Direct response without thinking block."


def test_strip_reasoning_trace():
    raw = "Prefix <think>internal monologue</think> and answer."
    cleaned = strip_reasoning_trace(raw)
    assert "<think>" not in cleaned
    assert "internal monologue" not in cleaned
    assert "Prefix  and answer." == cleaned or "Prefix and answer." in cleaned


def test_format_reasoning_prompt_preserve_true():
    prompt = "What is 2+2?"
    reasoning = "2 + 2 equals 4 via arithmetic."
    response = "The answer is 4."
    system = "You are a helpful reasoning model."

    formatted = format_reasoning_prompt(
        prompt=prompt,
        reasoning=reasoning,
        response=response,
        system=system,
        preserve_reasoning=True,
    )
    assert "System: You are a helpful reasoning model." in formatted
    assert "User: What is 2+2?" in formatted
    assert "<think>" in formatted
    assert "2 + 2 equals 4 via arithmetic." in formatted
    assert "</think>" in formatted
    assert "The answer is 4." in formatted


def test_format_reasoning_prompt_preserve_false():
    prompt = "What is 2+2?"
    reasoning = "internal steps should be removed"
    response = "The answer is 4."

    formatted = format_reasoning_prompt(
        prompt=prompt,
        reasoning=reasoning,
        response=response,
        preserve_reasoning=False,
    )
    assert "<think>" not in formatted
    assert "internal steps" not in formatted
    assert "The answer is 4." in formatted


# ── Dataset loading and splitting tests ───────────────────────────────────────

def test_load_and_split_dataset_deterministic(tmp_path):
    data_file = tmp_path / "train.jsonl"
    records = [{"id": i, "prompt": f"Q{i}", "response": f"A{i}"} for i in range(20)]
    with open(data_file, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    train_1, val_1 = load_and_split_dataset(str(data_file), val_split=0.2, seed=42)
    assert len(train_1) == 16
    assert len(val_1) == 4

    train_ids = {r["id"] for r in train_1}
    val_ids = {r["id"] for r in val_1}
    # Strict held-out split: no overlap between train and val
    assert train_ids.isdisjoint(val_ids)
    assert len(train_ids | val_ids) == 20

    # Determinism with same seed
    train_2, val_2 = load_and_split_dataset(str(data_file), val_split=0.2, seed=42)
    assert [r["id"] for r in train_1] == [r["id"] for r in train_2]
    assert [r["id"] for r in val_1] == [r["id"] for r in val_2]


def test_load_and_split_dataset_separate_val_file(tmp_path):
    train_file = tmp_path / "train.jsonl"
    val_file = tmp_path / "val.jsonl"

    train_records = [{"id": f"t_{i}", "prompt": f"Q{i}"} for i in range(10)]
    val_records = [{"id": f"v_{i}", "prompt": f"QV{i}"} for i in range(5)]

    with open(train_file, "w") as f:
        for r in train_records:
            f.write(json.dumps(r) + "\n")
    with open(val_file, "w") as f:
        for r in val_records:
            f.write(json.dumps(r) + "\n")

    train_res, val_res = load_and_split_dataset(
        dataset_path=str(train_file),
        val_dataset_path=str(val_file),
    )
    assert len(train_res) == 10
    assert len(val_res) == 5
    assert all(r["id"].startswith("t_") for r in train_res)
    assert all(r["id"].startswith("v_") for r in val_res)


def test_load_and_split_dataset_reasoning_preservation(tmp_path):
    data_file = tmp_path / "chat.jsonl"
    records = [
        {
            "id": 1,
            "messages": [
                {"role": "user", "content": "solve math"},
                {"role": "assistant", "content": "<think>steps</think>solution"},
            ],
        },
        {
            "id": 2,
            "reasoning": "thought process",
            "response": "solution 2",
        },
    ]
    with open(data_file, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    # With preserve_reasoning=True
    train_p, val_p = load_and_split_dataset(str(data_file), val_split=0.5, preserve_reasoning=True)
    all_p = train_p + val_p
    rec1 = next(r for r in all_p if r["id"] == 1)
    rec2 = next(r for r in all_p if r["id"] == 2)
    assert "<think>steps</think>" in rec1["messages"][1]["content"]
    assert "<think>" in rec2["response"]
    assert "thought process" in rec2["response"]

    # With preserve_reasoning=False
    train_s, val_s = load_and_split_dataset(str(data_file), val_split=0.5, preserve_reasoning=False)
    all_s = train_s + val_s
    rec1_s = next(r for r in all_s if r["id"] == 1)
    rec2_s = next(r for r in all_s if r["id"] == 2)
    assert "<think>" not in rec1_s["messages"][1]["content"]
    assert "<think>" not in rec2_s["response"]
    assert rec2_s["response"] == "solution 2"


def test_load_and_split_dataset_edge_cases(tmp_path):
    # Single sample dataset
    single_file = tmp_path / "single.jsonl"
    with open(single_file, "w") as f:
        f.write(json.dumps({"id": 0}) + "\n")
    train, val = load_and_split_dataset(str(single_file), val_split=0.2)
    assert len(train) == 1
    assert len(val) == 0

    # Non-existent file raises FileNotFoundError
    with pytest.raises(FileNotFoundError):
        load_and_split_dataset(str(tmp_path / "non_existent.jsonl"))

    # None path returns empty lists
    t_empty, v_empty = load_and_split_dataset(None)
    assert t_empty == []
    assert v_empty == []


# ── SFTTrainer class execution & telemetry tests ─────────────────────────────

def test_sft_trainer_config_parsing(tmp_path):
    config = TrainerConfig(
        mission_id="mission-sft-01",
        model_record_id="model-rec-01",
        data_dir=str(tmp_path),
        hyperparameters={
            "base_model": "meta-llama/Llama-3.1-8B",
            "val_split": 0.15,
            "lora_r": 32,
            "lora_alpha": 64,
            "learning_rate": 1e-4,
            "max_seq_length": 2048,
            "preserve_reasoning": True,
            "simulation_mode": True,
        },
    )
    trainer = SFTTrainer(config)
    assert trainer.base_model == "meta-llama/Llama-3.1-8B"
    assert trainer.val_split == 0.15
    assert trainer.lora_r == 32
    assert trainer.lora_alpha == 64
    assert trainer.learning_rate == 1e-4
    assert trainer.max_seq_length == 2048
    assert trainer.preserve_reasoning is True
    assert trainer.simulation_mode is True


def test_sft_trainer_simulation_training_loop(tmp_path):
    data_file = tmp_path / "data.jsonl"
    with open(data_file, "w") as f:
        for i in range(12):
            f.write(json.dumps({"prompt": f"Q{i}", "reasoning": f"T{i}", "response": f"A{i}"}) + "\n")

    config = TrainerConfig(
        mission_id="mission-sft-sim",
        model_record_id="model-sft-sim",
        data_dir=str(tmp_path),
        hyperparameters={
            "dataset_path": str(data_file),
            "val_split": 0.25,
            "batch_size": 2,
            "num_train_epochs": 2,
            "eval_steps": 5,
            "save_steps": 5,
            "simulation_mode": True,
        },
    )
    trainer = SFTTrainer(config)
    trainer.run()

    # Check telemetry
    telemetry_file = tmp_path / "telemetry.jsonl"
    assert telemetry_file.exists()
    events = [json.loads(line) for line in telemetry_file.read_text().splitlines() if line]
    metric_names = {e["name"] for e in events}
    assert "train_loss" in metric_names
    assert "eval_loss" in metric_names
    assert "perplexity" in metric_names

    # Checkpoints
    assert os.path.exists(trainer.checkpoint_dir)
    assert trainer.best_checkpoint_path is not None
    assert os.path.exists(trainer.best_checkpoint_path)

    # Check metadata
    meta_path = os.path.join(trainer.best_checkpoint_path, "checkpoint_metadata.json")
    assert os.path.exists(meta_path)
    with open(meta_path) as f:
        meta = json.load(f)
    assert meta["mission_id"] == "mission-sft-sim"
    assert "eval_loss" in meta
    assert meta["val_split"] == 0.25


def test_sft_trainer_save_and_load_checkpoint(tmp_path):
    config = TrainerConfig(
        mission_id="mission-sft-ckpt",
        model_record_id="model-sft-ckpt",
        data_dir=str(tmp_path),
        hyperparameters={"simulation_mode": True},
    )
    trainer = SFTTrainer(config)
    trainer._iteration = 42
    trainer.best_eval_loss = 1.15
    ckpt_path = trainer.save_checkpoint()

    assert os.path.exists(ckpt_path)
    meta_path = os.path.join(ckpt_path, "checkpoint_metadata.json")
    assert os.path.exists(meta_path)

    # Load into fresh trainer
    trainer2 = SFTTrainer(config)
    assert trainer2._iteration == 0
    trainer2.load_checkpoint(ckpt_path)
    assert trainer2._iteration == 42
    assert trainer2.best_eval_loss == 1.15
