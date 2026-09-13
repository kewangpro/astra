"""Integration and stress tests for multi-turn reasoning and Chain-of-Thought (<think>...</think>) preservation."""
from __future__ import annotations

import json
import os
import tempfile
import pytest

from backend.trainers.base import TrainerConfig
from backend.trainers.sft_trainer import (
    SFTTrainer,
    extract_reasoning_trace,
    strip_reasoning_trace,
    format_reasoning_prompt,
    load_and_split_dataset,
    REASONING_START_TAG,
    REASONING_END_TAG,
)


@pytest.fixture
def reasoning_dataset_file(tmp_path):
    """Creates a diverse multi-turn reasoning dataset containing various CoT patterns."""
    dataset_path = tmp_path / "multi_turn_reasoning_stress.jsonl"
    records = [
        # 1. Multi-turn chat with deep mathematical reasoning in assistant turns
        {
            "id": "math_multi_turn",
            "messages": [
                {"role": "system", "content": "You are a scientific reasoning assistant."},
                {"role": "user", "content": "Calculate the derivative of f(x) = x^3 * sin(x)."},
                {
                    "role": "assistant",
                    "content": (
                        "<think>\n"
                        "To differentiate f(x) = u(x) * v(x), we apply the product rule:\n"
                        "d/dx [u*v] = u'v + uv'\n"
                        "Let u(x) = x^3, then u'(x) = 3x^2.\n"
                        "Let v(x) = sin(x), then v'(x) = cos(x).\n"
                        "Therefore, f'(x) = 3x^2 * sin(x) + x^3 * cos(x).\n"
                        "</think>\n"
                        "The derivative is f'(x) = 3x^2 * sin(x) + x^3 * cos(x)."
                    ),
                },
                {"role": "user", "content": "Now evaluate f'(pi)."},
                {
                    "role": "assistant",
                    "content": (
                        "<think>\n"
                        "We have f'(x) = 3x^2 sin(x) + x^3 cos(x).\n"
                        "At x = pi:\n"
                        "sin(pi) = 0, so 3(pi)^2 * 0 = 0.\n"
                        "cos(pi) = -1, so (pi)^3 * (-1) = -pi^3.\n"
                        "Total = 0 - pi^3 = -pi^3.\n"
                        "</think>\n"
                        "Evaluating at x = pi gives f'(pi) = -pi^3 (approx -31.006)."
                    ),
                },
            ],
        },
        # 2. Logic / planning dialogue with structured JSON and tool tags inside <think>
        {
            "id": "planning_dialogue",
            "messages": [
                {"role": "user", "content": "Plan a deployment pipeline for service alpha."},
                {
                    "role": "assistant",
                    "content": (
                        "<think>\n"
                        "1. Verify dependencies: check if Dockerfile and tests exist.\n"
                        "2. Formulate JSON task:\n"
                        "{\n"
                        '  "action": "deploy",\n'
                        '  "target": "staging",\n'
                        '  "rollback": true\n'
                        "}\n"
                        "3. Confirm security checks pass.\n"
                        "</think>\n"
                        "Here is the deployment plan for service alpha:\n"
                        "1. Run unit and integration tests.\n"
                        "2. Build and push container to staging registry.\n"
                        "3. Perform canary rollout with automated rollback."
                    ),
                },
            ],
        },
        # 3. Prompt-response pair with explicit reasoning field
        {
            "id": "field_split_pair",
            "prompt": "Explain why the sky is blue.",
            "reasoning": (
                "Rayleigh scattering occurs when sunlight reaches Earth's atmosphere.\n"
                "Shorter wavelengths (blue/violet) scatter more than longer wavelengths (red).\n"
                "Human eyes are more sensitive to blue than violet light."
            ),
            "response": "The sky appears blue due to Rayleigh scattering of sunlight by atmospheric gases.",
        },
        # 4. Embedded <think> tags in standard prompt/response format
        {
            "id": "embedded_prompt_response",
            "prompt": "What is the capital of France?",
            "response": (
                "<think>\n"
                "Geography query: capital of France.\n"
                "Country: France (Europe).\n"
                "Capital: Paris.\n"
                "</think>\n"
                "The capital of France is Paris."
            ),
        },
        # 5. Raw text completion format with multiple thought milestones
        {
            "id": "raw_text_milestones",
            "text": (
                "User: Solve 2x + 5 = 15.\n"
                "Assistant: <think>Subtract 5: 2x = 10. Divide by 2: x = 5.</think>\n"
                "The value of x is 5."
            ),
        },
        # 6. Edge case: empty reasoning tags
        {
            "id": "empty_think_tags",
            "prompt": "Say hello.",
            "response": "<think></think>Hello! How can I assist you today?",
        },
    ]

    with open(dataset_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    return str(dataset_path)


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_extract_and_format_complex_reasoning_traces():
    """Verify trace extraction with nested characters, JSON, and LaTeX."""
    raw = (
        "<think>\n"
        "Step 1: calculate \\int x dx = \\frac{1}{2}x^2 + C.\n"
        "Check JSON: {\"val\": 42, \"tags\": [\"<plan>\", \"<action>\"]}.\n"
        "</think>\n"
        "Final result: x^2 / 2 + C"
    )
    reasoning, response = extract_reasoning_trace(raw)
    assert reasoning is not None
    assert "\\int x dx" in reasoning
    assert '"tags": ["<plan>", "<action>"]' in reasoning
    assert response == "Final result: x^2 / 2 + C"

    # Test re-formatting with preserve_reasoning=True
    rebuilt = format_reasoning_prompt(
        prompt="Integrate x",
        reasoning=reasoning,
        response=response,
        preserve_reasoning=True,
    )
    assert REASONING_START_TAG in rebuilt
    assert REASONING_END_TAG in rebuilt
    assert "\\int x dx" in rebuilt
    assert "Final result: x^2 / 2 + C" in rebuilt

    # Test stripping with preserve_reasoning=False
    stripped = format_reasoning_prompt(
        prompt="Integrate x",
        reasoning=reasoning,
        response=response,
        preserve_reasoning=False,
    )
    assert REASONING_START_TAG not in stripped
    assert REASONING_END_TAG not in stripped
    assert "\\int x dx" not in stripped
    assert "Final result: x^2 / 2 + C" in stripped


def test_multi_turn_dataset_preservation(reasoning_dataset_file):
    """Verify that all conversation formats retain reasoning tokens when preserve_reasoning=True."""
    train_records, val_records = load_and_split_dataset(
        reasoning_dataset_file,
        val_split=0.33,
        seed=42,
        preserve_reasoning=True,
    )
    assert len(train_records) > 0
    assert len(val_records) > 0

    all_records = train_records + val_records

    # 1. Check multi-turn chat record
    math_rec = next(r for r in all_records if r.get("id") == "math_multi_turn")
    assistant_msgs = [m for m in math_rec["messages"] if m["role"] == "assistant"]
    assert len(assistant_msgs) == 2
    for msg in assistant_msgs:
        assert REASONING_START_TAG in msg["content"]
        assert REASONING_END_TAG in msg["content"]
        assert "product rule" in msg["content"] or "cos(pi)" in msg["content"]

    # 2. Check field split pair (reasoning + response)
    field_rec = next(r for r in all_records if r.get("id") == "field_split_pair")
    assert REASONING_START_TAG in field_rec["response"]
    assert "Rayleigh scattering" in field_rec["response"]
    assert "The sky appears blue" in field_rec["response"]

    # 3. Check embedded prompt/response record
    embedded_rec = next(r for r in all_records if r.get("id") == "embedded_prompt_response")
    assert REASONING_START_TAG in embedded_rec["response"]
    assert "Paris" in embedded_rec["response"]


def test_multi_turn_dataset_stripping(reasoning_dataset_file):
    """Verify that all conversation formats cleanly strip reasoning tokens when preserve_reasoning=False."""
    train_records, val_records = load_and_split_dataset(
        reasoning_dataset_file,
        val_split=0.33,
        seed=42,
        preserve_reasoning=False,
    )
    all_records = train_records + val_records

    # 1. Multi-turn chat
    math_rec = next(r for r in all_records if r.get("id") == "math_multi_turn")
    assistant_msgs = [m for m in math_rec["messages"] if m["role"] == "assistant"]
    for msg in assistant_msgs:
        assert REASONING_START_TAG not in msg["content"]
        assert REASONING_END_TAG not in msg["content"]
        assert "product rule" not in msg["content"]
        assert "The derivative is" in msg["content"] or "Evaluating at x = pi" in msg["content"]

    # 2. Field split pair
    field_rec = next(r for r in all_records if r.get("id") == "field_split_pair")
    assert REASONING_START_TAG not in field_rec["response"]
    assert "Shorter wavelengths" not in field_rec["response"]
    assert field_rec["response"] == "The sky appears blue due to Rayleigh scattering of sunlight by atmospheric gases."

    # 3. Embedded prompt/response
    embedded_rec = next(r for r in all_records if r.get("id") == "embedded_prompt_response")
    assert REASONING_START_TAG not in embedded_rec["response"]
    assert "Geography query" not in embedded_rec["response"]
    assert "The capital of France is Paris." in embedded_rec["response"]


def test_sft_trainer_reasoning_training_loop(reasoning_dataset_file, tmp_path):
    """Run full SFTTrainer training loop on the multi-turn reasoning dataset with preserve_reasoning=True."""
    output_dir = str(tmp_path / "sft_reasoning_output")
    cfg = TrainerConfig(
        mission_id="test-reasoning-sft",
        model_record_id="model-reasoning-01",
        data_dir=output_dir,
        hyperparameters={
            "base_model": "test-reasoning-model",
            "dataset_path": reasoning_dataset_file,
            "val_split": 0.33,
            "seed": 42,
            "batch_size": 2,
            "learning_rate": 1e-4,
            "num_train_epochs": 2,
            "eval_steps": 1,
            "save_steps": 2,
            "preserve_reasoning": True,
            "simulation_mode": True,
        },
    )

    trainer = SFTTrainer(cfg)
    trainer.run()

    # Verify trainer produced valid metrics
    metrics = trainer._latest_metrics
    assert "train_loss" in metrics
    assert "eval_loss" in metrics
    assert "perplexity" in metrics
    assert metrics["eval_loss"] < 900.0

    # Verify checkpoint metadata was created and contains preserve_reasoning configuration
    best_meta_path = os.path.join(output_dir, "checkpoints", "best", "checkpoint_metadata.json")
    assert os.path.exists(best_meta_path)
    with open(best_meta_path) as f:
        meta = json.load(f)
    assert meta["preserve_reasoning"] is True
    assert meta["eval_loss"] == trainer.best_eval_loss

