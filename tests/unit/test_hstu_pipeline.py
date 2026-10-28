"""Unit tests for simplified from-scratch HSTU missions (task_type: hstu)."""
from __future__ import annotations

import asyncio
import os
from pathlib import Path

import pytest
import yaml
from unittest.mock import AsyncMock

from backend.agent.code_generator import (
    CodeGenerator,
    _ENV_RECIPE,
    _HSTU_LOCAL_WRAPPER,
    _resolve_hyperparams,
    recipe_metric_ceiling,
)
from backend.agent.code_safety_classifier import CodeSafetyClassifier
from backend.agent.lead_agent import _PLAN_SCHEMA
from backend.config import settings
from backend.loop.state_machine import (
    LoopStateMachine,
    _HR_AT_10_RE,
    _NDCG_AT_10_RE,
)
from backend.routers.missions import _infer_task_type_from_goal
from backend.sandbox.manager import _FINETUNE_REMOTE_TASK_TYPES
from backend.services.manifest_generator import _CHECKPOINT_PATTERNS, generate_manifest
from backend.trainers.hstu_train import HSTUModel, _synthetic_sequences, train as hstu_train
from backend.trainers import hstu_train as hstu_mod


def _make_provider(response: str = "pass") -> AsyncMock:
    provider = AsyncMock()
    provider.generate = AsyncMock(return_value=response)
    return provider


def test_hstu_not_finetune_remote():
    assert "hstu" not in _FINETUNE_REMOTE_TASK_TYPES
    assert "hstu" in _CHECKPOINT_PATTERNS
    assert "hstu" in LoopStateMachine._NO_CRYSTALLIZE_TASK_TYPES


def test_hstu_in_plan_schema_enum():
    enum = _PLAN_SCHEMA["properties"]["plan"]["properties"]["task_type"]["enum"]
    assert "hstu" in enum


def test_hstu_task_type_inferred_from_goal():
    assert _infer_task_type_from_goal("Train HSTU on MovieLens-1M to HR@10 0.05") == "hstu"
    assert _infer_task_type_from_goal("generative recommenders sequential model") == "hstu"


def test_hstu_ml1m_v1_recipe_structure():
    recipe_path = os.path.join(settings.recipes_path, "hstu_ml1m_v1.yaml")
    assert os.path.isfile(recipe_path)
    with open(recipe_path) as f:
        data = yaml.safe_load(f)
    assert data["task_type"] == "hstu"
    assert data["domain"] == "recsys"
    assert "hr_at_10" in data["target_metric"]
    hp = data["hyperparameters"]
    assert hp["epochs"] == 5
    assert hp["d_model"] == 64
    assert "gr_root" not in hp
    assert "gin_config" not in hp


def test_env_recipe_maps_hstu():
    assert _ENV_RECIPE["hstu"] == "hstu_ml1m_v1.yaml"
    assert recipe_metric_ceiling("hstu").get("hr_at_10") == pytest.approx(0.35)


def test_resolve_hyperparams_hstu_is_recipe_authoritative():
    hp = _resolve_hyperparams("hstu", {"epochs": 999, "d_model": 1})
    assert hp["epochs"] == 5
    assert hp["d_model"] == 64


def test_hstu_wrapper_invokes_inrepo_module():
    code = _HSTU_LOCAL_WRAPPER.format(
        checkpoint_dir="/tmp/ckpt",
        python_bin="/tmp/.venv/bin/python",
        data_dir="/tmp/data/hstu",
        epochs=5,
        batch_size=256,
        learning_rate=0.001,
        d_model=64,
        n_layers=2,
        n_heads=2,
        max_len=50,
        dropout=0.1,
        weight_decay=0.01,
        seed=42,
    )
    assert "backend.trainers.hstu_train" in code
    assert "--out" in code
    assert "gin_config" not in code
    assert "generative-recommenders" not in code
    assert "os.execv(" in code


def test_generate_training_script_hstu_uses_canned_wrapper(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "data_path", str(tmp_path))
    gen = CodeGenerator(_make_provider("SHOULD_NOT_BE_USED"))
    plan = {
        "task_type": "hstu",
        "recipe": "hstu_ml1m_v1",
        "algorithm": "HSTU",
        "target_metric": {"hr_at_10": 0.05},
        "hyperparameters": {},
    }
    path = asyncio.get_event_loop().run_until_complete(
        gen.generate_training_script("hstu-mission-1", plan)
    )
    content = open(path).read()
    assert "backend.trainers.hstu_train" in content
    assert "--gin_config_file" not in content
    gen._provider.generate.assert_not_called()


def test_hstu_metric_regexes():
    sample = "epoch 3/5 loss=1.2 HR@10: 0.0612 NDCG@10: 0.0310\nHR@10: 0.0612\n"
    hrs = [float(m.group(1)) for m in _HR_AT_10_RE.finditer(sample)]
    ndcgs = [float(m.group(1)) for m in _NDCG_AT_10_RE.finditer(sample)]
    assert 0.0612 in hrs
    assert 0.0310 in ndcgs


def test_hstu_hr_from_log(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "data_path", str(tmp_path))
    mission_id = "hstu-log-1"
    log_dir = tmp_path / "missions" / mission_id
    log_dir.mkdir(parents=True)
    (log_dir / "sandbox.log").write_text(
        "epoch 1 HR@10: 0.02\nHR@10: 0.05\nNDCG@10: 0.02\n", encoding="utf-8"
    )
    sm = LoopStateMachine.__new__(LoopStateMachine)
    sm._sandbox = None
    assert sm._hstu_hr_at_10_from_log(mission_id) == pytest.approx(0.05)
    assert sm._hstu_ndcg_at_10_from_log(mission_id) == pytest.approx(0.02)


def test_hstu_code_safety_auto_approves_module_execv():
    code = _HSTU_LOCAL_WRAPPER.format(
        checkpoint_dir="/tmp/ckpt",
        python_bin="/tmp/.venv/bin/python",
        data_dir="/tmp/data/hstu",
        epochs=5,
        batch_size=256,
        learning_rate=0.001,
        d_model=64,
        n_layers=2,
        n_heads=2,
        max_len=50,
        dropout=0.1,
        weight_decay=0.01,
        seed=42,
    )
    verdict = CodeSafetyClassifier._static_check(code)
    assert verdict.safe is True
    assert verdict.classifier == "static"


def test_hstu_manifest_uses_pt_pattern():
    m = generate_manifest("m1", "Train HSTU", "hstu", {"hr_at_10": 0.05})
    arts = [r for r in m.requirements if r.category == "artifact"]
    assert arts
    assert "*.pt" in arts[0].path_pattern


def test_hstu_model_forward_shapes():
    model = HSTUModel(n_items=20, d_model=32, n_layers=1, n_heads=2, max_len=16)
    seq = __import__("torch").randint(1, 21, (4, 16))
    seq[:, :3] = 0
    logits = model.logits_at_last(seq)
    assert logits.shape == (4, 20)


def test_hstu_synthetic_train_smoke(tmp_path):
    """One-epoch synthetic run — proves the LSTM-style loop works offline."""
    out = tmp_path / "ckpt"
    args = hstu_mod.build_argparser().parse_args([
        "--out", str(out),
        "--synthetic",
        "--epochs", "1",
        "--batch-size", "32",
        "--d-model", "32",
        "--n-layers", "1",
        "--n-heads", "2",
        "--max-len", "16",
        "--synthetic-users", "40",
        "--synthetic-items", "20",
    ])
    hstu_train(args)
    assert (out / "best.pt").is_file()
    assert (out / "last.pt").is_file()
