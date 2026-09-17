"""
Unit tests for the STaR (Self-Taught Reasoner) Data Bootstrapping Pipeline
running under task_type: "star".
"""
from __future__ import annotations

import os
import pytest
import yaml
from unittest.mock import AsyncMock, MagicMock, patch

from backend.config import settings
from backend.sandbox.manager import _FINETUNE_REMOTE_TASK_TYPES
from backend.services.manifest_generator import generate_manifest, _CHECKPOINT_PATTERNS
from backend.loop.state_machine import LoopStateMachine
from backend.agent.code_generator import (
    CodeGenerator,
    finetune_checkpoint_dir,
    _load_recipe_for_env,
    _clamp_finetune_pivot_hp,
)
from backend.agent.code_safety_classifier import CodeSafetyClassifier
from backend.services.preflight import PreflightChecker
from backend.routers.missions import _infer_task_type_from_goal
from backend.routers.recipes import dispatch_recipe


def _make_db(record=None):
    db = AsyncMock()
    db.get = AsyncMock(return_value=record)
    scalars = MagicMock()
    scalars.first = MagicMock(return_value=record)
    result = MagicMock()
    result.scalars = MagicMock(return_value=scalars)
    db.execute = AsyncMock(return_value=result)
    db.add = MagicMock()
    db.commit = AsyncMock()
    db.refresh = AsyncMock()
    return db


def test_star_task_type_registered():
    """Verify star is recognized as a remote finetune task type and checkpoint pattern."""
    assert "star" in _FINETUNE_REMOTE_TASK_TYPES
    assert "star" in _CHECKPOINT_PATTERNS
    assert "star" in LoopStateMachine._NO_CRYSTALLIZE_TASK_TYPES


def test_star_task_type_inferred_from_goal():
    """Verify _infer_task_type_from_goal identifies STaR keywords."""
    assert _infer_task_type_from_goal("STaR autonomous reasoning flywheel for Ensemble") == "star"
    assert _infer_task_type_from_goal("Train a self-taught reasoner for conductor routing") == "star"
    assert _infer_task_type_from_goal("Apply rationalization fine-tuning to reach 85% pass rate") == "star"


def test_ensemble_star_v1_recipe_structure():
    """Verify canonical ensemble_star_v1.yaml recipe structure and hyperparameters."""
    recipe_path = os.path.join(settings.recipes_path, "ensemble_star_v1.yaml")
    assert os.path.isfile(recipe_path), f"Recipe file missing: {recipe_path}"

    with open(recipe_path) as f:
        data = yaml.safe_load(f)

    assert data["name"] == "ensemble_star_v1"
    assert data["task_type"] == "star"
    assert data["domain"] == "routing"
    assert data["target_metric"] == {"pass_rate": 0.85}
    assert data.get("metric_ceiling") == {"pass_rate": 0.95}

    hp = data.get("hyperparameters", {})
    assert hp["base_model"] == "mlx-community/gemma-3-12b-it-4bit"
    assert hp["k_samples"] == 8
    assert hp["temp"] == 1.0
    assert hp["rationalize_temp"] == 0.8
    assert hp["num_layers"] == 4
    assert hp["lora_rank"] == 8
    assert hp["lora_scale"] == 5.0
    assert hp["lora_dropout"] == 0.1
    assert hp["iters"] == 100
    assert hp["val_split"] == 0.15


def test_star_pivot_ranges():
    """Verify hyperparameter clamping for STaR pivot adjustments."""
    clamped = _clamp_finetune_pivot_hp("star", {
        "k_samples": 32,
        "temp": 2.5,
        "rationalize_temp": 0.2,
        "learning_rate": 0.05,
    })
    assert clamped["k_samples"] == 16
    assert clamped["temp"] == 1.5
    assert clamped["rationalize_temp"] == 0.5
    assert clamped["learning_rate"] == 5e-4


@pytest.mark.asyncio
async def test_star_code_generation_remote_wrapper():
    """Verify CodeGenerator generates deterministic os.execv wrapper for star on remote host."""
    generator = CodeGenerator(provider=MagicMock())
    mission_id = "test-star-1234-5678-abcdef"
    plan = {
        "task_type": "star",
        "recipe": "ensemble_star_v1",
        "hyperparameters": {
            "base_model": "mlx-community/gemma-3-12b-it-4bit",
            "finetune_dir": "/Users/kewang/finetune",
            "python_bin": "/Users/kewang/finetune-env/bin/python",
            "prompt_template": "backend/prompts/conductor_min.md",
            "k_samples": 8,
            "temp": 1.0,
            "rationalize_temp": 0.8,
            "max_tokens": 256,
            "eval_max_tokens": 256,
            "num_layers": 4,
            "lora_rank": 8,
            "lora_scale": 5.0,
            "lora_dropout": 0.1,
            "iters": 100,
            "batch_size": 2,
            "learning_rate": 0.0001,
            "steps_per_eval": 25,
            "save_every": 25,
            "max_seq_len": 2048,
            "val_split": 0.15,
            "reward_schema": "v1",
            "routing_only": True,
        },
        "target_metric": {"pass_rate": 0.85},
    }

    with patch.object(settings, "sandbox_host", "mac-mini.local"):
        script_path = await generator.generate_training_script(
            mission_id=mission_id,
            plan=plan,
            current_iteration=0,
            warm_start_adapter="adapters/prior_best",
        )

    assert os.path.isfile(script_path)
    with open(script_path) as f:
        code = f.read()

    assert "star_train.py" in code
    assert 'os.chdir("/Users/kewang/finetune")' in code
    assert 'os.execv("/Users/kewang/finetune-env/bin/python"' in code
    assert '"--k-samples", "8"' in code
    assert '"--rationalize-temp", "0.8"' in code
    assert '"--adapter", "adapters/prior_best"' in code
    assert '"--routing-only"' in code


def test_star_code_safety_classifier():
    """Verify star_train.py execution scripts are auto-approved by CodeSafetyClassifier."""
    code = (
        "import sys\nimport os\n"
        "os.chdir('/Users/kewang/finetune')\n"
        "os.execv('/Users/kewang/finetune-env/bin/python', ['/Users/kewang/finetune-env/bin/python', '/Users/kewang/finetune/star_train.py'])\n"
    )
    verdict = CodeSafetyClassifier._static_check(code)
    assert verdict.safe is True
    assert "auto-approved" in verdict.reason
    assert verdict.classifier == "static"


def test_star_preflight_check():
    """Verify PreflightChecker queries star_train.py for star task type."""
    with patch("backend.config.settings.sandbox_host", "mac-mini.local"), \
         patch("backend.agent.code_generator._resolve_hyperparams", return_value={"finetune_dir": "/Users/kewang/finetune"}), \
         patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(stdout="yes\n", returncode=0)
        results = PreflightChecker._check_remote_script("star")

    assert len(results) == 1
    assert results[0]["name"] == "remote_script_star_train"
    assert results[0]["passed"] is True


@pytest.mark.asyncio
async def test_dispatch_star_recipe():
    """Verify dispatching ensemble_star_v1 creates a valid star mission."""
    db = _make_db(record=None)
    mock_loop = MagicMock()
    mock_loop.run = AsyncMock()

    with patch("backend.routers.agent._build_loop", return_value=mock_loop), \
         patch("backend.routers.agent._running_tasks", {}):
        resp = await dispatch_recipe("ensemble_star_v1", db=db)
        assert resp.status == "dispatched"
        assert resp.task_type == "star"

        added_mission = db.add.call_args[0][0]
        assert added_mission.task_type == "star"
        assert added_mission.target_metric == {"pass_rate": 0.85}
