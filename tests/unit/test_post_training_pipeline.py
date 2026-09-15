"""
Unit tests for the 3-Stage Post-Training Pipeline (SFT -> DPO -> GRPO)
running under a single mission with task_type: "post-training".
"""
from __future__ import annotations

import os
import pytest
import yaml
from unittest.mock import AsyncMock, MagicMock, patch

from backend.config import settings
from backend.sandbox.manager import _FINETUNE_REMOTE_TASK_TYPES
from backend.services.manifest_generator import generate_manifest, _CHECKPOINT_PATTERNS
from backend.agent.code_generator import (
    CodeGenerator,
    finetune_checkpoint_dir,
    finetune_checkpoint_dir_relative,
    _load_recipe_for_env,
)
from backend.agent.code_safety_classifier import CodeSafetyClassifier
from backend.services.preflight import PreflightChecker
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


def test_post_training_task_type_registered():
    """Verify post-training is recognized as a remote finetune task type and checkpoint pattern."""
    assert "post-training" in _FINETUNE_REMOTE_TASK_TYPES
    assert "post-training" in _CHECKPOINT_PATTERNS


def test_ensemble_post_training_v1_recipe_structure():
    """Verify canonical ensemble_post_training_v1.yaml recipe structure and stages."""
    recipe_path = os.path.join(settings.recipes_path, "ensemble_post_training_v1.yaml")
    assert os.path.isfile(recipe_path), f"Recipe file missing: {recipe_path}"

    with open(recipe_path) as f:
        data = yaml.safe_load(f)

    assert data["name"] == "ensemble_post_training_v1"
    assert data["task_type"] == "post-training"
    assert data["domain"] == "NLP"
    assert data["target_metric"] == {"pass_rate": 0.80}
    assert data.get("metric_ceiling") == {"pass_rate": 0.91}

    stages = data.get("stages", [])
    assert len(stages) == 3

    # Stage 1: SFT
    assert stages[0]["stage"] == 1
    assert stages[0]["task"] == "sft"
    assert stages[0]["target_metric"] == {"eval_loss": 0.80}

    # Stage 2: DPO
    assert stages[1]["stage"] == 2
    assert stages[1]["task"] == "dpo"
    assert stages[1]["target_metric"] == {"pass_rate": 0.70}

    # Stage 3: GRPO
    assert stages[2]["stage"] == 3
    assert stages[2]["task"] == "grpo"
    assert stages[2]["target_metric"] == {"pass_rate": 0.80}


@pytest.mark.asyncio
async def test_dispatch_post_training_recipe():
    """Verify dispatch_recipe initializes current_plan with stages, stage_index, and active_task_type."""
    db = _make_db(record=None)
    mock_loop = MagicMock()
    mock_loop.run = AsyncMock()

    with patch("backend.routers.agent._build_loop", return_value=mock_loop), \
         patch("backend.routers.agent._running_tasks", {}):
        resp = await dispatch_recipe("ensemble_post_training_v1", db=db)
        assert resp.status == "dispatched"
        assert resp.task_type == "post-training"

        # Check the Mission added to DB
        added_mission = db.add.call_args[0][0]
        assert added_mission.task_type == "post-training"
        plan = added_mission.current_plan
        assert plan["task_type"] == "post-training"
        assert plan["stage_index"] == 0
        assert plan["active_task_type"] == "sft"
        assert len(plan["stages"]) == 3
        assert plan["stages"][0]["task"] == "sft"
        assert plan["stages"][1]["task"] == "dpo"
        assert plan["stages"][2]["task"] == "grpo"


def test_stage_scoped_checkpoint_dir():
    """Verify finetune_checkpoint_dir produces distinct stage-scoped paths for each stage."""
    mission_id = "f299cdd6-d070-40ef-96a1-ccfc1727ab2a"

    plan_stage1 = {
        "recipe": "ensemble_post_training_v1",
        "stage_index": 0,
        "active_task_type": "sft",
        "hyperparameters": {"finetune_dir": "/Users/kewang/finetune"},
    }
    dir_sft = finetune_checkpoint_dir("post-training", plan_stage1, mission_id, iteration=0)
    assert dir_sft == "/Users/kewang/finetune/adapters/astra_f299cdd6_stage1_sft_iter0"

    plan_stage2 = {
        "recipe": "ensemble_post_training_v1",
        "stage_index": 1,
        "active_task_type": "dpo",
        "hyperparameters": {"finetune_dir": "/Users/kewang/finetune"},
    }
    dir_dpo = finetune_checkpoint_dir("post-training", plan_stage2, mission_id, iteration=1)
    assert dir_dpo == "/Users/kewang/finetune/adapters/astra_f299cdd6_stage2_dpo_iter1"

    plan_stage3 = {
        "recipe": "ensemble_post_training_v1",
        "stage_index": 2,
        "active_task_type": "grpo",
        "hyperparameters": {"finetune_dir": "/Users/kewang/finetune"},
    }
    dir_grpo = finetune_checkpoint_dir("post-training", plan_stage3, mission_id, iteration=2)
    assert dir_grpo == "/Users/kewang/finetune/adapters/astra_f299cdd6_stage3_grpo_iter2"

    # Also check relative helper
    rel = finetune_checkpoint_dir_relative(mission_id, iteration=1, stage_index=1, active_task_type="dpo")
    assert rel == "adapters/astra_f299cdd6_stage2_dpo_iter1/best"


def test_code_safety_classifier_auto_approves_sft_dpo_grpo():
    """Verify CodeSafetyClassifier auto-approves wrappers targeting sft, dpo, and grpo scripts."""
    sft_script = (
        "import sys\nimport os\n"
        "os.chdir('/Users/kewang/finetune')\n"
        "os.execv('/Users/kewang/finetune-env/bin/python', ['python', 'sft_train.py', '--model', 'foo'])\n"
    )
    verdict_sft = CodeSafetyClassifier._static_check(sft_script)
    assert verdict_sft.safe is True

    dpo_script = (
        "import sys\nimport os\n"
        "os.chdir('/Users/kewang/finetune')\n"
        "os.execv('/Users/kewang/finetune-env/bin/python', ['python', 'dpo_train.py', '--adapter', 'bar'])\n"
    )
    verdict_dpo = CodeSafetyClassifier._static_check(dpo_script)
    assert verdict_dpo.safe is True

    grpo_script = (
        "import sys\nimport os\n"
        "os.chdir('/Users/kewang/finetune')\n"
        "os.execv('/Users/kewang/finetune-env/bin/python', ['python', 'grpo_train.py', '--adapter', 'baz'])\n"
    )
    verdict_grpo = CodeSafetyClassifier._static_check(grpo_script)
    assert verdict_grpo.safe is True


def test_preflight_checks_all_post_training_scripts():
    """Verify PreflightChecker._check_remote_script checks sft, dpo, and grpo train scripts."""
    with patch("backend.config.settings.sandbox_host", "mac-mini.local"), \
         patch("backend.agent.code_generator._resolve_hyperparams", return_value={"finetune_dir": "/Users/kewang/finetune"}), \
         patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(stdout="yes\n", returncode=0)
        results = PreflightChecker._check_remote_script("post-training")

        assert len(results) == 3
        script_names = [r["name"] for r in results]
        assert "remote_script_sft_train" in script_names
        assert "remote_script_dpo_train" in script_names
        assert "remote_script_grpo_train" in script_names
        assert all(r["passed"] for r in results)


@pytest.mark.asyncio
async def test_codegen_sft_remote_wrapper_for_post_training_stage1(tmp_path):
    """Verify CodeGenerator emits the SFT remote wrapper when active_task_type is sft."""
    provider = MagicMock()
    codegen = CodeGenerator(provider=provider)

    plan = {
        "task_type": "post-training",
        "stage_index": 0,
        "active_task_type": "sft",
        "recipe": "ensemble_post_training_v1",
        "hyperparameters": {
            "finetune_dir": "/Users/kewang/finetune",
            "python_bin": "/Users/kewang/finetune-env/bin/python",
            "base_model": "mlx-community/gemma-3-12b-it-4bit",
            "dataset_path": "data_routing",
            "iters": 150,
        }
    }

    with patch("backend.config.settings.data_path", str(tmp_path)), \
         patch("backend.config.settings.sandbox_host", "mac-mini.local"):
        script_path = await codegen.generate_training_script("testmission123", plan, current_iteration=0)
        with open(script_path) as f:
            code = f.read()

        assert "sft_train.py" in code
        assert "os.execv" in code
        assert "--model" in code
        assert "astra_testmiss_stage1_sft_iter0" in code


def test_manifest_generation_for_stage():
    """Verify generate_manifest correctly builds requirements for each stage milestone."""
    # Stage 1: SFT with eval_loss
    m_sft = generate_manifest(
        mission_id="m1", goal="Test post-training", task_type="sft", target_metric={"eval_loss": 0.80},
    )
    metric_reqs = [r for r in m_sft.requirements if r.check_type == "metric_threshold"]
    assert len(metric_reqs) == 1
    assert metric_reqs[0].metric_name == "eval_loss"
    assert metric_reqs[0].threshold == 0.80
    assert metric_reqs[0].operator == "<="

    # Stage 2: DPO with pass_rate
    m_dpo = generate_manifest(
        mission_id="m1", goal="Test post-training", task_type="dpo", target_metric={"pass_rate": 0.70},
    )
    metric_reqs = [r for r in m_dpo.requirements if r.check_type == "metric_threshold"]
    assert len(metric_reqs) == 1
    assert metric_reqs[0].metric_name == "pass_rate"
    assert metric_reqs[0].threshold == 0.70
    assert metric_reqs[0].operator == ">="
