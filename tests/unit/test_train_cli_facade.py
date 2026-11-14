"""Astra train-center CLI entrypoints + facade -m argv."""
from __future__ import annotations

from astrarl.mlx_facade import MLXLearnerFacade
from astrarl.mlx_learners import CLI_MODULE as RL_CLI
from astratune.mlx_facade import ASTRATUNE_FACADE_CONFIG, make_mlx_facade
from astratune.mlx_learners import CLI_MODULE as TUNE_CLI


def test_astrarl_cli_module_constant() -> None:
    assert RL_CLI == "astrarl.cli"


def test_astratune_cli_module_constant() -> None:
    assert TUNE_CLI == "astratune.cli"


def test_facade_build_argv_uses_module_form() -> None:
    learner = MLXLearnerFacade(
        "mlx_grpo_routing",
        finetune_dir="/tmp/ws",
        python_bin="/usr/bin/python3",
        iters=3,
    )
    argv = learner.build_argv()
    assert argv[:4] == ["/usr/bin/python3", "-m", "astrarl.cli", "grpo"]
    assert "--workspace" in argv
    assert "/tmp/ws" in argv
    assert "--use-case" in argv
    assert "ensemble" in argv


def test_astratune_facade_build_argv() -> None:
    learner = make_mlx_facade(
        "mlx_dpo",
        finetune_dir="/tmp/ws",
        python_bin="/usr/bin/python3",
        epochs=2,
    )
    assert learner.config is ASTRATUNE_FACADE_CONFIG
    argv = learner.build_argv()
    assert argv[:4] == ["/usr/bin/python3", "-m", "astratune.cli", "dpo"]
