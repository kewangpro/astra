"""Epoch 9 — AstraGym / AstraRL factories and resolve helpers."""
from __future__ import annotations

import pytest

import astragym
import astrarl


def test_infer_gym_id_env_rl_snake():
    assert astragym.infer_gym_id(task_type="rl", env_id="Snake-v0") == "gymnasium:Snake-v0"


def test_infer_gym_id_agent_gym_vector():
    assert (
        astragym.infer_gym_id(task_type="rl", env_id="MultiTurnAgentGym-v0")
        == "agent_gym_vector"
    )


def test_infer_gym_id_agrpo_modes():
    assert astragym.infer_gym_id(task_type="agrpo", reward_mode="agent_gym") == "agent_gym_text"
    assert astragym.infer_gym_id(task_type="agrpo", reward_mode="e2e") == "e2e_stubs"


def test_infer_gym_id_routing():
    assert astragym.infer_gym_id(task_type="grpo") == "routing_prompt"
    assert astragym.infer_gym_id(task_type="opd") == "routing_prompt"


def test_infer_learner_ids():
    assert astrarl.infer_learner_id(task_type="rl", algorithm="PPO") == "sb3_ppo"
    assert astrarl.infer_learner_id(task_type="rl", algorithm="DQN") == "sb3_dqn"
    assert astrarl.infer_learner_id(task_type="agrpo") == "mlx_grpo_traj"
    assert astrarl.infer_learner_id(task_type="grpo") == "mlx_grpo_routing"
    assert astrarl.infer_learner_id(task_type="opd") == "mlx_opd"


def test_make_breakout_gym():
    g = astragym.make("gymnasium:MinAtar-Breakout-v0")
    assert g.gym_id == "gymnasium:MinAtar-Breakout-v0"
    obs, info = g.reset(seed=0)
    assert obs is not None
    g.close()


def test_make_agent_gym_vector():
    g = astragym.make("agent_gym_vector")
    assert g.gym_id == "agent_gym_vector"
    obs, _ = g.reset(seed=0)
    assert obs is not None
    g.close()


def test_sb3_dqn_predict_smoke():
    g = astragym.make("gymnasium:MinAtar-Breakout-v0")
    learner = astrarl.make("sb3_dqn", g, learning_starts=100_000)
    obs, _ = g.reset(seed=1)
    action = learner.predict(obs)
    assert int(action) in (0, 1, 2)
    g.close()


def test_mlx_learner_descriptor():
    d = astrarl.make("mlx_grpo_traj", reward_mode="e2e")
    assert d["entrypoint"] == "agent_grpo_train.py"
    assert d["gym_id"] == "e2e_stubs"
    assert astrarl.entrypoint_for("mlx_grpo_routing") == "grpo_train.py"
