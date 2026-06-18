"""Unit tests for backend/routers/play.py — train_config loading and algo dispatch."""
from __future__ import annotations

import json
import os
import pytest


def test_load_train_config_reads_json(tmp_path):
    from backend.routers.play import _load_train_config

    cfg = {"algorithm": "DQN", "env_id": "Snake-v0", "env_kwargs": {"food_reward": 20.0}}
    (tmp_path / "train_config.json").write_text(json.dumps(cfg))

    result = _load_train_config(str(tmp_path))
    assert result["algorithm"] == "DQN"
    assert result["env_kwargs"]["food_reward"] == 20.0


def test_load_train_config_returns_defaults_when_missing(tmp_path):
    from backend.routers.play import _load_train_config

    result = _load_train_config(str(tmp_path))
    assert result["algorithm"] == "PPO"
    assert result["env_kwargs"] == {}


def test_load_train_config_empty_env_kwargs(tmp_path):
    from backend.routers.play import _load_train_config

    cfg = {"algorithm": "PPO", "env_id": "CartPole-v1", "env_kwargs": {}}
    (tmp_path / "train_config.json").write_text(json.dumps(cfg))

    result = _load_train_config(str(tmp_path))
    assert result["env_kwargs"] == {}


def test_get_algo_class_ppo():
    from backend.routers.play import _get_algo_class
    from stable_baselines3 import PPO

    assert _get_algo_class("PPO") is PPO


def test_get_algo_class_dqn():
    from backend.routers.play import _get_algo_class
    from stable_baselines3 import DQN

    assert _get_algo_class("DQN") is DQN


def test_get_algo_class_a2c():
    from backend.routers.play import _get_algo_class
    from stable_baselines3 import A2C

    assert _get_algo_class("A2C") is A2C


def test_get_algo_class_unknown_falls_back_to_ppo():
    from backend.routers.play import _get_algo_class
    from stable_baselines3 import PPO

    # Unknown algorithm → defaults to PPO
    assert _get_algo_class("UNKNOWN") is PPO


def test_get_algo_class_case_insensitive():
    from backend.routers.play import _get_algo_class
    from stable_baselines3 import DQN

    assert _get_algo_class("dqn") is DQN
