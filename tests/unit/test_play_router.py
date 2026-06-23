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


# ── _checkpoint_algorithm ─────────────────────────────────────────────────────

def test_checkpoint_algorithm_prefers_algo_file(tmp_path):
    from backend.routers.play import _checkpoint_algorithm

    (tmp_path / "best_model_algo.txt").write_text("DQN")
    cfg = {"algorithm": "PPO", "env_kwargs": {}}
    assert _checkpoint_algorithm(str(tmp_path), cfg) == "DQN"


def test_checkpoint_algorithm_falls_back_to_config(tmp_path):
    from backend.routers.play import _checkpoint_algorithm

    cfg = {"algorithm": "PPO", "env_kwargs": {}}
    assert _checkpoint_algorithm(str(tmp_path), cfg) == "PPO"


def test_checkpoint_algorithm_empty_algo_file_falls_back(tmp_path):
    from backend.routers.play import _checkpoint_algorithm

    (tmp_path / "best_model_algo.txt").write_text("   ")
    cfg = {"algorithm": "A2C", "env_kwargs": {}}
    assert _checkpoint_algorithm(str(tmp_path), cfg) == "A2C"


def test_checkpoint_algorithm_no_config_defaults_ppo(tmp_path):
    from backend.routers.play import _checkpoint_algorithm

    assert _checkpoint_algorithm(str(tmp_path), {}) == "PPO"


# ── _tetris_viewer_grid ───────────────────────────────────────────────────────

def test_tetris_viewer_grid_returns_224_elements():
    from backend.routers.play import _tetris_viewer_grid
    from envs.tetris_env import TetrisEnv

    env = TetrisEnv()
    env.reset(seed=0)
    grid = _tetris_viewer_grid(env)
    assert len(grid) == 224


def test_tetris_viewer_grid_board_section_is_binary():
    from backend.routers.play import _tetris_viewer_grid
    from envs.tetris_env import TetrisEnv

    env = TetrisEnv()
    env.reset(seed=0)
    grid = _tetris_viewer_grid(env)
    board = grid[:200]
    assert all(v in (0, 1) for v in board)


def test_tetris_viewer_grid_current_piece_one_hot():
    from backend.routers.play import _tetris_viewer_grid
    from envs.tetris_env import TetrisEnv

    env = TetrisEnv()
    env.reset(seed=0)
    env._current_piece = 3  # S-piece
    grid = _tetris_viewer_grid(env)
    cur_oh = grid[200:207]
    assert cur_oh[3] == 1.0
    assert sum(cur_oh) == 1.0


def test_tetris_viewer_grid_next_piece_one_hot():
    from backend.routers.play import _tetris_viewer_grid
    from envs.tetris_env import TetrisEnv

    env = TetrisEnv()
    env.reset(seed=0)
    env._next_piece = 5  # J-piece
    grid = _tetris_viewer_grid(env)
    nxt_oh = grid[207:214]
    assert nxt_oh[5] == 1.0
    assert sum(nxt_oh) == 1.0


def test_tetris_viewer_grid_heights_match_empty_board():
    from backend.routers.play import _tetris_viewer_grid
    from envs.tetris_env import TetrisEnv

    env = TetrisEnv()
    env.reset(seed=0)
    # Empty board — all column heights should be 0.0
    grid = _tetris_viewer_grid(env)
    heights = grid[214:224]
    assert all(h == 0.0 for h in heights)
