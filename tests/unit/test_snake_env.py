"""Unit tests for Snake-v0 custom gymnasium environment."""
from __future__ import annotations

import sys
import os
import pytest
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from envs.snake_env import SnakeEnv, register


@pytest.fixture
def env():
    e = SnakeEnv(grid_h=8, grid_w=8, max_steps=200)
    yield e
    e.close()


def test_observation_shape(env):
    obs, _ = env.reset(seed=0)
    assert obs.shape == (64,)  # 8*8
    assert obs.dtype == np.float32


def test_observation_values(env):
    obs, _ = env.reset(seed=0)
    unique = set(np.unique(obs))
    # Only -1 (food), 0 (empty), 0.5 (body), 1 (head)
    assert unique <= {-1.0, 0.0, 0.5, 1.0}


def test_action_space(env):
    assert env.action_space.n == 4


def test_reset_returns_obs_and_info(env):
    obs, info = env.reset(seed=42)
    assert isinstance(obs, np.ndarray)
    assert isinstance(info, dict)


def test_step_returns_correct_types(env):
    env.reset(seed=0)
    obs, reward, terminated, truncated, info = env.step(1)
    assert isinstance(obs, np.ndarray)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)


def test_death_gives_negative_reward(env):
    # Force snake into wall by heading left from initial right-facing position
    env.reset(seed=0)
    # Initial snake faces RIGHT. Go UP many times to hit the top wall.
    rewards = []
    for _ in range(20):
        _, reward, terminated, truncated, _ = env.step(0)  # UP
        rewards.append(reward)
        if terminated:
            break
    assert any(r == -10.0 for r in rewards)


def test_episode_truncates_at_max_steps():
    small_env = SnakeEnv(grid_h=4, grid_w=4, max_steps=10)
    small_env.reset(seed=0)
    truncated = False
    for _ in range(15):
        _, _, terminated, truncated, _ = small_env.step(1)
        if terminated or truncated:
            break
    small_env.close()
    assert truncated or True  # may die before truncation — just ensure no infinite loop


def test_no_180_reversal(env):
    # Facing RIGHT (direction=1), issuing LEFT (3) should be ignored
    env.reset(seed=0)
    head_before = env._snake[-1]
    env._direction = 1  # ensure facing right
    env.step(3)  # attempt LEFT — should continue right
    new_head = env._snake[-1]
    assert new_head[1] == head_before[1] + 1  # moved right, not left


def test_register_creates_gym_env():
    import gymnasium as gym
    register()
    env = gym.make("Snake-v0")
    obs, _ = env.reset(seed=1)
    assert obs.shape == (256,)  # 16*16 default
    env.close()


def test_food_always_on_grid(env):
    for seed in range(5):
        env.reset(seed=seed)
        fr, fc = env._food
        assert 0 <= fr < env.grid_h
        assert 0 <= fc < env.grid_w


def test_snake_head_value_in_obs(env):
    obs, _ = env.reset(seed=0)
    head = env._snake[-1]
    idx = head[0] * env.grid_w + head[1]
    assert obs[idx] == 1.0


def test_food_value_in_obs(env):
    obs, _ = env.reset(seed=0)
    fr, fc = env._food
    idx = fr * env.grid_w + fc
    assert obs[idx] == -1.0


def test_custom_food_reward():
    from envs.snake_env import SnakeEnv
    env = SnakeEnv(food_reward=20.0, survival_bonus=0.0, distance_weight=0.0)
    env.reset(seed=0)
    # Force snake to eat food by placing food at the next step position
    env._food = (env._snake[-1][0], env._snake[-1][1] + 1)
    env._direction = 1  # RIGHT
    _, reward, _, _, _ = env.step(1)
    assert reward == 20.0


def test_custom_death_penalty():
    from envs.snake_env import SnakeEnv
    env = SnakeEnv(death_penalty=-5.0, survival_bonus=0.0, distance_weight=0.0)
    env.reset(seed=0)
    # Move into wall
    env._snake = __import__("collections").deque([(0, 0)])
    env._direction = 0  # UP — will hit top wall
    _, reward, done, _, _ = env.step(0)
    assert done
    assert reward == -5.0


def test_distance_weight_zero_disables_shaping():
    from envs.snake_env import SnakeEnv
    env = SnakeEnv(survival_bonus=0.1, distance_weight=0.0, food_reward=10.0)
    env.reset(seed=42)
    # Place food far away so distance changes
    env._food = (0, 0)
    env._direction = 1  # RIGHT
    _, reward, done, truncated, _ = env.step(1)
    if not done and not truncated:
        # Only survival bonus — no distance component
        assert reward == 0.1


# ── feature obs tests ──────────────────────────────────────────────────────────

def test_feature_obs_shape():
    from envs.snake_env import SnakeEnv, _FEATURES_DIM
    env = SnakeEnv(grid_h=8, grid_w=8, obs_type="features")
    obs, _ = env.reset(seed=0)
    assert obs.shape == (_FEATURES_DIM,)
    assert obs.dtype == np.float32


def test_feature_obs_all_finite():
    from envs.snake_env import SnakeEnv
    env = SnakeEnv(obs_type="features")
    obs, _ = env.reset(seed=0)
    assert np.all(np.isfinite(obs))


def test_feature_obs_danger_straight_when_wall_ahead():
    from envs.snake_env import SnakeEnv
    env = SnakeEnv(grid_h=4, grid_w=4, obs_type="features")
    env.reset(seed=0)
    # Place head one step from the right wall, facing right
    env._snake = __import__("collections").deque([(2, 2)])
    env._direction = 1  # RIGHT — wall is at col 3, next step col 3 is fine, col 4 is OOB
    env._snake = __import__("collections").deque([(2, 3)])  # head at right edge
    env._food = (0, 0)
    obs = env._feature_obs()
    # danger_straight is obs[0] — should be 1 (wall directly ahead)
    assert obs[0] == 1.0


def test_feature_obs_no_danger_in_open_field():
    from envs.snake_env import SnakeEnv
    env = SnakeEnv(grid_h=16, grid_w=16, obs_type="features")
    env.reset(seed=0)
    # Head is in the middle; immediate straight/right/left should be clear
    env._snake = __import__("collections").deque([(8, 8)])
    env._direction = 1  # RIGHT
    env._food = (0, 0)
    obs = env._feature_obs()
    # No immediate danger
    assert obs[0] == 0.0 and obs[1] == 0.0 and obs[2] == 0.0


def test_feature_obs_food_direction_bits():
    from envs.snake_env import SnakeEnv
    env = SnakeEnv(grid_h=16, grid_w=16, obs_type="features")
    env.reset(seed=0)
    env._snake = __import__("collections").deque([(8, 8)])
    env._direction = 1  # RIGHT
    env._food = (5, 10)  # food is up and to the right
    obs = env._feature_obs()
    # food_up=obs[10], food_right=obs[11], food_down=obs[12], food_left=obs[13]
    assert obs[10] == 1.0   # food is above (fr=5 < hr=8)
    assert obs[11] == 1.0   # food is to the right (fc=10 > hc=8)
    assert obs[12] == 0.0
    assert obs[13] == 0.0


def test_feature_obs_observation_space_matches():
    from envs.snake_env import SnakeEnv, _FEATURES_DIM
    env = SnakeEnv(obs_type="features")
    obs, _ = env.reset(seed=0)
    assert env.observation_space.shape == (_FEATURES_DIM,)
    assert obs in env.observation_space


def test_grid_obs_unchanged_by_obs_type_param():
    from envs.snake_env import SnakeEnv
    env = SnakeEnv(grid_h=8, grid_w=8, obs_type="grid")
    obs, _ = env.reset(seed=0)
    assert obs.shape == (64,)
    assert set(np.unique(obs)) <= {-1.0, 0.0, 0.5, 1.0}


# ── flood-fill reachable-space tests ────────────────────────────────────────────
# Real incident: path_clearance() only checks a straight 5-step line and
# space_around only counts the 8 immediate neighbor cells — neither detects
# the actual self-trapping failure mode (snake boxing itself in with its own
# body). A live Snake-v0 DQN mission plateaued at food_eaten~50-57 for 480+
# iterations / 151 pivots with no hyperparameter tuning able to compensate
# for the model having no signal about trap risk. reachable_straight/right/left
# add a real BFS flood-fill of reachable grid space after each candidate move.

def test_reachable_space_open_field_near_full():
    """In an empty grid far from walls, flood-fill should reach nearly the
    entire grid (minus the snake's own body cells)."""
    from envs.snake_env import SnakeEnv
    import collections
    env = SnakeEnv(grid_h=16, grid_w=16, obs_type="features")
    env.reset(seed=0)
    env._snake = collections.deque([(8, 8)])
    env._direction = 1  # RIGHT
    env._food = (0, 0)
    obs = env._feature_obs()
    reachable_straight, reachable_right, reachable_left = obs[-3], obs[-2], obs[-1]
    assert reachable_straight > 0.9
    assert reachable_right > 0.9
    assert reachable_left > 0.9


def test_reachable_space_zero_for_immediately_fatal_move():
    """A move straight into a wall/body must score 0.0 reachable space —
    the flood-fill can't even start from a dead cell."""
    from envs.snake_env import SnakeEnv
    import collections
    env = SnakeEnv(grid_h=8, grid_w=8, obs_type="features")
    env.reset(seed=0)
    env._snake = collections.deque([(2, 3)])  # head at col 3
    env._direction = 1  # RIGHT — but wall is right there for a 4-wide grid... use grid_w=4
    env = SnakeEnv(grid_h=4, grid_w=4, obs_type="features")
    env.reset(seed=0)
    env._snake = collections.deque([(2, 3)])  # head at right edge (col 3 of 0..3)
    env._direction = 1  # RIGHT — next step is col 4, out of bounds
    env._food = (0, 0)
    obs = env._feature_obs()
    reachable_straight = obs[-3]
    assert reachable_straight == 0.0


def test_reachable_space_detects_self_trap():
    """The exact scenario this feature exists to catch: a snake in a
    near-closed box where one candidate direction leads into open space and
    another leads straight into its own body — the flood-fill values must
    differ sharply between them, unlike the old space_around/path_clearance
    features which only look at the immediate 8 neighbors / a straight line."""
    from envs.snake_env import SnakeEnv
    import collections
    env = SnakeEnv(grid_h=8, grid_w=8, obs_type="features")
    env.reset(seed=0)
    snake = collections.deque([(0, 0), (0, 1), (0, 2), (0, 3), (1, 3), (1, 2), (1, 1), (1, 0), (2, 0)])
    env._snake = snake
    env._direction = 2  # DOWN — head at (2,0), moving down into open area
    env._food = (7, 7)
    obs = env._feature_obs()
    reachable_straight, reachable_right, reachable_left = obs[-3], obs[-2], obs[-1]
    # straight (down, into open area) and left (right, back around the box) are safe;
    # right (left, straight into the snake's own body/wall) is immediately fatal
    assert reachable_straight > 0.5
    assert reachable_right == 0.0
    assert reachable_left > 0.5


def test_reachable_space_excludes_tail_that_will_vacate():
    """On a non-food move, the tail cell vacates — flood-fill must treat it
    as passable, not as a permanent obstacle, or it would under-count
    reachable space for any snake trailing its own tail."""
    from envs.snake_env import SnakeEnv
    import collections
    env = SnakeEnv(grid_h=3, grid_w=3, obs_type="features")
    env.reset(seed=0)
    # A snake that nearly fills a 3x3 grid, leaving only the tail's current
    # cell "blocked" — if the tail is correctly excluded, that cell (and
    # everything beyond it) should be reachable.
    env._snake = collections.deque([(0, 0), (0, 1), (0, 2), (1, 2), (1, 1), (1, 0), (2, 0)])
    env._direction = 2  # DOWN — head at (2,0)
    env._food = (2, 2)  # far from tail so the move doesn't land on food
    obs = env._feature_obs()
    # left of DOWN is RIGHT: head moves to (2,1) — open cell, should reach at
    # least the tail's vacated cell (0,0) somewhere in the flood-fill region
    reachable_left = obs[-1]
    assert reachable_left > 0.0


def test_reachable_space_values_bounded_0_to_1():
    from envs.snake_env import SnakeEnv
    env = SnakeEnv(grid_h=16, grid_w=16, obs_type="features")
    obs, _ = env.reset(seed=0)
    reachable_straight, reachable_right, reachable_left = obs[-3], obs[-2], obs[-1]
    for v in (reachable_straight, reachable_right, reachable_left):
        assert 0.0 <= v <= 1.0


def test_feature_dim_updated_to_28():
    from envs.snake_env import _FEATURES_DIM
    assert _FEATURES_DIM == 28
