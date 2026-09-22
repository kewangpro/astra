"""Unit tests for Grid Pac-Man (envs/grid_pacman_env.py)."""
import gymnasium as gym
import numpy as np
import pytest

from envs.grid_pacman_env import GridPacManEnv, register


@pytest.fixture(autouse=True)
def ensure_registered():
    register()


def test_pacman_spaces():
    env = GridPacManEnv()
    assert env.action_space.n == 5
    assert env.observation_space.shape == (400,)
    assert env.observation_space.dtype == np.float32


def test_pacman_reset_and_walls():
    env = GridPacManEnv()
    obs, info = env.reset(seed=42)
    assert obs.shape == (400,)
    assert env._player_r == 7
    assert env._player_c == 4
    assert (0, 0) in env._walls
    assert info["pellets_left"] > 0


def test_pacman_wall_blocks():
    env = GridPacManEnv()
    env.reset(seed=0)
    env._ghosts = []
    env._player_r, env._player_c = 1, 1
    env.step(3)  # UP into wall row 0
    assert env._player_r == 1


def test_pacman_faces_move_direction():
    env = GridPacManEnv()
    env.reset(seed=0)
    env._ghosts = []
    assert env._player_dir == 0  # default RIGHT
    assert env._info()["player_dir"] == 0

    env.step(1)  # LEFT from (7, 4) along the open corridor
    assert env._player_c == 3
    assert env._player_dir == 2

    env.step(2)  # RIGHT back to spawn
    assert env._player_c == 4
    assert env._player_dir == 0

    env.step(3)  # UP into the center gap
    assert env._player_r == 6
    assert env._player_dir == 3

    env.step(4)  # DOWN
    assert env._player_r == 7
    assert env._player_dir == 1

    env.step(0)  # NOOP keeps last facing
    assert env._player_dir == 1

    env._player_r, env._player_c = 1, 1
    env._player_dir = 0
    env.step(3)  # UP into wall — blocked, heading unchanged
    assert env._player_r == 1
    assert env._player_dir == 0


def test_pacman_pellet_eat():
    env = GridPacManEnv()
    env.reset(seed=0)
    env._ghosts = []
    pos = (env._player_r, env._player_c)
    env._pellets.add(pos)
    _, reward, terminated, _, info = env.step(0)
    assert not terminated
    assert reward == 1.0
    assert pos not in env._pellets
    assert info["pellets_eaten"] == 1


def test_pacman_ghost_death():
    env = GridPacManEnv(death_penalty=-3.0)
    env.reset(seed=0)
    env._power_timer = 0
    env._pellets.discard((env._player_r, env._player_c))
    env._power.discard((env._player_r, env._player_c))
    env._ghosts = [{"r": env._player_r, "c": env._player_c, "home": (1, 4)}]
    _, reward, terminated, _, _ = env.step(0)
    assert terminated
    assert reward == -3.0


def test_pacman_eat_frightened_ghost():
    env = GridPacManEnv()
    env.reset(seed=0)
    env._power_timer = 10
    env._pellets.discard((env._player_r, env._player_c))
    env._power.discard((env._player_r, env._player_c))
    env._ghosts = [{"r": env._player_r, "c": env._player_c, "home": (1, 4)}]
    _, reward, terminated, _, info = env.step(0)
    assert not terminated
    assert reward == 5.0
    assert info["ghosts_eaten"] == 1
    assert env._ghosts[0]["r"] == 1


def test_ghost_prefers_up_on_manhattan_tie():
    """Leftover-pellet dance: ghost at (7,4), Pac-Man in the (2,5) alcove.
    LEFT/RIGHT and UP can tie on Manhattan; UP must win so the ghost climbs
    instead of sliding sideways forever."""
    env = GridPacManEnv()
    env.reset(seed=0)
    env._power_timer = 0
    env._player_r, env._player_c = 2, 5
    env._ghosts = [{"r": 7, "c": 4, "home": (1, 4), "last": None}]
    env._move_ghosts()
    assert (env._ghosts[0]["r"], env._ghosts[0]["c"]) == (6, 4)


def test_ghost_does_not_reverse_when_other_options_exist():
    env = GridPacManEnv()
    env.reset(seed=0)
    env._power_timer = 0
    env._player_r, env._player_c = 7, 8
    env._ghosts = [{"r": 7, "c": 4, "home": (1, 4), "last": (7, 5)}]
    env._move_ghosts()
    assert (env._ghosts[0]["r"], env._ghosts[0]["c"]) != (7, 5)


def test_ghost_chase_does_not_oscillate_in_corridor():
    env = GridPacManEnv()
    env.reset(seed=0)
    env._power_timer = 0
    env._player_r, env._player_c = 2, 4
    env._ghosts = [{"r": 7, "c": 4, "home": (1, 4), "last": None}]
    path = []
    for _ in range(5):
        env._move_ghosts()
        path.append((env._ghosts[0]["r"], env._ghosts[0]["c"]))
    # Climbs the center shaft toward row 2; no left-right bounce.
    assert path == [(6, 4), (5, 4), (4, 4), (3, 4), (2, 4)]
    cols = {c for _, c in path}
    assert cols == {4}


def test_ghosts_step_every_other_tick():
    env = GridPacManEnv()
    env.reset(seed=0)
    env._power_timer = 0
    env._player_r, env._player_c = 2, 4
    env._ghosts = [{"r": 7, "c": 4, "home": (1, 4), "last": None}]
    env.step(0)
    assert (env._ghosts[0]["r"], env._ghosts[0]["c"]) == (7, 4)
    env.step(0)
    assert (env._ghosts[0]["r"], env._ghosts[0]["c"]) == (6, 4)


def test_pacman_gym_make():
    env = gym.make("GridPacMan-v0")
    obs, _ = env.reset(seed=2)
    assert obs.shape == (400,)
    env.close()


def test_pacman_recipe_hps():
    from backend.agent.code_generator import _resolve_hyperparams
    result = _resolve_hyperparams("GridPacMan-v0", {}, algorithm="DQN")
    assert result["learning_rate"] == 0.00025
    assert result["total_timesteps"] == 300000
