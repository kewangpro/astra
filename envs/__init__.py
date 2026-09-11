"""Astra Custom Game Environments."""
from envs.snake_env import SnakeEnv
from envs.tetris_env import TetrisEnv
from envs.game2048_env import Game2048Env
from envs.minatar_env import MinAtarBreakoutEnv
from envs.minatar_space_invaders_env import MinAtarSpaceInvadersEnv
from envs.minatar_asteroids_env import MinAtarAsteroidsEnv

__all__ = [
    "SnakeEnv",
    "TetrisEnv",
    "Game2048Env",
    "MinAtarBreakoutEnv",
    "MinAtarSpaceInvadersEnv",
    "MinAtarAsteroidsEnv",
]

