"""Astra custom game environments.

Environment classes are loaded lazily so importing ``envs.register`` does not
import and register every game. Besides reducing startup side effects, this
keeps targeted env registration testable when a single env module is mocked.
"""
from importlib import import_module

__all__ = [
    "SnakeEnv",
    "TetrisEnv",
    "Game2048Env",
    "MinAtarBreakoutEnv",
    "MinAtarSpaceInvadersEnv",
    "MinAtarAsteroidsEnv",
    "MinAtarFreewayEnv",
    "MinAtarSeaquestEnv",
    "MinAtarAsterixEnv",
    "GridPacManEnv",
]

_ENV_CLASS_MODULES = {
    "SnakeEnv": "envs.snake_env",
    "TetrisEnv": "envs.tetris_env",
    "Game2048Env": "envs.game2048_env",
    "MinAtarBreakoutEnv": "envs.minatar_env",
    "MinAtarSpaceInvadersEnv": "envs.minatar_space_invaders_env",
    "MinAtarAsteroidsEnv": "envs.minatar_asteroids_env",
    "MinAtarFreewayEnv": "envs.minatar_freeway_env",
    "MinAtarSeaquestEnv": "envs.minatar_seaquest_env",
    "MinAtarAsterixEnv": "envs.minatar_asterix_env",
    "GridPacManEnv": "envs.grid_pacman_env",
}


def __getattr__(name: str):
    module_name = _ENV_CLASS_MODULES.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(module_name), name)
    globals()[name] = value
    return value

