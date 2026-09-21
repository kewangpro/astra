"""Register ASTRA custom Gymnasium envs by env_id."""


def register_for_env_id(env_id: str) -> None:
    eid = env_id or ""
    key = eid.lower()
    if eid == "Snake-v0" or key == "snake":
        from envs.snake_env import register as _reg
        _reg()
    elif eid == "Tetris-v0":
        from envs.tetris_env import register as _reg
        _reg()
    elif eid in ("Game2048-v0", "2048"):
        from envs.game2048_env import register as _reg
        _reg()
    elif eid in ("MinAtar-Breakout-v0", "MinAtar-v0", "minatar", "minatar-breakout"):
        from envs.minatar_env import register as _reg
        _reg()
    elif eid in ("MinAtar-SpaceInvaders-v0", "MinAtar-Space-Invaders-v0"):
        from envs.minatar_space_invaders_env import register as _reg
        _reg()
    elif "asterix" in key:
        from envs.minatar_asterix_env import register as _reg
        _reg()
    elif eid in ("MinAtar-Asteroids-v0",) or "asteroid" in key:
        from envs.minatar_asteroids_env import register as _reg
        _reg()
    elif eid in ("MinAtar-Freeway-v0", "MinAtar-Freeway", "freeway"):
        from envs.minatar_freeway_env import register as _reg
        _reg()
    elif eid in ("MinAtar-Seaquest-v0", "MinAtar-Seaquest", "seaquest"):
        from envs.minatar_seaquest_env import register as _reg
        _reg()
    elif eid in ("GridPacMan-v0", "PacMan-v0") or "pacman" in key or "pac-man" in key:
        from envs.grid_pacman_env import register as _reg
        _reg()
    elif eid in ("MultiTurnAgentGym-v0", "AgentGym-v0", "agent-gym"):
        from envs.agent_gym import register as _reg
        _reg()
