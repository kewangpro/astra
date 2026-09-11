"""
WebSocket endpoint for running trained RL model inference and streaming
game frames to the mission HUD.

WS /ws/missions/{id}/play?env_id=Snake-v0

Streams JSON frames:
  {"type": "frame", "grid": [...256 floats...], "episode": 1, "step": 42,
   "episode_reward": 73.4, "done": false}
  {"type": "episode_end", "episode": 1, "total_reward": 73.4}
  {"type": "error", "message": "..."}
"""
from __future__ import annotations

import asyncio
import json
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from backend.config import settings
from backend.logging_config import get_logger

logger = get_logger(__name__)

router = APIRouter()

_EXECUTOR = ThreadPoolExecutor(max_workers=2, thread_name_prefix="snake-play")

# Maps algorithm name → SB3 class import path
_SB3_ALGO_MAP = {
    "PPO": ("stable_baselines3", "PPO"),
    "DQN": ("stable_baselines3", "DQN"),
    "SAC": ("stable_baselines3", "SAC"),
    "A2C": ("stable_baselines3", "A2C"),
}


def _load_train_config(ckpt_dir: str) -> dict:
    """Read train_config.json written by CodeGenerator; fall back to PPO defaults."""
    config_path = os.path.join(ckpt_dir, "train_config.json")
    if os.path.exists(config_path):
        with open(config_path) as f:
            return json.load(f)
    return {"algorithm": "PPO", "env_id": "", "env_kwargs": {}}


def _checkpoint_algorithm(ckpt_dir: str, cfg: dict) -> str:
    """Return the algorithm that actually saved best_model.zip.

    Prefers best_model_algo.txt (written by the training callback at save time)
    over train_config.json (which reflects the most recently *generated* plan and
    may differ when the previous algorithm's best_model.zip was never beaten).
    """
    algo_file = os.path.join(ckpt_dir, "best_model_algo.txt")
    if os.path.exists(algo_file):
        algo = open(algo_file).read().strip()
        if algo:
            return algo
    return cfg.get("algorithm", "PPO")


def _get_algo_class(algorithm: str):
    """Return the SB3 algorithm class for the given name."""
    import importlib
    module_name, cls_name = _SB3_ALGO_MAP.get(algorithm.upper(), ("stable_baselines3", "PPO"))
    module = importlib.import_module(module_name)
    return getattr(module, cls_name)


def _tetris_viewer_grid(base_env) -> list:
    """Build the 224-element viewer grid from live TetrisEnv state.

    The training obs is a compact 4-feature vector, but TetrisPlayer.tsx
    expects the old 224-element layout so it can render the board visually:
      [0..199]   20×10 board (0/1)
      [200..206] current-piece one-hot (7 pieces)
      [207..213] next-piece one-hot (7 pieces)
      [214..223] column heights (10)
    """
    board = base_env._board.flatten().tolist()          # 200
    cur_oh = [0.0] * 7
    nxt_oh = [0.0] * 7
    cur, nxt = base_env._current_piece, base_env._next_piece
    if 0 <= cur < 7:
        cur_oh[cur] = 1.0
    if 0 <= nxt < 7:
        nxt_oh[nxt] = 1.0
    heights = [float(h) for h in base_env._column_heights()]  # 10
    return board + cur_oh + nxt_oh + heights                   # 224


def _run_episode_actor_critic(model, env) -> tuple[list[dict], float]:
    """Run one episode with a PyTorch Actor-Critic / Lookahead model using get_next_states()."""
    import torch
    import numpy as np
    obs, _ = env.reset()
    frames = []
    episode_reward = 0.0
    step = 0
    done = False
    truncated = False
    base_env = env.unwrapped
    is_2048 = hasattr(base_env, "_board") and hasattr(base_env, "_max_tile")
    action_names_2048 = {0: "UP", 1: "DOWN", 2: "LEFT", 3: "RIGHT"}

    while not done and not truncated:
        next_states = base_env.get_next_states()
        q_vals = {}
        action_probs = {}
        entropy = None
        if next_states:
            vals = {}
            with torch.no_grad():
                best_action, best_val = None, float("-inf")
                for act, st in next_states.items():
                    val = model(torch.tensor(st, dtype=torch.float32).unsqueeze(0))
                    if isinstance(val, tuple):
                        val = val[1]  # critic head
                    v = float(val.squeeze())
                    vals[act] = v
                    act_name = action_names_2048.get(act, str(act)) if is_2048 else f"A{act}"
                    q_vals[act_name] = round(v, 2)
                    if v > best_val:
                        best_val, best_action = v, act
            if vals:
                arr = np.array(list(vals.values()), dtype=np.float32)
                exp_v = np.exp(arr - np.max(arr))
                probs = exp_v / np.sum(exp_v)
                entropy = float(-np.sum(probs * np.log(np.clip(probs, 1e-8, 1.0))))
                for (act, _), p in zip(vals.items(), probs):
                    act_name = action_names_2048.get(act, str(act)) if is_2048 else f"A{act}"
                    action_probs[act_name] = round(float(p), 3)
            action = best_action
        else:
            action = 0

        piece_before_step = getattr(base_env, "_current_piece", None)
        obs, reward, done, truncated, info = env.step(action)
        episode_reward += float(reward)
        step += 1

        chosen_name = (action_names_2048.get(action, str(action)) if is_2048 else f"A{action}")

        if is_2048:
            frames.append({
                "type": "frame",
                "grid": base_env.get_viewer_grid(),
                "step": step,
                "episode_reward": round(episode_reward, 2),
                "done": bool(done or truncated),
                "score": int(base_env._score),
                "max_tile": int(base_env._max_tile),
                "q_values": q_vals,
                "action_probs": action_probs,
                "entropy": round(entropy, 3) if entropy is not None else None,
                "selected_action": chosen_name,
            })
        else:
            # Tetris
            lines_cleared = int(getattr(base_env, "_lines_cleared_last", 0))
            if lines_cleared > 0:
                cleared_rows = getattr(base_env, "_last_cleared_rows", [])
                pre_clear = getattr(base_env, "_pre_clear_board", None)
                if cleared_rows and pre_clear is not None:
                    cur_oh = [0.0] * 7
                    if piece_before_step is not None and 0 <= piece_before_step < 7:
                        cur_oh[piece_before_step] = 1.0
                    nxt_oh = [0.0] * 7
                    nxt = getattr(base_env, "_current_piece", -1)
                    if 0 <= nxt < 7:
                        nxt_oh[nxt] = 1.0
                    heights = [float(h) for h in base_env._column_heights()]
                    frames.append({
                        "type": "frame",
                        "grid": pre_clear.flatten().tolist() + cur_oh + nxt_oh + heights,
                        "step": step,
                        "episode_reward": round(episode_reward, 2),
                        "done": False,
                        "lines_cleared_last": 0,
                        "lines_cleared": base_env._lines_cleared_episode,
                        "highlight_rows": cleared_rows,
                        "q_values": q_vals,
                        "action_probs": action_probs,
                        "entropy": round(entropy, 3) if entropy is not None else None,
                        "selected_action": chosen_name,
                    })
            frames.append({
                "type": "frame",
                "grid": _tetris_viewer_grid(base_env),
                "step": step,
                "episode_reward": round(episode_reward, 2),
                "done": bool(done or truncated),
                "lines_cleared_last": 0,
                "lines_cleared": base_env._lines_cleared_episode,
                "q_values": q_vals,
                "action_probs": action_probs,
                "entropy": round(entropy, 3) if entropy is not None else None,
                "selected_action": chosen_name,
            })
    return frames, round(episode_reward, 2)


def _snake_viewer_grid(base_env) -> list:
    """Build the flat 256-element grid from live SnakeEnv state for the canvas renderer."""
    h, w = base_env.grid_h, base_env.grid_w
    grid = [0.0] * (h * w)
    snake = list(base_env._snake)
    for r, c in snake[:-1]:
        grid[r * w + c] = 0.5   # body
    head_r, head_c = snake[-1]
    grid[head_r * w + head_c] = 1.0  # head
    fr, fc = base_env._food
    grid[fr * w + fc] = -1.0   # food
    return grid


def _run_episode(model, env) -> tuple[list[dict], float]:
    """Run one episode synchronously; return list of frame dicts and total reward."""
    import torch
    import numpy as np
    obs, _ = env.reset()
    frames = []
    episode_reward = 0.0
    step = 0
    done = False
    truncated = False
    base_env = env.unwrapped
    is_tetris = hasattr(base_env, "_lines_cleared_episode")
    is_snake = hasattr(base_env, "_snake")
    is_2048 = hasattr(base_env, "_max_tile")
    is_minatar = hasattr(base_env, "_bricks") or hasattr(base_env, "_ramming") or hasattr(base_env, "_alien_dir")

    # Action names for explainability
    if is_snake:
        action_names = ["UP", "RIGHT", "DOWN", "LEFT"]
    elif is_2048:
        action_names = ["UP", "DOWN", "LEFT", "RIGHT"]
    elif hasattr(base_env, "_bricks"):  # MinAtar Breakout
        action_names = ["NOOP", "LEFT", "RIGHT"]
    elif hasattr(base_env, "_alien_dir"):  # MinAtar Space Invaders
        action_names = ["NOOP", "LEFT", "RIGHT", "FIRE"]
    elif hasattr(base_env, "_ramming"):  # MinAtar Asteroids
        action_names = ["NOOP", "TURN_L", "TURN_R", "THRUST", "FIRE"]
    else:
        action_names = [f"A{i}" for i in range(getattr(env.action_space, "n", 4))]

    while not done and not truncated:
        q_vals = {}
        action_probs = {}
        entropy = None
        try:
            if hasattr(model, "q_net"):
                with torch.no_grad():
                    obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(model.device)
                    q_arr = np.atleast_1d(model.q_net(obs_t).squeeze().cpu().numpy())
                    exp_q = np.exp(q_arr - np.max(q_arr))
                    probs = exp_q / np.sum(exp_q)
                    entropy = float(-np.sum(probs * np.log(np.clip(probs, 1e-8, 1.0))))
                    for idx, val in enumerate(q_arr):
                        name = action_names[idx] if idx < len(action_names) else str(idx)
                        q_vals[name] = round(float(val), 2)
                        action_probs[name] = round(float(probs[idx]), 3)
            elif hasattr(model, "policy"):
                with torch.no_grad():
                    obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(model.device)
                    dist = model.policy.get_distribution(obs_t)
                    if hasattr(dist.distribution, "probs"):
                        probs = np.atleast_1d(dist.distribution.probs.squeeze().cpu().numpy())
                        entropy = float(-np.sum(probs * np.log(np.clip(probs, 1e-8, 1.0))))
                        for idx, val in enumerate(probs):
                            name = action_names[idx] if idx < len(action_names) else str(idx)
                            action_probs[name] = round(float(val), 3)
        except Exception:
            pass

        action, _ = model.predict(obs, deterministic=True)
        act_arr = np.asarray(action)
        act_int = int(act_arr.flat[0]) if act_arr.size > 0 else 0
        chosen_name = action_names[act_int] if act_int < len(action_names) else str(act_int)

        obs, reward, done, truncated, _ = env.step(act_int)
        episode_reward += float(reward)
        step += 1

        if is_tetris:
            grid = _tetris_viewer_grid(base_env)
        elif is_snake:
            grid = _snake_viewer_grid(base_env)
        elif is_2048:
            grid = base_env.get_viewer_grid()
        elif is_minatar:
            grid = base_env.get_viewer_grid()
        else:
            grid = obs.tolist()

        frame: dict = {
            "type": "frame",
            "grid": grid,
            "step": step,
            "episode_reward": round(episode_reward, 2),
            "done": bool(done or truncated),
            "q_values": q_vals,
            "action_probs": action_probs,
            "entropy": round(entropy, 3) if entropy is not None else None,
            "selected_action": chosen_name,
        }
        if is_snake:
            frame["food_eaten"] = base_env._food_eaten
        elif is_tetris:
            frame["lines_cleared"] = base_env._lines_cleared_episode
        elif is_2048:
            frame["score"] = int(base_env._score)
            frame["max_tile"] = int(base_env._max_tile)
        elif is_minatar:
            frame["score"] = round(base_env._score, 2)
            if hasattr(base_env, "_bricks_cleared"):
                frame["bricks_cleared"] = int(base_env._bricks_cleared)
            elif hasattr(base_env, "_aliens_killed"):
                frame["aliens_killed"] = int(base_env._aliens_killed)
            elif hasattr(base_env, "_asteroids_hit"):
                frame["asteroids_hit"] = int(base_env._asteroids_hit)

        frames.append(frame)
    return frames, round(episode_reward, 2)


@router.websocket("/ws/missions/{mission_id}/play")
async def play_ws(
    ws: WebSocket,
    mission_id: str,
    env_id: str = "Snake-v0",
    fps: int = 12,
):
    await ws.accept()

    ckpt_dir = os.path.join(settings.data_path, "missions", mission_id, "checkpoints")
    # Prefer PyTorch .pth for actor_critic trainers; fall back to SB3 .zip
    ckpt_pth = os.path.join(ckpt_dir, "best_model.pth")
    ckpt_zip = os.path.join(ckpt_dir, "best_model.zip")
    ckpt_path = ckpt_pth if os.path.exists(ckpt_pth) else ckpt_zip
    if not os.path.exists(ckpt_path):
        await ws.send_json({"type": "error", "message": "No best_model.pth or best_model.zip found for this mission."})
        await ws.close()
        return

    try:
        loop = asyncio.get_event_loop()

        def _load():
            import sys
            project_root = os.path.abspath(os.path.join(settings.data_path, ".."))
            if project_root not in sys.path:
                sys.path.insert(0, project_root)

            import gymnasium as gym

            cfg = _load_train_config(ckpt_dir)
            env_kwargs = cfg.get("env_kwargs") or {}
            resolved_env_id = cfg.get("env_id") or env_id

            if resolved_env_id == "Snake-v0":
                from envs.snake_env import register as _reg
                _reg()
            elif resolved_env_id == "Tetris-v0":
                from envs.tetris_env import register as _reg
                _reg()
            elif resolved_env_id in ("Game2048-v0", "2048"):
                from envs.game2048_env import register as _reg
                _reg()
            elif resolved_env_id in ("MinAtar-Breakout-v0", "MinAtar-v0", "minatar"):
                from envs.minatar_env import register as _reg
                _reg()
            elif resolved_env_id in ("MinAtar-SpaceInvaders-v0", "MinAtar-Space-Invaders-v0"):
                from envs.minatar_space_invaders_env import register as _reg
                _reg()
            elif resolved_env_id in ("MinAtar-Asteroids-v0",):
                from envs.minatar_asteroids_env import register as _reg
                _reg()

            env = gym.make(resolved_env_id, **env_kwargs)

            # Detect actor_critic / lookahead_dqn / lookahead_ppo / lookahead_a2c
            # PyTorch model — all share the same ActorCriticNet checkpoint shape.
            tt_path = os.path.join(ckpt_dir, "trainer_type.txt")
            is_actor_critic = (
                os.path.exists(tt_path)
                and open(tt_path).read().strip() in (
                    "actor_critic", "lookahead_dqn", "lookahead_ppo", "lookahead_a2c",
                )
            ) or ckpt_path.endswith(".pth")
            if is_actor_critic:
                import sys
                import torch
                from envs.actor_critic_net import ActorCriticNet, Game2048ValueNet
                # Inject into __main__ so torch.load can unpickle models saved
                # from train.py (where the class was defined as __main__.ActorCriticNet)
                sys.modules["__main__"].ActorCriticNet = ActorCriticNet
                sys.modules["__main__"].Game2048ValueNet = Game2048ValueNet
                model = torch.load(ckpt_path, weights_only=False)
                model.eval()
                logger.info("play_ws: loaded ActorCritic PyTorch model for mission=%s env=%s", mission_id, resolved_env_id)
                return model, env, True  # True = is_actor_critic

            # Try the detected algorithm first; if it fails with a policy mismatch,
            # fall back through all known algorithms so stale algo files don't hard-crash.
            algorithm = _checkpoint_algorithm(ckpt_dir, cfg)
            algo_order = [algorithm] + [a for a in _SB3_ALGO_MAP if a != algorithm.upper()]
            last_exc: Optional[Exception] = None
            for algo_name in algo_order:
                try:
                    AlgoClass = _get_algo_class(algo_name)
                    model = AlgoClass.load(ckpt_path, env=env)
                    if algo_name != algorithm:
                        logger.warning(
                            "play_ws: %s.load failed — loaded with %s instead (mission=%s)",
                            algorithm, algo_name, mission_id,
                        )
                    logger.info(
                        "play_ws: loaded %s model for mission=%s env=%s env_kwargs=%s",
                        algo_name, mission_id, resolved_env_id, env_kwargs,
                    )
                    return model, env, False
                except Exception as exc:
                    last_exc = exc
                    continue
            raise RuntimeError(f"Could not load best_model.zip with any known algorithm: {last_exc}")


        model, env, is_ac = await loop.run_in_executor(_EXECUTOR, _load)
        episode_fn = _run_episode_actor_critic if is_ac else _run_episode

        frame_delay = 1.0 / max(1, min(fps, 30))
        episode = 0

        while True:
            episode += 1
            frames, total_reward = await loop.run_in_executor(
                _EXECUTOR, episode_fn, model, env
            )

            for frame in frames:
                frame["episode"] = episode
                try:
                    await ws.send_json(frame)
                except WebSocketDisconnect:
                    return
                await asyncio.sleep(frame_delay)

            try:
                await ws.send_json({
                    "type": "episode_end",
                    "episode": episode,
                    "total_reward": total_reward,
                })
            except WebSocketDisconnect:
                return

            await asyncio.sleep(1.0)

    except WebSocketDisconnect:
        logger.info("play_ws: client disconnected mission=%s", mission_id)
    except Exception as exc:
        logger.exception("play_ws: error mission=%s: %s", mission_id, exc)
        try:
            await ws.send_json({"type": "error", "message": str(exc)})
        except Exception:
            pass
    finally:
        try:
            env.close()
        except Exception:
            pass
