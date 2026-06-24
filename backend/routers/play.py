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
    """Run one episode with a PyTorch Actor-Critic model using get_next_states()."""
    import torch
    obs, _ = env.reset()
    frames = []
    episode_reward = 0.0
    step = 0
    done = False
    truncated = False
    base_env = env.unwrapped
    while not done and not truncated:
        next_states = base_env.get_next_states()
        if next_states:
            with torch.no_grad():
                best_action, best_val = None, float("-inf")
                for act, st in next_states.items():
                    val = model(torch.tensor(st, dtype=torch.float32).unsqueeze(0))
                    if isinstance(val, tuple):
                        val = val[1]  # critic head
                    v = float(val.squeeze())
                    if v > best_val:
                        best_val, best_action = v, act
            action = best_action
        else:
            action = 0
        # Capture the piece being placed so the highlight frame uses the right color
        piece_before_step = base_env._current_piece
        obs, reward, done, truncated, _ = env.step(action)
        episode_reward += float(reward)
        step += 1
        lines_cleared = int(base_env._lines_cleared_last)
        # When lines are cleared, emit a highlight frame (pre-clear board + cleared row indices)
        # so the client can flash exactly those rows before showing the post-clear board.
        if lines_cleared > 0:
            cleared_rows = getattr(base_env, "_last_cleared_rows", [])
            pre_clear = getattr(base_env, "_pre_clear_board", None)
            if cleared_rows and pre_clear is not None:
                cur_oh = [0.0] * 7
                if 0 <= piece_before_step < 7:
                    cur_oh[piece_before_step] = 1.0
                nxt_oh = [0.0] * 7
                nxt = base_env._current_piece  # after step, _current_piece is the next piece
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
                })
        frames.append({
            "type": "frame",
            "grid": _tetris_viewer_grid(base_env),
            "step": step,
            "episode_reward": round(episode_reward, 2),
            "done": bool(done or truncated),
            "lines_cleared_last": 0,
            "lines_cleared": base_env._lines_cleared_episode,
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
    obs, _ = env.reset()
    frames = []
    episode_reward = 0.0
    step = 0
    done = False
    truncated = False
    base_env = env.unwrapped
    is_tetris = hasattr(base_env, '_board')
    is_snake = hasattr(base_env, '_snake')
    while not done and not truncated:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, _ = env.step(action)
        episode_reward += float(reward)
        step += 1
        if is_tetris:
            grid = _tetris_viewer_grid(base_env)
        elif is_snake:
            grid = _snake_viewer_grid(base_env)
        else:
            grid = obs.tolist()
        frame: dict = {
            "type": "frame",
            "grid": grid,
            "step": step,
            "episode_reward": round(episode_reward, 2),
            "done": bool(done or truncated),
        }
        if is_snake:
            frame["food_eaten"] = base_env._food_eaten
        elif is_tetris:
            frame["lines_cleared"] = base_env._lines_cleared_episode
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

            env = gym.make(resolved_env_id, **env_kwargs)

            # Detect actor_critic PyTorch model
            tt_path = os.path.join(ckpt_dir, "trainer_type.txt")
            is_actor_critic = (
                os.path.exists(tt_path) and open(tt_path).read().strip() == "actor_critic"
            ) or ckpt_path.endswith(".pth")
            if is_actor_critic:
                import sys
                import torch
                from envs.actor_critic_net import ActorCriticNet
                # Inject into __main__ so torch.load can unpickle models saved
                # from train.py (where the class was defined as __main__.ActorCriticNet)
                sys.modules["__main__"].ActorCriticNet = ActorCriticNet
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
