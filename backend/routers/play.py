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


def _run_episode(model, env) -> tuple[list[dict], float]:
    """Run one episode synchronously; return list of frame dicts and total reward."""
    obs, _ = env.reset()
    frames = []
    episode_reward = 0.0
    step = 0
    done = False
    truncated = False
    while not done and not truncated:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, _ = env.step(action)
        episode_reward += float(reward)
        step += 1
        frames.append({
            "type": "frame",
            "grid": obs.tolist(),
            "step": step,
            "episode_reward": round(episode_reward, 2),
            "done": bool(done or truncated),
        })
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
    ckpt_path = os.path.join(ckpt_dir, "best_model.zip")
    if not os.path.exists(ckpt_path):
        await ws.send_json({"type": "error", "message": "No best_model.zip found for this mission."})
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
            # Use best_model_algo.txt when available — it records which algorithm
            # actually saved best_model.zip, which may differ from train_config.json
            # if a pivot switched algorithms but the previous algo still holds the best score.
            algorithm = _checkpoint_algorithm(ckpt_dir, cfg)
            env_kwargs = cfg.get("env_kwargs") or {}
            resolved_env_id = cfg.get("env_id") or env_id

            if resolved_env_id == "Snake-v0":
                from envs.snake_env import register as _reg
                _reg()

            env = gym.make(resolved_env_id, **env_kwargs)

            # Try the detected algorithm first; if it fails with a policy mismatch,
            # fall back through all known algorithms so stale algo files don't hard-crash.
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
                    return model, env
                except Exception as exc:
                    last_exc = exc
                    continue
            raise RuntimeError(f"Could not load best_model.zip with any known algorithm: {last_exc}")


        model, env = await loop.run_in_executor(_EXECUTOR, _load)

        frame_delay = 1.0 / max(1, min(fps, 30))
        episode = 0

        while True:
            episode += 1
            frames, total_reward = await loop.run_in_executor(
                _EXECUTOR, _run_episode, model, env
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
