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
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from backend.config import settings
from backend.logging_config import get_logger

logger = get_logger(__name__)

router = APIRouter()

_EXECUTOR = ThreadPoolExecutor(max_workers=2, thread_name_prefix="snake-play")


def _run_episode(model, env) -> list[dict]:
    """Run one episode synchronously; return list of frame dicts."""
    import numpy as np
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

    ckpt_path = os.path.join(
        settings.data_path, "missions", mission_id, "checkpoints", "best_model.zip"
    )
    if not os.path.exists(ckpt_path):
        await ws.send_json({"type": "error", "message": "No best_model.zip found for this mission."})
        await ws.close()
        return

    try:
        # Load model + env in thread pool (SB3/numpy are not async-native)
        loop = asyncio.get_event_loop()

        def _load():
            import sys
            project_root = os.path.abspath(os.path.join(settings.data_path, ".."))
            if project_root not in sys.path:
                sys.path.insert(0, project_root)

            import gymnasium as gym
            from stable_baselines3 import PPO

            if env_id == "Snake-v0":
                from envs.snake_env import register as _reg
                _reg()

            env = gym.make(env_id)
            model = PPO.load(ckpt_path, env=env)
            return model, env

        model, env = await loop.run_in_executor(_EXECUTOR, _load)
        logger.info("play_ws: loaded model for mission=%s env=%s", mission_id, env_id)

        frame_delay = 1.0 / max(1, min(fps, 30))
        episode = 0

        while True:
            episode += 1
            frames, total_reward = await loop.run_in_executor(
                _EXECUTOR, _run_episode, model, env
            )

            for i, frame in enumerate(frames):
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

            # Brief pause between episodes
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
