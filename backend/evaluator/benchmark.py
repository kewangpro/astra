"""
BenchmarkSuite — Step 3.4 (hardened in Phase 6.3).

Runs the trained model against a fixed "Golden Set" of domain challenges.
The Golden Set is defined per domain (snake, tetris, nlp) and is immutable
across runs to ensure comparable results.
"""
from __future__ import annotations

import gc
import os
import sys
from dataclasses import dataclass, field
from typing import Callable, Optional

from backend.logging_config import get_logger

logger = get_logger(__name__)

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))


def _release_gpu_memory() -> None:
    """Free GPU-backed tensors from a failed/discarded model load. Best-effort —
    torch may not be importable or MPS may not be the active backend."""
    gc.collect()
    try:
        import torch
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
    except Exception:
        pass


def _load_env_kwargs(checkpoint_path: str) -> dict:
    """Read env_kwargs from train_config.json next to the checkpoint.

    Ensures the benchmark env matches the obs_type (and other kwargs) the
    model was actually trained with — e.g. obs_type='features' for Snake.
    """
    import json
    config_path = os.path.join(os.path.dirname(checkpoint_path), "train_config.json")
    if os.path.exists(config_path):
        try:
            with open(config_path) as f:
                return json.load(f).get("env_kwargs") or {}
        except Exception:
            pass
    return {}


_LOOKAHEAD_TRAINER_TYPES = {"actor_critic", "lookahead_dqn", "lookahead_ppo", "lookahead_a2c"}


def _is_actor_critic(checkpoint_path: str) -> bool:
    """Return True if the checkpoint was saved by a custom PyTorch lookahead
    trainer (actor_critic, or one of the lookahead_dqn/ppo/a2c variants — all
    share the same ActorCriticNet checkpoint shape and get_next_states()-based
    rollout procedure, just trained differently)."""
    ckpt_dir = os.path.dirname(checkpoint_path)
    tt_path = os.path.join(ckpt_dir, "trainer_type.txt")
    if os.path.exists(tt_path):
        return open(tt_path).read().strip() in _LOOKAHEAD_TRAINER_TYPES
    # Also detect by file extension — .pth = PyTorch, .zip = SB3
    return checkpoint_path.endswith(".pth")


def _rollout_actor_critic(checkpoint_path: str, env_id: str, n_episodes: int = 10, env_kwargs: Optional[dict] = None) -> tuple[float, dict]:
    """Rollout a custom PyTorch Actor-Critic model using get_next_states()."""
    import numpy as np
    import torch

    if _PROJECT_ROOT not in sys.path:
        sys.path.insert(0, _PROJECT_ROOT)

    try:
        import gymnasium as gym
        if env_id == "Tetris-v0":
            from envs.tetris_env import register as _reg
            _reg()
        elif env_id in ("Game2048-v0", "2048"):
            from envs.game2048_env import register as _reg
            _reg()

        from envs.actor_critic_net import ActorCriticNet, Game2048ValueNet
        sys.modules["__main__"].ActorCriticNet = ActorCriticNet
        sys.modules["__main__"].Game2048ValueNet = Game2048ValueNet
        model = torch.load(checkpoint_path, weights_only=False)
        model.eval()

        env = gym.make(env_id, **(env_kwargs or {}))
        rewards, info_accum = [], {}
        for _ in range(n_episodes):
            obs, _ = env.reset()
            ep_reward, done = 0.0, False
            ep_info = {}
            while not done:
                next_states = env.unwrapped.get_next_states()
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
                obs, r, terminated, truncated, info = env.step(action)
                ep_reward += float(r)
                done = terminated or truncated
                if done:
                    ep_info = info
            rewards.append(ep_reward)
            for k, v in ep_info.items():
                try:
                    info_accum.setdefault(k, []).append(float(v))
                except (TypeError, ValueError):
                    pass
        env.close()
        max_info = {k: float(np.max(vs)) for k, vs in info_accum.items()}
        return float(np.mean(rewards)), max_info
    except Exception as exc:
        logger.warning("BenchmarkSuite actor_critic rollout failed env=%s: %s", env_id, exc)
        return 0.0, {}


def _rollout(checkpoint_path: str, env_id: str, n_episodes: int = 10, env_kwargs: Optional[dict] = None) -> tuple[float, dict]:
    """Load checkpoint, run n_episodes deterministically, return (mean_reward, mean_info).

    Returns info values averaged across episodes. Only collects info at episode end
    (terminated/truncated step). Detects actor_critic PyTorch models automatically.
    """
    if env_kwargs is None:
        env_kwargs = _load_env_kwargs(checkpoint_path)

    if _is_actor_critic(checkpoint_path):
        pth = checkpoint_path.replace(".zip", ".pth")
        if not pth.endswith(".pth"):
            pth = checkpoint_path
        if os.path.exists(pth):
            return _rollout_actor_critic(pth, env_id, n_episodes, env_kwargs=env_kwargs)
        # fall through to SB3 if .pth not found

    import numpy as np

    if _PROJECT_ROOT not in sys.path:
        sys.path.insert(0, _PROJECT_ROOT)

    try:
        from stable_baselines3 import PPO, SAC, A2C, DQN, TD3
        import gymnasium as gym

        # Try loading with each algo until one succeeds. A failed load can
        # still allocate GPU-backed tensors before erroring, and nothing
        # frees them between attempts by default — on a memory-constrained
        # run (e.g. concurrent with an active training subprocess) that can
        # stack up real GPU pressure across this loop's up-to-5 attempts.
        # Release explicitly after each failure rather than only trusting
        # Python's GC to get to it eventually.
        model = None
        for cls in (PPO, SAC, A2C, DQN, TD3):
            try:
                model = cls.load(checkpoint_path)
                break
            except Exception:
                _release_gpu_memory()
                continue
        if model is None:
            return 0.0, {}

        if env_id == "Tetris-v0":
            from envs.tetris_env import register as _reg
            _reg()
        elif env_id == "Snake-v0":
            from envs.snake_env import register as _reg
            _reg()
        elif env_id in ("Game2048-v0", "2048"):
            from envs.game2048_env import register as _reg
            _reg()
        elif env_id in ("MinAtar-Breakout-v0", "MinAtar-v0"):
            from envs.minatar_env import register as _reg
            _reg()
        elif env_id in ("MinAtar-SpaceInvaders-v0", "MinAtar-Space-Invaders-v0"):
            from envs.minatar_space_invaders_env import register as _reg
            _reg()
        elif env_id in ("MinAtar-Asteroids-v0",):
            from envs.minatar_asteroids_env import register as _reg
            _reg()

        env = gym.make(env_id, **(env_kwargs or {}))
        base_env = env.unwrapped
        is_2048 = hasattr(base_env, "_max_tile")
        rewards, info_accum = [], {}
        for _ in range(n_episodes):
            obs, _ = env.reset()
            ep_reward, done = 0.0, False
            ep_info = {}
            while not done:
                if is_2048:
                    valid_actions = list(base_env.get_next_states().keys())
                    if not valid_actions:
                        break
                    try:
                        if hasattr(model, "q_net"):
                            import torch
                            with torch.no_grad():
                                obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(model.device)
                                q_arr = np.atleast_1d(model.q_net(obs_t).squeeze().cpu().numpy())
                            action = int(max(valid_actions, key=lambda a: q_arr[a]))
                        else:
                            pred_action, _ = model.predict(obs, deterministic=True)
                            act_int = int(np.asarray(pred_action).flat[0])
                            action = act_int if act_int in valid_actions else valid_actions[0]
                    except Exception:
                        action, _ = model.predict(obs, deterministic=True)
                else:
                    action, _ = model.predict(obs, deterministic=True)

                obs, r, terminated, truncated, info = env.step(action)
                ep_reward += float(r)
                done = terminated or truncated
                if done:
                    ep_info = info
            rewards.append(ep_reward)
            for k, v in ep_info.items():
                try:
                    info_accum.setdefault(k, []).append(float(v))
                except (TypeError, ValueError):
                    pass
        env.close()
        # Use max for goal metrics (e.g. food_eaten, lines_cleared) — reflects peak
        # capability rather than average, consistent with "achieve X" goal semantics.
        max_info = {k: float(np.max(vs)) for k, vs in info_accum.items()}
        return float(np.mean(rewards)), max_info
    except Exception as exc:
        logger.warning("BenchmarkSuite rollout failed env=%s: %s", env_id, exc)
        return 0.0, {}


@dataclass
class GoldenChallenge:
    name: str
    domain: str
    description: str
    evaluate_fn: Callable[[str], dict]  # takes checkpoint_path, returns {metric: value}
    pass_threshold: dict = field(default_factory=dict)


# ── Built-in Golden Sets ───────────────────────────────────────────────────────

def _snake_eval(checkpoint_path: str) -> dict:
    if not os.path.exists(checkpoint_path):
        logger.warning("BenchmarkSuite: checkpoint not found: %s", checkpoint_path)
        return {"mean_reward": 0.0, "max_length": 0}
    try:
        if _PROJECT_ROOT not in sys.path:
            sys.path.insert(0, _PROJECT_ROOT)
        from envs.snake_env import register
        register()
    except Exception:
        pass
    logger.info("BenchmarkSuite: running Snake baseline on %s", checkpoint_path)
    mean_reward, info = _rollout(checkpoint_path, "Snake-v0", n_episodes=10)
    return {"mean_reward": mean_reward, "max_length": info.get("max_length", 0.0)}


def _snake_hard_eval(checkpoint_path: str) -> dict:
    if not os.path.exists(checkpoint_path):
        return {"mean_reward": 0.0, "max_length": 0}
    try:
        if _PROJECT_ROOT not in sys.path:
            sys.path.insert(0, _PROJECT_ROOT)
        from envs.snake_env import register
        register()
    except Exception:
        pass
    logger.info("BenchmarkSuite: running Snake hard on %s", checkpoint_path)
    mean_reward, info = _rollout(checkpoint_path, "Snake-v0", n_episodes=10)
    return {"mean_reward": mean_reward, "max_length": info.get("max_length", 0.0)}


def _tetris_eval(checkpoint_path: str) -> dict:
    if not os.path.exists(checkpoint_path):
        return {"mean_reward": 0.0, "lines_cleared": 0}
    try:
        if _PROJECT_ROOT not in sys.path:
            sys.path.insert(0, _PROJECT_ROOT)
        from envs.tetris_env import register
        register()
    except Exception:
        pass
    logger.info("BenchmarkSuite: running Tetris baseline on %s", checkpoint_path)
    mean_reward, info = _rollout(checkpoint_path, "Tetris-v0", n_episodes=10)
    return {"mean_reward": mean_reward, "lines_cleared": info.get("lines_cleared", 0.0)}


def _tetris_hard_eval(checkpoint_path: str) -> dict:
    if not os.path.exists(checkpoint_path):
        return {"mean_reward": 0.0, "lines_cleared": 0}
    try:
        if _PROJECT_ROOT not in sys.path:
            sys.path.insert(0, _PROJECT_ROOT)
        from envs.tetris_env import register
        register()
    except Exception:
        pass
    logger.info("BenchmarkSuite: running Tetris hard on %s", checkpoint_path)
    mean_reward, info = _rollout(checkpoint_path, "Tetris-v0", n_episodes=10)
    return {"mean_reward": mean_reward, "lines_cleared": info.get("lines_cleared", 0.0)}


def _nlp_loss_eval(checkpoint_path: str) -> dict:
    if not os.path.exists(checkpoint_path):
        return {"eval_loss": 999.0, "perplexity": 999.0}
    logger.info("BenchmarkSuite: running NLP loss eval on %s", checkpoint_path)
    meta_candidates = []
    if os.path.isdir(checkpoint_path):
        meta_candidates.append(os.path.join(checkpoint_path, "checkpoint_metadata.json"))
        meta_candidates.append(os.path.join(checkpoint_path, "best", "checkpoint_metadata.json"))
    parent_dir = os.path.dirname(checkpoint_path)
    meta_candidates.extend([
        os.path.join(parent_dir, "checkpoint_metadata.json"),
        os.path.join(parent_dir, "best", "checkpoint_metadata.json"),
        os.path.join(parent_dir, "..", "checkpoints", "best", "checkpoint_metadata.json"),
    ])

    for meta_path in meta_candidates:
        if os.path.isfile(meta_path):
            try:
                import json as _json, math as _math
                with open(meta_path, "r") as f:
                    meta = _json.load(f)
                el = float(meta.get("eval_loss", 999.0))
                perp = float(meta.get("perplexity", _math.exp(el) if el < 100 else 999.0))
                return {"eval_loss": el, "perplexity": perp}
            except Exception:
                pass

    # Fallback to reading eval_loss from telemetry.jsonl in mission dir
    tel_candidates = [
        os.path.join(parent_dir, "telemetry.jsonl"),
        os.path.join(parent_dir, "..", "telemetry.jsonl"),
        os.path.join(os.path.dirname(os.path.abspath(checkpoint_path)), "..", "telemetry.jsonl"),
    ]
    for tel_path in tel_candidates:
        if os.path.isfile(tel_path):
            try:
                import json as _json, math as _math
                best_el = None
                with open(tel_path, "r") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        evt = _json.loads(line)
                        if (evt.get("type") == "metric" or "name" in evt) and evt.get("name") == "eval_loss":
                            val = float(evt.get("value", 999.0))
                            if val < 900.0:
                                if best_el is None or val < best_el:
                                    best_el = val
                if best_el is not None:
                    perp = float(_math.exp(best_el) if best_el < 100 else 999.0)
                    return {"eval_loss": best_el, "perplexity": perp}
            except Exception:
                pass

    return {"eval_loss": 999.0, "perplexity": 999.0}


def _nlp_perplexity_eval(checkpoint_path: str) -> dict:
    """Stricter NLP scenario: perplexity on an out-of-domain validation set."""
    if not os.path.exists(checkpoint_path):
        return {"eval_loss": 999.0, "perplexity": 999.0}
    logger.info("BenchmarkSuite: running NLP perplexity eval on %s", checkpoint_path)
    return _nlp_loss_eval(checkpoint_path)


GOLDEN_SETS: dict[str, list[GoldenChallenge]] = {
    "snake": [
        GoldenChallenge(
            name="snake_baseline",
            domain="snake",
            description="Achieve mean_reward ≥ 20 on 16×12 grid",
            evaluate_fn=_snake_eval,
            pass_threshold={"mean_reward": 20},
        ),
        GoldenChallenge(
            name="snake_hard",
            domain="snake",
            description="Achieve mean_reward ≥ 80 on 32×24 grid",
            evaluate_fn=_snake_hard_eval,
            pass_threshold={"mean_reward": 80},
        ),
    ],
    "tetris": [
        GoldenChallenge(
            name="tetris_baseline",
            domain="tetris",
            description="Clear ≥ 10 lines on standard board",
            evaluate_fn=_tetris_eval,
            pass_threshold={"lines_cleared": 10},
        ),
        GoldenChallenge(
            name="tetris_hard",
            domain="tetris",
            description="Clear ≥ 40 lines in speed mode",
            evaluate_fn=_tetris_hard_eval,
            pass_threshold={"lines_cleared": 40},
        ),
    ],
    "nlp": [
        GoldenChallenge(
            name="nlp_loss",
            domain="nlp",
            description="Achieve eval_loss ≤ 1.5 on validation set",
            evaluate_fn=_nlp_loss_eval,
            pass_threshold={"eval_loss": 1.5},
        ),
        GoldenChallenge(
            name="nlp_perplexity",
            domain="nlp",
            description="Achieve perplexity ≤ 20.0 on OOD validation set",
            evaluate_fn=_nlp_perplexity_eval,
            pass_threshold={"perplexity": 20.0},
        ),
    ],
}


class BenchmarkSuite:
    def __init__(self, domain: str) -> None:
        self.domain = domain
        self.challenges = GOLDEN_SETS.get(domain, [])
        if not self.challenges:
            logger.warning("BenchmarkSuite: no Golden Set defined for domain '%s'", domain)

    def run(self, checkpoint_path: str) -> dict:
        """
        Run all Golden Challenges for this domain.
        Returns {"passed": int, "failed": int, "results": [...]}.
        """
        passed = 0
        failed = 0
        results = []

        for challenge in self.challenges:
            metrics = challenge.evaluate_fn(checkpoint_path)

            # NLP metrics use lower-is-better semantics for loss/perplexity
            def _passes(k: str, threshold: float, value: float) -> bool:
                if k in ("eval_loss", "perplexity", "adversarial_loss", "ood_perplexity"):
                    return value <= threshold
                return value >= threshold

            challenge_passed = all(
                _passes(k, v, metrics.get(k, 0))
                for k, v in challenge.pass_threshold.items()
            )
            status = "passed" if challenge_passed else "failed"
            if challenge_passed:
                passed += 1
            else:
                failed += 1
            results.append({
                "name": challenge.name,
                "status": status,
                "metrics": metrics,
                "threshold": challenge.pass_threshold,
            })
            logger.info("BenchmarkSuite: %s → %s (%s)", challenge.name, status, metrics)

        return {"passed": passed, "failed": failed, "results": results}


def run_tournament_match(
    checkpoint_entries: List[dict],
    env_id: str = "Snake-v0",
    n_episodes: int = 10,
    env_kwargs: Optional[dict] = None,
) -> dict:
    """
    Run side-by-side tournament across multiple model checkpoints on fixed seeds.
    checkpoint_entries: list of {"id": str, "name": str, "path": str}
    """
    import numpy as np
    import gymnasium as gym

    # Register custom environments
    if env_id == "Tetris-v0":
        from envs.tetris_env import register as _reg; _reg()
    elif env_id == "Snake-v0":
        from envs.snake_env import register as _reg; _reg()
    elif env_id in ("Game2048-v0", "2048"):
        from envs.game2048_env import register as _reg; _reg()
    elif env_id in ("MinAtar-Breakout-v0", "MinAtar-v0"):
        from envs.minatar_env import register as _reg; _reg()
    elif env_id in ("MinAtar-SpaceInvaders-v0", "MinAtar-Space-Invaders-v0"):
        from envs.minatar_space_invaders_env import register as _reg; _reg()
    elif env_id in ("MinAtar-Asteroids-v0",):
        from envs.minatar_asteroids_env import register as _reg; _reg()

    loaded_models = []
    for entry in checkpoint_entries:
        path = entry["path"]
        is_ac = _is_actor_critic(path)
        if is_ac and not path.endswith(".pth"):
            pth_cand = path.replace(".zip", ".pth")
            if os.path.exists(pth_cand):
                path = pth_cand

        m_obj = None
        if path.endswith(".pth"):
            import torch
            from envs.actor_critic_net import ActorCriticNet, Game2048ValueNet
            sys.modules["__main__"].ActorCriticNet = ActorCriticNet
            sys.modules["__main__"].Game2048ValueNet = Game2048ValueNet
            try:
                m_obj = torch.load(path, weights_only=False)
                m_obj.eval()
            except Exception as e:
                logger.warning("Tournament failed to load pth %s: %s", path, e)
        else:
            from stable_baselines3 import PPO, SAC, A2C, DQN, TD3
            for cls in (PPO, DQN, SAC, A2C, TD3):
                try:
                    m_obj = cls.load(path)
                    break
                except Exception:
                    continue

        loaded_models.append({
            "id": entry["id"],
            "name": entry.get("name", entry["id"]),
            "path": path,
            "is_ac": path.endswith(".pth"),
            "model": m_obj,
            "scores": [],
        })

    env = gym.make(env_id, **(env_kwargs or {}))
    base_env = env.unwrapped
    for ep in range(n_episodes):
        seed = 2000 + ep
        for m in loaded_models:
            if m["model"] is None:
                m["scores"].append(0.0)
                continue
            obs, _ = env.reset(seed=seed)
            done, truncated = False, False
            ep_score = 0.0
            ep_reward = 0.0
            while not done and not truncated:
                if m["is_ac"]:
                    import torch
                    next_states = base_env.get_next_states()
                    if next_states:
                        with torch.no_grad():
                            best_act, best_v = None, float("-inf")
                            for act, st in next_states.items():
                                val = m["model"](torch.tensor(st, dtype=torch.float32).unsqueeze(0))
                                if isinstance(val, tuple):
                                    val = val[1]
                                v = float(val.squeeze())
                                if v > best_v:
                                    best_v, best_act = v, act
                        action = best_act
                    else:
                        action = 0
                else:
                    action, _ = m["model"].predict(obs, deterministic=True)

                obs, r, done, truncated, info = env.step(action)
                ep_reward += float(r)
                if done or truncated:
                    if hasattr(base_env, "_lines_cleared_episode"):
                        ep_score = float(base_env._lines_cleared_episode)
                    elif hasattr(base_env, "_food_eaten"):
                        ep_score = float(base_env._food_eaten)
                    elif hasattr(base_env, "_score"):
                        ep_score = float(base_env._score)
                    elif "score" in info:
                        ep_score = float(info["score"])
                    else:
                        ep_score = float(ep_reward)

            m["scores"].append(ep_score)
    env.close()

    # Calculate win rates & rankings
    wins = {m["id"]: 0.0 for m in loaded_models}
    for ep in range(n_episodes):
        ep_scores = [m["scores"][ep] for m in loaded_models]
        max_s = max(ep_scores) if ep_scores else 0.0
        winners = [m["id"] for m in loaded_models if m["scores"][ep] == max_s]
        if winners:
            share = 1.0 / len(winners)
            for w in winners:
                wins[w] += share

    leaderboard = []
    for m in loaded_models:
        scores = m["scores"]
        mean_s = float(np.mean(scores)) if scores else 0.0
        std_s = float(np.std(scores)) if scores else 0.0
        min_s = float(np.min(scores)) if scores else 0.0
        max_s = float(np.max(scores)) if scores else 0.0
        win_rate = round(float(wins[m["id"]] / max(1, n_episodes)), 3)
        leaderboard.append({
            "model_id": m["id"],
            "name": m["name"],
            "checkpoint_path": m["path"],
            "mean_score": round(mean_s, 2),
            "std_score": round(std_s, 2),
            "min_score": round(min_s, 2),
            "max_score": round(max_s, 2),
            "win_rate": win_rate,
            "scores": scores,
        })

    leaderboard.sort(key=lambda x: (x["mean_score"], x["win_rate"]), reverse=True)
    for idx, entry in enumerate(leaderboard):
        entry["rank"] = idx + 1

    return {
        "env_id": env_id,
        "episodes": n_episodes,
        "leaderboard": leaderboard,
        "champion_id": leaderboard[0]["model_id"] if leaderboard else None,
    }

