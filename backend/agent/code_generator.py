"""
CodeGenerator — Step 3.2.

Produces runnable Python training scripts from a plan dict.
The script is written to data/missions/{id}/train.py and executed
by the SandboxManager.
"""
from __future__ import annotations

import json
import os
from typing import Optional

from backend.agent.inference.base import InferenceProvider, Message, GenerationConfig
from backend.config import settings
from backend.logging_config import get_logger
from backend.sandbox.manager import _FINETUNE_REMOTE_TASK_TYPES

logger = get_logger(__name__)

# ── System prompts ─────────────────────────────────────────────────────────────

_BASE_SYSTEM = """\
You are ASTRA's Code Generator — an expert ML engineer.
Generate complete, runnable Python training scripts.
Requirements:
- The script must run standalone inside a sandboxed Python environment.
- Import all dependencies at the top.
- For RL scripts: ALWAYS use `import gymnasium as gym` — NOT `import gym`.
  The `gym` package is NOT installed; only `gymnasium` is available.
- NEVER import or call `stable_baselines3.common.logger.configure()` — it is
  not needed. Use Python's standard `logging` module for any logging.
- Log metrics by POSTing to the ASTRA telemetry endpoint:
    POST {api_url}/telemetry/missions/{mission_id}/metrics
    Body: {{"mission_id": "...", "name": "...", "value": 0.0, "step": 0}}
- Save checkpoints to: {checkpoint_dir}
- On error, print the full traceback and exit with code 1.
- DO NOT use markdown code blocks (```python ... ```).
- Return ONLY the raw Python script, no explanation, no preamble, no stop tokens."""

# Injected verbatim at the top of custom-env training scripts
_SNAKE_SETUP = """\
import sys as _sys
_sys.path.insert(0, "{project_root}")
from envs.snake_env import register as _register_snake
_register_snake()
"""

_TETRIS_SETUP = """\
import sys as _sys
_sys.path.insert(0, "{project_root}")
import gymnasium as gym
from envs.tetris_env import register as _register_tetris
_register_tetris()
"""

_RL_TEMPLATE = """\
Generate a complete RL training script using Stable-Baselines3.
{env_setup}
Mission ID: {mission_id}
Algorithm: {algorithm}
Environment: {env_id}
Hyperparameters: {hyperparameters}
Policy kwargs (network architecture): {policy_kwargs}
Target metric: {target_metric}
Checkpoint directory: {checkpoint_dir}
Telemetry URL: {api_url}/telemetry/missions/{mission_id}/metrics

The script MUST start with these exact imports (do not omit any):
    import os
    import gymnasium as gym
    import numpy as np
    import requests
    import logging
    from stable_baselines3 import {algorithm}
    from stable_baselines3.common.callbacks import BaseCallback

The script must:
1. Create the environment with EXACTLY these lines — copy verbatim:
       import gymnasium as gym
       env = gym.make("{env_id}"{env_kwargs_str})
   The env_id is "{env_id}". Do NOT use "CartPole-v1" or any other env name.
   Do NOT read it from hyperparameters. Hard-code "{env_id}" in gym.make().
2. The script MUST use these EXACT lines to construct the model — copy verbatim, do NOT change any values:

       def _linear_schedule(initial_value):
           def _schedule(progress_remaining):
               return progress_remaining * initial_value
           return _schedule

       {valid_keys_var} = {valid_keys_set}
       _hp = {hyperparameters}
       _filtered = {{k: v for k, v in _hp.items() if k in {valid_keys_var}}}
       if _hp.get("lr_schedule") == "linear" and "learning_rate" in _filtered:
           _filtered["learning_rate"] = _linear_schedule(_filtered["learning_rate"])
       _policy_kwargs = {policy_kwargs}
       model = {algorithm}("MlpPolicy", env, **_filtered,
                   **(dict(policy_kwargs=_policy_kwargs) if _policy_kwargs else {{}}))

   These values are set by the optimizer — do NOT substitute your own hyperparameter values.
   `lr_schedule` is an opt-in recipe setting (not an SB3 constructor kwarg, so it is
   automatically excluded from `_filtered` by the {valid_keys_var} check above). When set to
   "linear", `_linear_schedule` decays the learning rate linearly from its initial value to 0
   over the course of THIS run's `total_timesteps` (SB3 calls it with `progress_remaining`
   going 1 → 0). When absent, learning_rate stays a constant scalar as before. This is in
   addition to, not a replacement for, the optimizer's across-iteration pivot adjustments to
   the starting learning rate.
3. Immediately after constructing the model, copy this warm-start block EXACTLY — do not modify:

       _best_ckpt = "{checkpoint_dir}/best_model.zip"
       if os.path.exists(_best_ckpt):
           try:
               _warm = {algorithm}.load(_best_ckpt, env=env)
               model.policy.load_state_dict(_warm.policy.state_dict())
               del _warm
           except Exception as _e:
               logging.warning("Warm-start skipped (architecture mismatch or load error): %s", _e)

   This resumes training from the best previously saved weights while keeping the new hyperparameters.
   If the checkpoint architecture differs (e.g. after a net_arch pivot), the except branch silently
   falls back to random weights. `os` and `logging` are already imported.
   The block is MANDATORY — do not remove or skip it.
4. Implement a custom BaseCallback. Copy this ENTIRE class EXACTLY — do not add, remove, or
   modify any line:

       class CustomCallback(BaseCallback):
           def __init__(self, verbose=0):
               super().__init__(verbose=verbose)
               try:
                   self._best_reward = float(open("{checkpoint_dir}/best_score.txt").read().strip())
               except Exception:
                   self._best_reward = float("-inf")

           def _on_step(self) -> bool:
               if self.n_calls % {telemetry_interval} == 0 and len(self.model.ep_info_buffer) > 0:
                   mean_reward = float(np.mean([ep["r"] for ep in self.model.ep_info_buffer]))
                   try:
                       response = requests.post(
                           "{api_url}/telemetry/missions/{mission_id}/metrics",
                           json={{"mission_id": "{mission_id}", "name": "mean_reward",
                                  "value": mean_reward, "step": self.n_calls,
                                  "iteration": {current_iteration}}},
                           timeout=2,
                       )
                       if not response.ok:
                           logging.warning("Telemetry failed: %s", response.status_code)
                   except Exception as exc:
                       logging.warning("Telemetry error: %s", exc)
                   if mean_reward > self._best_reward:
                       self._best_reward = mean_reward
                       self.model.save("{checkpoint_dir}/best_model")
                       with open("{checkpoint_dir}/best_score.txt", "w") as _f:
                           _f.write(str(mean_reward))
                       with open("{checkpoint_dir}/best_model_algo.txt", "w") as _f:
                           _f.write(self.model.__class__.__name__)
                   if mean_reward >= {target_reward}:
                       return False  # stop training — target reached
               return True

   The `self.n_calls % {telemetry_interval} == 0` guard is MANDATORY. Never remove it.
   The best_model save block is MANDATORY — it ensures the peak model is preserved.
   The __init__ loading from best_score.txt is MANDATORY — it preserves peak weights across restarts.
5. Call model.learn(total_timesteps={total_timesteps}, callback=callback).
6. After training, save the final model: model.save("{checkpoint_dir}/last_model")
7. Exit cleanly when target mean_reward is reached."""

_ACTOR_CRITIC_CONTRACT = """\
Generate a complete custom Actor-Critic training script for Tetris using PyTorch.
Do NOT use Stable-Baselines3. Implement everything with PyTorch and plain Python.

{env_setup}
Mission ID: {mission_id}
Environment: {env_id}
Checkpoint directory: {checkpoint_dir}
Telemetry URL: {api_url}/telemetry/missions/{mission_id}/metrics
Target: lines_cleared >= {target_lines}
Hyperparameters: {hyperparameters}

== Mandatory imports and setup (copy exactly) ==
from envs.actor_critic_net import ActorCriticNet   # MANDATORY — do NOT redefine inline
model = ActorCriticNet()                            # shared MLP [4→64→64] + critic head Linear(64,1)
optimizer = torch.optim.Adam(model.parameters(), lr={lr})

== Training skeleton (follow exactly — do NOT deviate from these API calls) ==
env = gym.make("{env_id}")          # MANDATORY — must appear before the training loop
BUFFER = collections.deque(maxlen={replay_buffer_size})
ep_rewards, ep_lines = [], []
_eps_path = "{checkpoint_dir}/epsilon.txt"  # persist epsilon across iterations — otherwise every fresh script resets to 1.0 (near-total random exploration) even though the model itself is warm-started, making training-time mean_reward meaningless relative to the model's real learned quality
epsilon = float(open(_eps_path).read().strip()) if os.path.exists(_eps_path) else 1.0
best_reward = float("-inf")
total_steps = 0
episode = 0

while total_steps < {total_timesteps}:
    obs, _ = env.reset()          # gymnasium reset → (obs, info); unpack both
    ep_reward = 0.0
    ep_lines_cleared = 0
    done = False

    while not done:
        next_states = env.unwrapped.get_next_states()  # dict{{action: np.array(4,)}}
        if not next_states:
            action = 0
        elif random.random() < epsilon:
            action = random.choice(list(next_states.keys()))
        else:
            with torch.no_grad():
                action = max(
                    next_states,
                    key=lambda a: model(
                        torch.tensor(next_states[a], dtype=torch.float32).unsqueeze(0)
                    ).item()
                )

        next_obs, reward, terminated, truncated, info = env.step(action)  # 5-tuple
        done = terminated or truncated
        BUFFER.append((obs[:4], action, reward, next_obs[:4], float(done)))  # ActorCriticNet takes the 4 board-quality features only — obs/next_obs may carry extra piece-identity dims for other trainers
        obs = next_obs
        ep_reward += reward
        total_steps += 1
        if done:
            ep_lines_cleared = info.get("lines_cleared", 0)

        if len(BUFFER) >= {batch_size}:
            batch = random.sample(BUFFER, {batch_size})
            s, _, r, ns, d = zip(*batch)
            s  = torch.tensor(s,  dtype=torch.float32)
            ns = torch.tensor(ns, dtype=torch.float32)
            r  = torch.tensor(r,  dtype=torch.float32).unsqueeze(1)
            d  = torch.tensor(d,  dtype=torch.float32).unsqueeze(1)
            with torch.no_grad():
                td_target = r + {gamma} * model(ns) * (1 - d)
            loss = nn.MSELoss()(model(s), td_target)
            optimizer.zero_grad(); loss.backward(); optimizer.step()

    epsilon = max({epsilon_min}, epsilon * {epsilon_decay})
    open(_eps_path, "w").write(str(epsilon))  # persist so the NEXT iteration's fresh script resumes decay instead of resetting to 1.0
    ep_rewards.append(ep_reward)
    ep_lines.append(ep_lines_cleared)
    episode += 1

    if len(ep_rewards) >= {ac_telemetry_interval} and episode % {ac_telemetry_interval} == 0:
        mean_reward_50 = float(np.mean(ep_rewards[-{ac_telemetry_interval}:]))
        mean_lines_50  = float(np.mean(ep_lines[-{ac_telemetry_interval}:]))
        # telemetry POSTs here (catch all exceptions)
        if mean_reward_50 > best_reward:
            best_reward = mean_reward_50
            # save best model here

== ASTRA integration contract (ALL items MANDATORY) ==
1. Write this file once at startup (so play/eval endpoints detect the trainer):
     open("{checkpoint_dir}/trainer_type.txt", "w").write("actor_critic")
2. Warm-start: if "{checkpoint_dir}/best_model.pth" exists, load with weights_only=False
   (MANDATORY — ActorCriticNet is a custom class and torch 2.6 requires this flag):
     if os.path.exists("{checkpoint_dir}/best_model.pth"):
         checkpoint = torch.load("{checkpoint_dir}/best_model.pth", weights_only=False)
         if isinstance(checkpoint, dict):
             model.load_state_dict(checkpoint)
         else:
             model.load_state_dict(checkpoint.state_dict())
3. Track rolling mean_reward over last {ac_telemetry_interval} episodes.
   Every {ac_telemetry_interval} episodes POST only mean_reward to telemetry:
     POST {api_url}/telemetry/missions/{mission_id}/metrics
     json={{"mission_id": "{mission_id}", "name": "mean_reward",
            "value": mean_reward_50, "step": episode, "iteration": {current_iteration}}}
   DO NOT post goal-metric names (lines_cleared, food_eaten, etc.) during training —
   those are reserved for end-of-iteration eval rollouts by the orchestrator.
   Use timeout=2, catch all exceptions (telemetry is non-critical).
4. When rolling mean_reward improves over best seen so far:
     torch.save(model, "{checkpoint_dir}/best_model.pth")
     open("{checkpoint_dir}/best_score.txt", "w").write(str(best_reward))
     open("{checkpoint_dir}/best_model_algo.txt", "w").write("ActorCritic")
5. After the timestep budget ({total_timesteps} steps): torch.save(model, "{checkpoint_dir}/last_model.pth")

Return ONLY the raw Python script. No markdown fences, no explanation."""

_LOOKAHEAD_DQN_CONTRACT = """\
Generate a complete custom lookahead-DQN training script for Tetris using PyTorch.
Do NOT use Stable-Baselines3. Implement everything with PyTorch and plain Python.

Real incident this trainer exists to fix: vanilla SB3 DQN structurally cannot
compete on Tetris-v0 — it picks blind among 40 raw action indices with no
knowledge of what each one produces, confirmed live across 130+ pivots all
plateaued at lines_cleared≈0-1. This trainer uses the SAME get_next_states()
lookahead action-selection spine as the proven Actor-Critic trainer (which hit
394 lines in 3 iterations): evaluate every legal placement's actual resulting
board through the value network, act greedily on it. What makes this
recognizably DQN (not just a copy of Actor-Critic) is retaining DQN's core
stabilization mechanism: an off-policy replay buffer PLUS a target network for
stable TD targets — the existing Actor-Critic trainer has no target network at
all, so this is a genuine upgrade, not a rename.

{env_setup}
Mission ID: {mission_id}
Environment: {env_id}
Checkpoint directory: {checkpoint_dir}
Telemetry URL: {api_url}/telemetry/missions/{mission_id}/metrics
Target: lines_cleared >= {target_lines}
Hyperparameters: {hyperparameters}

== Mandatory imports and setup (copy exactly) ==
from envs.actor_critic_net import ActorCriticNet   # MANDATORY — do NOT redefine inline
model = ActorCriticNet()                            # value network: shared MLP [4→64→64] + critic head Linear(64,1)
target_model = ActorCriticNet()
target_model.load_state_dict(model.state_dict())
target_model.eval()
optimizer = torch.optim.Adam(model.parameters(), lr={lr})

== Training skeleton (follow exactly — do NOT deviate from these API calls) ==
env = gym.make("{env_id}")          # MANDATORY — must appear before the training loop
BUFFER = collections.deque(maxlen={replay_buffer_size})
ep_rewards, ep_lines = [], []
_eps_path = "{checkpoint_dir}/epsilon.txt"  # persist epsilon across iterations — otherwise every fresh script resets to 1.0 (near-total random exploration) even though the model itself is warm-started, making training-time mean_reward meaningless relative to the model's real learned quality
epsilon = float(open(_eps_path).read().strip()) if os.path.exists(_eps_path) else 1.0
best_reward = float("-inf")
total_steps = 0
episode = 0

while total_steps < {total_timesteps}:
    obs, _ = env.reset()
    ep_reward = 0.0
    ep_lines_cleared = 0
    done = False

    while not done:
        next_states = env.unwrapped.get_next_states()  # dict{{action: np.array(4,)}}
        if not next_states:
            action = 0
        elif random.random() < epsilon:
            action = random.choice(list(next_states.keys()))
        else:
            with torch.no_grad():
                action = max(
                    next_states,
                    key=lambda a: model(
                        torch.tensor(next_states[a], dtype=torch.float32).unsqueeze(0)
                    ).item()
                )

        next_obs, reward, terminated, truncated, info = env.step(action)  # 5-tuple
        done = terminated or truncated
        BUFFER.append((obs[:4], action, reward, next_obs[:4], float(done)))  # ActorCriticNet takes the 4 board-quality features only — obs/next_obs may carry extra piece-identity dims for other trainers
        obs = next_obs
        ep_reward += reward
        total_steps += 1
        if done:
            ep_lines_cleared = info.get("lines_cleared", 0)

        if len(BUFFER) >= {batch_size}:
            batch = random.sample(BUFFER, {batch_size})
            s, _, r, ns, d = zip(*batch)
            s  = torch.tensor(s,  dtype=torch.float32)
            ns = torch.tensor(ns, dtype=torch.float32)
            r  = torch.tensor(r,  dtype=torch.float32).unsqueeze(1)
            d  = torch.tensor(d,  dtype=torch.float32).unsqueeze(1)
            with torch.no_grad():
                td_target = r + {gamma} * target_model(ns) * (1 - d)   # target network, NOT live model
            loss = nn.MSELoss()(model(s), td_target)
            optimizer.zero_grad(); loss.backward(); optimizer.step()

        if total_steps % {target_update_interval} == 0:
            target_model.load_state_dict(model.state_dict())          # periodic target sync — the DQN-defining step

    epsilon = max({epsilon_min}, epsilon * {epsilon_decay})
    open(_eps_path, "w").write(str(epsilon))  # persist so the NEXT iteration's fresh script resumes decay instead of resetting to 1.0
    ep_rewards.append(ep_reward)
    ep_lines.append(ep_lines_cleared)
    episode += 1

    if len(ep_rewards) >= {ac_telemetry_interval} and episode % {ac_telemetry_interval} == 0:
        mean_reward_50 = float(np.mean(ep_rewards[-{ac_telemetry_interval}:]))
        mean_lines_50  = float(np.mean(ep_lines[-{ac_telemetry_interval}:]))
        # telemetry POSTs here (catch all exceptions)
        if mean_reward_50 > best_reward:
            best_reward = mean_reward_50
            # save best model here

== ASTRA integration contract (ALL items MANDATORY) ==
1. Write this file once at startup (so play/eval endpoints detect the trainer):
     open("{checkpoint_dir}/trainer_type.txt", "w").write("lookahead_dqn")
2. Warm-start: if "{checkpoint_dir}/best_model.pth" exists, load with weights_only=False
   (MANDATORY — ActorCriticNet is a custom class and torch 2.6 requires this flag),
   into BOTH model and target_model:
     if os.path.exists("{checkpoint_dir}/best_model.pth"):
         checkpoint = torch.load("{checkpoint_dir}/best_model.pth", weights_only=False)
         state = checkpoint.state_dict() if not isinstance(checkpoint, dict) else checkpoint
         model.load_state_dict(state)
         target_model.load_state_dict(state)
3. Track rolling mean_reward over last {ac_telemetry_interval} episodes.
   Every {ac_telemetry_interval} episodes POST only mean_reward to telemetry:
     POST {api_url}/telemetry/missions/{mission_id}/metrics
     json={{"mission_id": "{mission_id}", "name": "mean_reward",
            "value": mean_reward_50, "step": episode, "iteration": {current_iteration}}}
   DO NOT post goal-metric names (lines_cleared, food_eaten, etc.) during training —
   those are reserved for end-of-iteration eval rollouts by the orchestrator.
   Use timeout=2, catch all exceptions (telemetry is non-critical).
4. When rolling mean_reward improves over best seen so far:
     torch.save(model, "{checkpoint_dir}/best_model.pth")
     open("{checkpoint_dir}/best_score.txt", "w").write(str(best_reward))
     open("{checkpoint_dir}/best_model_algo.txt", "w").write("LookaheadDQN")
5. After the timestep budget ({total_timesteps} steps): torch.save(model, "{checkpoint_dir}/last_model.pth")

Return ONLY the raw Python script. No markdown fences, no explanation."""

_LOOKAHEAD_PPO_CONTRACT = """\
Generate a complete custom lookahead-PPO training script for Tetris using PyTorch.
Do NOT use Stable-Baselines3. Implement everything with PyTorch and plain Python.

Real incident this trainer exists to fix: vanilla SB3 PPO structurally cannot
compete on Tetris-v0 — it samples blind from a learned distribution over 40 raw
action indices with no knowledge of what each one produces, confirmed live
across dozens of pivots all plateaued at lines_cleared≈0-1. This trainer uses
the SAME get_next_states() lookahead action-selection spine as the proven
Actor-Critic trainer (which hit 394 lines in 3 iterations): evaluate every
legal placement's actual resulting board through the value network, act
greedily on it. There is deliberately no stochastic policy head — once actions
are chosen by exhaustive lookahead, a probability distribution over 40 raw
indices has no role to play. What makes this recognizably PPO (not just a copy
of Actor-Critic) is retaining PPO's actual distinguishing mechanism: batched
ON-POLICY rollouts, GAE(λ) advantage estimation, and MULTIPLE EPOCHS of
clipped value updates per batch — genuinely different training dynamics from
Actor-Critic's single-pass off-policy replay updates.

{env_setup}
Mission ID: {mission_id}
Environment: {env_id}
Checkpoint directory: {checkpoint_dir}
Telemetry URL: {api_url}/telemetry/missions/{mission_id}/metrics
Target: lines_cleared >= {target_lines}
Hyperparameters: {hyperparameters}

== Mandatory imports and setup (copy exactly) ==
from envs.actor_critic_net import ActorCriticNet   # MANDATORY — do NOT redefine inline
model = ActorCriticNet()                            # value network: shared MLP [4→64→64] + critic head Linear(64,1)
optimizer = torch.optim.Adam(model.parameters(), lr={lr})

== Training skeleton (follow exactly — do NOT deviate from these API calls) ==
env = gym.make("{env_id}")          # MANDATORY — must appear before the training loop
ep_rewards, ep_lines = [], []
_eps_path = "{checkpoint_dir}/epsilon.txt"  # persist epsilon across iterations — otherwise every fresh script resets to 1.0 (near-total random exploration) even though the model itself is warm-started, making training-time mean_reward meaningless relative to the model's real learned quality
epsilon = float(open(_eps_path).read().strip()) if os.path.exists(_eps_path) else 1.0
best_reward = float("-inf")
total_steps = 0
episode = 0
_last_posted_episode = -1  # the outer loop runs once per ROLLOUT, not once per episode — an episode can span multiple rollouts, so episode may not change between passes; without this guard, "episode % {ac_telemetry_interval} == 0" stays true and re-posts the identical payload on every subsequent pass until a new episode finally completes

while total_steps < {total_timesteps}:
    # 1. Collect one on-policy rollout of {n_steps} steps (or until episode ends)
    rollout = []   # list of (obs, reward, next_obs, done)
    obs, _ = env.reset()
    ep_reward, ep_lines_cleared = 0.0, 0
    for _ in range({n_steps}):
        next_states = env.unwrapped.get_next_states()
        if not next_states:
            action = 0
        elif random.random() < epsilon:
            action = random.choice(list(next_states.keys()))
        else:
            with torch.no_grad():
                action = max(
                    next_states,
                    key=lambda a: model(
                        torch.tensor(next_states[a], dtype=torch.float32).unsqueeze(0)
                    ).item()
                )
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        rollout.append((obs[:4], reward, next_obs[:4], float(done)))  # ActorCriticNet takes the 4 board-quality features only — obs/next_obs may carry extra piece-identity dims for other trainers
        obs = next_obs
        ep_reward += reward
        total_steps += 1
        if done:
            ep_lines_cleared = info.get("lines_cleared", 0)
            ep_rewards.append(ep_reward); ep_lines.append(ep_lines_cleared)
            episode += 1
            obs, _ = env.reset()
            ep_reward, ep_lines_cleared = 0.0, 0
        if total_steps >= {total_timesteps}:
            break

    # 2. Compute GAE(λ) advantages/returns over the rollout (the PPO-defining step)
    s, r, ns, d = zip(*rollout)
    s  = torch.tensor(s,  dtype=torch.float32)
    ns = torch.tensor(ns, dtype=torch.float32)
    r  = torch.tensor(r,  dtype=torch.float32)
    d  = torch.tensor(d,  dtype=torch.float32)
    with torch.no_grad():
        values      = model(s).squeeze(-1)
        next_values = model(ns).squeeze(-1)
        deltas = r + {gamma} * next_values * (1 - d) - values
        advantages = torch.zeros_like(deltas)
        gae = 0.0
        for t in reversed(range(len(deltas))):
            gae = deltas[t] + {gamma} * {gae_lambda} * (1 - d[t]) * gae
            advantages[t] = gae
        returns = advantages + values
        old_values = values.clone()

    # 3. {n_epochs} epochs of clipped-value minibatch updates (the OTHER PPO-defining step)
    n = len(rollout)
    for _epoch in range({n_epochs}):
        perm = torch.randperm(n)
        for start in range(0, n, {batch_size}):
            idx = perm[start:start + {batch_size}]
            v_pred = model(s[idx]).squeeze(-1)
            v_clipped = old_values[idx] + torch.clamp(v_pred - old_values[idx], -{clip_range}, {clip_range})
            loss = torch.max((v_pred - returns[idx]) ** 2, (v_clipped - returns[idx]) ** 2).mean()
            optimizer.zero_grad(); loss.backward(); optimizer.step()

    epsilon = max({epsilon_min}, epsilon * {epsilon_decay})
    open(_eps_path, "w").write(str(epsilon))  # persist so the NEXT iteration's fresh script resumes decay instead of resetting to 1.0

    if len(ep_rewards) >= {ac_telemetry_interval} and episode % {ac_telemetry_interval} == 0 and episode > 0 and episode != _last_posted_episode:
        _last_posted_episode = episode
        mean_reward_50 = float(np.mean(ep_rewards[-{ac_telemetry_interval}:]))
        mean_lines_50  = float(np.mean(ep_lines[-{ac_telemetry_interval}:]))
        # telemetry POSTs here (catch all exceptions)
        if mean_reward_50 > best_reward:
            best_reward = mean_reward_50
            # save best model here

== ASTRA integration contract (ALL items MANDATORY) ==
1. Write this file once at startup (so play/eval endpoints detect the trainer):
     open("{checkpoint_dir}/trainer_type.txt", "w").write("lookahead_ppo")
2. Warm-start: if "{checkpoint_dir}/best_model.pth" exists, load with weights_only=False
   (MANDATORY — ActorCriticNet is a custom class and torch 2.6 requires this flag):
     if os.path.exists("{checkpoint_dir}/best_model.pth"):
         checkpoint = torch.load("{checkpoint_dir}/best_model.pth", weights_only=False)
         if isinstance(checkpoint, dict):
             model.load_state_dict(checkpoint)
         else:
             model.load_state_dict(checkpoint.state_dict())
3. Track rolling mean_reward over last {ac_telemetry_interval} episodes.
   Every {ac_telemetry_interval} episodes (checked after each rollout's episodes complete)
   POST only mean_reward to telemetry:
     POST {api_url}/telemetry/missions/{mission_id}/metrics
     json={{"mission_id": "{mission_id}", "name": "mean_reward",
            "value": mean_reward_50, "step": episode, "iteration": {current_iteration}}}
   DO NOT post goal-metric names (lines_cleared, food_eaten, etc.) during training —
   those are reserved for end-of-iteration eval rollouts by the orchestrator.
   Use timeout=2, catch all exceptions (telemetry is non-critical).
4. When rolling mean_reward improves over best seen so far:
     torch.save(model, "{checkpoint_dir}/best_model.pth")
     open("{checkpoint_dir}/best_score.txt", "w").write(str(best_reward))
     open("{checkpoint_dir}/best_model_algo.txt", "w").write("LookaheadPPO")
5. After the timestep budget ({total_timesteps} steps): torch.save(model, "{checkpoint_dir}/last_model.pth")

Return ONLY the raw Python script. No markdown fences, no explanation."""

_LOOKAHEAD_A2C_CONTRACT = """\
Generate a complete custom lookahead-A2C training script for Tetris using PyTorch.
Do NOT use Stable-Baselines3. Implement everything with PyTorch and plain Python.

Real incident this trainer exists to fix: vanilla SB3 A2C structurally cannot
compete on Tetris-v0 — it samples blind from a learned distribution over 40 raw
action indices with no knowledge of what each one produces, confirmed live
across multiple pivots all plateaued at lines_cleared≈0-1. This trainer uses
the SAME get_next_states() lookahead action-selection spine as the proven
Actor-Critic trainer (which hit 394 lines in 3 iterations): evaluate every
legal placement's actual resulting board through the value network, act
greedily on it. There is deliberately no stochastic policy head — once actions
are chosen by exhaustive lookahead, a probability distribution over 40 raw
indices has no role to play. What makes this recognizably A2C (not just a copy
of Actor-Critic) is retaining A2C's actual defining trait: a SHORT on-policy
rollout ({n_steps} steps) followed by a SINGLE synchronous advantage update,
with the rollout then discarded — no replay buffer (unlike lookahead-DQN), no
multi-epoch reuse (unlike lookahead-PPO). This is the simplest, lightest-weight
of the three variants, matching A2C's own position relative to DQN and PPO.

{env_setup}
Mission ID: {mission_id}
Environment: {env_id}
Checkpoint directory: {checkpoint_dir}
Telemetry URL: {api_url}/telemetry/missions/{mission_id}/metrics
Target: lines_cleared >= {target_lines}
Hyperparameters: {hyperparameters}

== Mandatory imports and setup (copy exactly) ==
from envs.actor_critic_net import ActorCriticNet   # MANDATORY — do NOT redefine inline
model = ActorCriticNet()                            # value network: shared MLP [4→64→64] + critic head Linear(64,1)
optimizer = torch.optim.Adam(model.parameters(), lr={lr})

== Training skeleton (follow exactly — do NOT deviate from these API calls) ==
env = gym.make("{env_id}")          # MANDATORY — must appear before the training loop
ep_rewards, ep_lines = [], []
_eps_path = "{checkpoint_dir}/epsilon.txt"  # persist epsilon across iterations — otherwise every fresh script resets to 1.0 (near-total random exploration) even though the model itself is warm-started, making training-time mean_reward meaningless relative to the model's real learned quality
epsilon = float(open(_eps_path).read().strip()) if os.path.exists(_eps_path) else 1.0
best_reward = float("-inf")
total_steps = 0
episode = 0
_last_posted_episode = -1  # the outer loop runs once per ROLLOUT, not once per episode — an episode can span multiple rollouts, so episode may not change between passes; without this guard, "episode % {ac_telemetry_interval} == 0" stays true and re-posts the identical payload on every subsequent pass until a new episode finally completes
obs, _ = env.reset()
ep_reward, ep_lines_cleared = 0.0, 0

while total_steps < {total_timesteps}:
    # 1. Collect one short on-policy rollout of {n_steps} steps — no replay buffer
    rollout = []   # list of (obs, reward, next_obs, done)
    for _ in range({n_steps}):
        next_states = env.unwrapped.get_next_states()
        if not next_states:
            action = 0
        elif random.random() < epsilon:
            action = random.choice(list(next_states.keys()))
        else:
            with torch.no_grad():
                action = max(
                    next_states,
                    key=lambda a: model(
                        torch.tensor(next_states[a], dtype=torch.float32).unsqueeze(0)
                    ).item()
                )
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        rollout.append((obs[:4], reward, next_obs[:4], float(done)))  # ActorCriticNet takes the 4 board-quality features only — obs/next_obs may carry extra piece-identity dims for other trainers
        obs = next_obs
        ep_reward += reward
        total_steps += 1
        if done:
            ep_lines_cleared = info.get("lines_cleared", 0)
            ep_rewards.append(ep_reward); ep_lines.append(ep_lines_cleared)
            episode += 1
            obs, _ = env.reset()
            ep_reward, ep_lines_cleared = 0.0, 0
        if total_steps >= {total_timesteps}:
            break

    # 2. Single synchronous advantage update over the rollout — the A2C-defining
    #    step: ONE gradient step per rollout, then the rollout is discarded.
    s, r, ns, d = zip(*rollout)
    s  = torch.tensor(s,  dtype=torch.float32)
    ns = torch.tensor(ns, dtype=torch.float32)
    r  = torch.tensor(r,  dtype=torch.float32).unsqueeze(1)
    d  = torch.tensor(d,  dtype=torch.float32).unsqueeze(1)
    with torch.no_grad():
        td_target = r + {gamma} * model(ns) * (1 - d)
    loss = nn.MSELoss()(model(s), td_target)
    optimizer.zero_grad(); loss.backward(); optimizer.step()

    epsilon = max({epsilon_min}, epsilon * {epsilon_decay})
    open(_eps_path, "w").write(str(epsilon))  # persist so the NEXT iteration's fresh script resumes decay instead of resetting to 1.0

    if len(ep_rewards) >= {ac_telemetry_interval} and episode % {ac_telemetry_interval} == 0 and episode > 0 and episode != _last_posted_episode:
        _last_posted_episode = episode
        mean_reward_50 = float(np.mean(ep_rewards[-{ac_telemetry_interval}:]))
        mean_lines_50  = float(np.mean(ep_lines[-{ac_telemetry_interval}:]))
        # telemetry POSTs here (catch all exceptions)
        if mean_reward_50 > best_reward:
            best_reward = mean_reward_50
            # save best model here

== ASTRA integration contract (ALL items MANDATORY) ==
1. Write this file once at startup (so play/eval endpoints detect the trainer):
     open("{checkpoint_dir}/trainer_type.txt", "w").write("lookahead_a2c")
2. Warm-start: if "{checkpoint_dir}/best_model.pth" exists, load with weights_only=False
   (MANDATORY — ActorCriticNet is a custom class and torch 2.6 requires this flag):
     if os.path.exists("{checkpoint_dir}/best_model.pth"):
         checkpoint = torch.load("{checkpoint_dir}/best_model.pth", weights_only=False)
         if isinstance(checkpoint, dict):
             model.load_state_dict(checkpoint)
         else:
             model.load_state_dict(checkpoint.state_dict())
3. Track rolling mean_reward over last {ac_telemetry_interval} episodes.
   Every {ac_telemetry_interval} episodes (checked after each rollout's episodes complete)
   POST only mean_reward to telemetry:
     POST {api_url}/telemetry/missions/{mission_id}/metrics
     json={{"mission_id": "{mission_id}", "name": "mean_reward",
            "value": mean_reward_50, "step": episode, "iteration": {current_iteration}}}
   DO NOT post goal-metric names (lines_cleared, food_eaten, etc.) during training —
   those are reserved for end-of-iteration eval rollouts by the orchestrator.
   Use timeout=2, catch all exceptions (telemetry is non-critical).
4. When rolling mean_reward improves over best seen so far:
     torch.save(model, "{checkpoint_dir}/best_model.pth")
     open("{checkpoint_dir}/best_score.txt", "w").write(str(best_reward))
     open("{checkpoint_dir}/best_model_algo.txt", "w").write("LookaheadA2C")
5. After the timestep budget ({total_timesteps} steps): torch.save(model, "{checkpoint_dir}/last_model.pth")

Return ONLY the raw Python script. No markdown fences, no explanation."""

_SFT_TEMPLATE = """\
Generate a complete SFT (QLoRA) fine-tuning script using HuggingFace + PEFT.

Mission ID: {mission_id}
Base model: {base_model}
Dataset path: {dataset_path}
LoRA config: r={lora_r}, alpha={lora_alpha}, dropout={lora_dropout}
Training args: batch={batch_size}, lr={learning_rate}, epochs={num_epochs}
save_strategy: steps, save_steps: {save_steps}
Checkpoint directory: {checkpoint_dir}
Telemetry URL: {api_url}/telemetry/missions/{mission_id}/metrics

The script must:
1. Load the base model in 4-bit (BitsAndBytes).
2. Apply LoRA via peft.get_peft_model.
3. Load the dataset from {dataset_path}.
4. Train using trl.SFTTrainer.
5. POST eval_loss to the telemetry endpoint after each save step.
6. Exit cleanly when eval_loss ≤ target."""

_MLX_LORA_TEMPLATE = """\
Generate a complete MLX LoRA fine-tuning script using mlx_lm.

Mission ID: {mission_id}
Base model: {base_model}
Train dataset: {train_dataset}
Valid dataset: {valid_dataset}
LoRA: rank={lora_rank}, scale={lora_scale}, dropout={lora_dropout}, layers={num_layers}
Training: batch={batch_size}, lr={learning_rate}, iters={iters}
save_every={save_every}, steps_per_eval={steps_per_eval}, steps_per_report={steps_per_report}
val_batches={val_batches}, max_seq_length={max_seq_length}
mask_prompt={mask_prompt}, grad_checkpoint={grad_checkpoint}
Adapter output: {checkpoint_dir}
Telemetry URL: {api_url}/telemetry/missions/{mission_id}/metrics

The script must:
1. Import subprocess, requests, json, logging, os, re.
2. Build and run this mlx_lm.lora command via subprocess.run (check=True):
       python -m mlx_lm.lora \\
         --model {base_model} \\
         --train --data <parent dir of train_dataset> \\
         --adapter-path {checkpoint_dir} \\
         --batch-size {batch_size} \\
         --learning-rate {learning_rate} \\
         --iters {iters} \\
         --rank {lora_rank} \\
         --scale {lora_scale} \\
         --dropout {lora_dropout} \\
         --num-layers {num_layers} \\
         --save-every {save_every} \\
         --steps-per-eval {steps_per_eval} \\
         --steps-per-report {steps_per_report} \\
         --val-batches {val_batches} \\
         --max-seq-length {max_seq_length} \\
         {mask_prompt_flag} \\
         {grad_checkpoint_flag}
3. Capture stdout/stderr. Parse lines matching "Val loss: <float>" to extract eval_loss.
4. POST each eval_loss to the telemetry endpoint with metric name "eval_loss" and step=iteration.
5. After training, POST the final eval_loss.
6. Exit 0 on success, 1 on error with full traceback."""

_DPO_TEMPLATE = """\
Generate a script that runs DPO (Direct Preference Optimization) fine-tuning by
invoking the EXISTING dpo_train.py script — do NOT reimplement the DPO training
loop, pair collection, or loss math yourself. This script is a thin orchestration
wrapper only — it does NOT report telemetry itself. Astra tails this process's
own log remotely and parses "Pass rate" lines on its own; do not add any
network calls (no requests, no HTTP) to this script.

Mission ID: {mission_id}
Finetune dir (remote host, dpo_train.py lives here): {finetune_dir}
Python interpreter (has mlx_lm installed): {python_bin}
Base model: {base_model}
Warm-start / reference adapter: {adapter}
Prompt template: {prompt_template}
LoRA: rank={lora_rank}, scale={lora_scale}, dropout={lora_dropout}, layers={num_layers}
Collection: K={k_collect}, temp={temp}, max_tokens={max_tokens}, email_weight={email_weight}
Training: epochs={epochs}, beta={beta}, lr={learning_rate}, max_grad_norm={max_grad_norm}
steps_per_eval={steps_per_eval}, eval_max_tokens={eval_max_tokens}
Adapter output: {checkpoint_dir}
Save collected pairs to: {save_pairs_path}

The script must:
1. Import sys, os only. Do NOT import subprocess or requests, and do NOT make any
   network calls. Use os.execv, NOT subprocess.run — this is CRITICAL: astra's
   sandbox tracks this wrapper process's own pid as "the training process." If
   this script fork+execs dpo_train.py as a child (subprocess.run/Popen), the
   child gets a DIFFERENT pid that astra never learns and can never clean up —
   if this wrapper dies or is killed for any reason, the child is silently
   orphaned, still running, invisible to astra. os.execv REPLACES this process's
   image in place (no fork, same pid) so the wrapper's tracked pid and the
   actual training process's pid are always identical, with zero orphan risk.
2. First os.chdir("{finetune_dir}") — dpo_train.py resolves --prompt-template AND
   its own hardcoded eval-cases path relative to the process's working directory,
   not relative to the script's own location. Without this chdir, both of those
   relative-path loads fail.
3. Ensure the log directory exists: os.makedirs(os.path.dirname("{save_pairs_path}"), exist_ok=True).
4. Then os.execv with this EXACT argv (note argv[0] repeats the interpreter path,
   standard C-style exec convention) — this call does not return; the process
   image becomes dpo_train.py from this point on, and its exit code becomes
   this process's exit code automatically (no sys.exit needed, no return value
   to check):
       os.execv(
           "{python_bin}",
           [
               "{python_bin}", "{finetune_dir}/dpo_train.py",
               "--model", "{base_model}",
               "--adapter", "{adapter}",
               "--save-dir", "{checkpoint_dir}",
               "--num-layers", "{num_layers}",
               "--lora-rank", "{lora_rank}",
               "--lora-dropout", "{lora_dropout}",
               "--lora-scale", "{lora_scale}",
               "--K-collect", "{k_collect}",
               "--temp", "{temp}",
               "--max-tokens", "{max_tokens}",
               "--email-weight", "{email_weight}",
               {routing_only_flag}
               "--epochs", "{epochs}",
               "--beta", "{beta}",
               "--learning-rate", "{learning_rate}",
               "--max-grad-norm", "{max_grad_norm}",
               "--steps-per-eval", "{steps_per_eval}",
               "--eval-max-tokens", "{eval_max_tokens}",
               "--prompt-template", "{prompt_template}",
               "--save-pairs", "{save_pairs_path}",
           ],
       )
   ({routing_only_flag} is either the string "--routing-only", (with a trailing
   comma, as its own list element) or omitted entirely if empty — do not insert
   an empty string element in the list.)
   Always pass --save-pairs (do not make it conditional) — it persists the
   collected (chosen, rejected) pairs to JSONL after the collection phase, so a
   future run can pass --load-pairs to skip re-collection (the slowest phase,
   ~30-60 min) when only retraining hyperparameters (beta/epochs) change."""

_GRPO_TEMPLATE = """\
Generate a script that runs GRPO (Group Relative Policy Optimization) fine-tuning by
invoking the EXISTING grpo_train.py script — do NOT reimplement the on-policy rollout,
reward scoring, advantage computation, or PPO-clipped update yourself. This script is a
thin orchestration wrapper only — it does NOT report telemetry itself. Astra tails this
process's own log remotely and parses "Pass rate" lines on its own; do not add any
network calls (no requests, no HTTP) to this script.

Mission ID: {mission_id}
Finetune dir (remote host, grpo_train.py lives here): {finetune_dir}
Python interpreter (has mlx_lm installed): {python_bin}
Base model: {base_model}
Warm-start adapter: {adapter}
Prompt template: {prompt_template}
LoRA: rank={lora_rank}, scale={lora_scale}, dropout={lora_dropout}, layers={num_layers}
GRPO: iters={iters}, num_generations={num_generations}, temp={temp}, max_tokens={max_tokens}
eval_max_tokens={eval_max_tokens}, reward_schema={reward_schema}
email_weight={email_weight}, focus_weight={focus_weight}
max_zero_steps={max_zero_steps}, max_one_steps={max_one_steps}
Optimizer: lr={learning_rate}, clip_epsilon={clip_epsilon}, max_grad_norm={max_grad_norm}
Logging: steps_per_report={steps_per_report}, steps_per_eval={steps_per_eval}, save_every={save_every}
Adapter output: {checkpoint_dir}

The script must:
1. Import sys, os only. Do NOT import subprocess or requests, and do NOT make any
   network calls. Use os.execv, NOT subprocess.run — this is CRITICAL: astra's
   sandbox tracks this wrapper process's own pid as "the training process." If
   this script fork+execs grpo_train.py as a child (subprocess.run/Popen), the
   child gets a DIFFERENT pid that astra never learns and can never clean up —
   if this wrapper dies or is killed for any reason, the child is silently
   orphaned, still running, invisible to astra. os.execv REPLACES this process's
   image in place (no fork, same pid) so the wrapper's tracked pid and the
   actual training process's pid are always identical, with zero orphan risk.
2. First os.chdir("{finetune_dir}") — grpo_train.py resolves --prompt-template AND
   its own hardcoded eval-cases path relative to the process's working directory,
   not relative to the script's own location. Without this chdir, both of those
   relative-path loads fail.
3. Then os.execv with this EXACT argv (note argv[0] repeats the interpreter path,
   standard C-style exec convention) — this call does not return; the process
   image becomes grpo_train.py from this point on, and its exit code becomes
   this process's exit code automatically (no sys.exit needed, no return value
   to check):
       os.execv(
           "{python_bin}",
           [
               "{python_bin}", "{finetune_dir}/grpo_train.py",
               "--model", "{base_model}",
               "--adapter", "{adapter}",
               "--save-dir", "{checkpoint_dir}",
               "--num-layers", "{num_layers}",
               "--lora-rank", "{lora_rank}",
               "--lora-dropout", "{lora_dropout}",
               "--lora-scale", "{lora_scale}",
               "--iters", "{iters}",
               "--num-generations", "{num_generations}",
               "--temp", "{temp}",
               "--max-tokens", "{max_tokens}",
               "--eval-max-tokens", "{eval_max_tokens}",
               "--reward-schema", "{reward_schema}",
               "--email-weight", "{email_weight}",
               "--focus-weight", "{focus_weight}",
               "--max-zero-steps", "{max_zero_steps}",
               "--max-one-steps", "{max_one_steps}",
               "--learning-rate", "{learning_rate}",
               "--clip-epsilon", "{clip_epsilon}",
               "--max-grad-norm", "{max_grad_norm}",
               "--steps-per-report", "{steps_per_report}",
               "--steps-per-eval", "{steps_per_eval}",
               "--save-every", "{save_every}",
               {routing_only_flag}
               "--prompt-template", "{prompt_template}",
           ],
       )
   ({routing_only_flag} is either the string "--routing-only", (with a trailing
   comma, as its own list element) or omitted entirely if empty — do not insert
   an empty string element in the list.)"""

_ML_TEMPLATE = """\
Generate a complete ML training script.

Mission ID: {mission_id}
Framework: {framework}
Model class: {model_class}
Dataset path: {dataset_path}
Target column: {target_column}
Model params: {model_params}
Checkpoint directory: {checkpoint_dir}
Telemetry URL: {api_url}/telemetry/missions/{mission_id}/metrics

The script must:
1. Load the dataset:
   - If {dataset_path} is a well-known sklearn dataset name ("iris", "digits",
     "breast_cancer", "wine"), call sklearn.datasets.load_{dataset_path}() and use
     the returned .data and .target arrays directly as X and y. Do NOT convert to a
     DataFrame and do NOT import pandas or numpy for this step.
   - Otherwise load from {dataset_path} using pandas (CSV) or json.
2. Split into train/val sets using train_test_split(X, y, ...).
3. Train the model.
4. POST accuracy to the telemetry endpoint with metric name "accuracy".
   Use response.ok (2xx) to check success; log a warning on failure but do NOT exit — telemetry is non-critical.
5. Save the model using this EXACT line — do not change the path:
       import joblib; joblib.dump(model, "{checkpoint_dir}/model.joblib")
6. Exit cleanly (exit(0) on success, exit(1) on error with traceback)."""


def _resolve_env_kwargs(env_id: str, plan_env_kwargs: Optional[dict]) -> dict:
    """Merge plan env_kwargs with recipe env_kwargs defaults."""
    kw = dict(plan_env_kwargs or {})
    recipe_kw = _load_recipe_for_env(env_id).get("env_kwargs", {}) or {}
    for k, v in recipe_kw.items():
        kw.setdefault(k, v)
    return kw


# Valid constructor kwargs per SB3 algorithm — used to filter _hp before model(...)
_VALID_ALGO_KEYS: dict[str, set] = {
    "PPO": {
        "learning_rate", "n_steps", "batch_size", "n_epochs", "gamma",
        "gae_lambda", "clip_range", "clip_range_vf", "ent_coef",
        "vf_coef", "max_grad_norm", "target_kl",
    },
    "DQN": {
        "learning_rate", "buffer_size", "learning_starts", "batch_size",
        "tau", "gamma", "train_freq", "gradient_steps",
        "target_update_interval", "exploration_fraction",
        "exploration_initial_eps", "exploration_final_eps", "max_grad_norm",
        # Real incident: optimize_memory_usage was here and reachable from
        # LLM pivot proposals — SB3's ReplayBuffer raises ValueError when
        # combined with the default handle_timeout_termination=True ("does
        # not support optimize_memory_usage=True and
        # handle_timeout_termination=True simultaneously"), and
        # handle_timeout_termination isn't exposed here either, so there's
        # no way to safely pair them. This already crashed one Snake-v0
        # mission (see snake_dqn_v1.yaml's v1.1.1 changelog — fixed there
        # only by dropping it from that one recipe's hyperparameters) and
        # recurred via a live pivot on a Tetris-v0 mission, since a
        # recipe-level omission does nothing to stop the LLM from proposing
        # it again in a pivot. Removed here instead, at the actual
        # reachable-key allowlist, so it can never be proposed again for
        # any DQN mission regardless of recipe or pivot.
    },
    "SAC": {
        "learning_rate", "buffer_size", "learning_starts", "batch_size",
        "tau", "gamma", "train_freq", "gradient_steps", "ent_coef",
        "target_update_interval", "target_entropy", "use_sde",
        "sde_sample_freq", "use_sde_at_warmup",
    },
    "A2C": {
        "learning_rate", "n_steps", "gamma", "gae_lambda", "ent_coef",
        "vf_coef", "max_grad_norm", "rms_prop_eps", "use_rms_prop",
        "use_sde", "sde_sample_freq", "normalize_advantage",
    },
    "TD3": {
        "learning_rate", "buffer_size", "learning_starts", "batch_size",
        "tau", "gamma", "train_freq", "gradient_steps",
        "action_noise", "target_policy_noise", "target_noise_clip",
        "policy_delay",
    },
}

# Canonical recipe file per env_id or task_type — hyperparameters and env_kwargs
# from these files are used as defaults when the LLM plan omits a value.
_ENV_RECIPE: dict = {
    "Snake-v0": "snake_ppo_v1.yaml",
    "Snake-v0/DQN": "snake_dqn_v1.yaml",   # algorithm-specific override
    "Tetris-v0": "tetris_actor_critic_v1.yaml",
    "sft": "sft_llama_lora_v1.yaml",       # keyed by task_type for non-RL tasks
    "mlx_lora": "mlx_lora_v1.yaml",
    "dpo": "ensemble_dpo_v1.yaml",
    "grpo": "ensemble_grpo_v1.yaml",
}


def _load_recipe_for_env(env_id: str, algorithm: str = "") -> dict:
    """Return the parsed recipe dict for env_id (+ optional algorithm), or {}."""
    # Algorithm-specific override takes priority (e.g. Snake-v0/DQN)
    filename = _ENV_RECIPE.get(f"{env_id}/{algorithm}") or _ENV_RECIPE.get(env_id)
    if not filename:
        return {}
    try:
        import yaml as _yaml
        path = os.path.join(settings.recipes_path, filename)
        with open(path) as f:
            return _yaml.safe_load(f) or {}
    except Exception:
        return {}


def _resolve_hyperparams(env_id: str, plan_hp: dict, algorithm: str = "") -> dict:
    """Apply recipe hyperparameters as defaults for keys the LLM plan did not set."""
    recipe_hp = _load_recipe_for_env(env_id, algorithm).get("hyperparameters", {})
    if env_id in ("dpo", "grpo"):
        # Recipe is authoritative, full stop — no plan/pivot override allowed.
        # These are LoRA/optimizer settings tuned against a specific warm-start
        # adapter; PIVOT_SYSTEM's hyperparameter guidance is RL-oriented (PPO/DQN
        # ranges) and doesn't know these are recipe-locked. Confirmed via a real
        # incident: a pivot's generic learning_rate=0.001 silently overrode the
        # recipe's 5e-7 through the old setdefault-only merge below, collapsing
        # a DPO run's pass_rate from a 62% baseline to 0% within 50 steps.
        return dict(recipe_hp)
    hp = dict(plan_hp)
    for k, v in recipe_hp.items():
        hp.setdefault(k, v)
    return hp


def finetune_checkpoint_dir(task_type: str, plan: dict, mission_id: str) -> str:
    """Remote checkpoint/adapter output dir for dpo/grpo missions — under
    finetune_dir/adapters/, the same directory ensemble/finetune's own manual
    workflow uses (grpo_v<N>_min/, dpo_v<N>_min/, retrain_best/, ...), so
    astra-produced adapters are part of the normal inventory rather than a
    separate mission-scoped location. Shared by CodeGenerator.generate_training_script
    (to bake --save-dir into the wrapper script) and SandboxManager.launch (to know
    where SSHSandbox._sync_back() should rsync the adapter from)."""
    hp = _resolve_hyperparams(task_type, plan.get("hyperparameters", {}))
    finetune_dir = hp.get("finetune_dir", "")
    return os.path.join(finetune_dir, "adapters", f"astra_{mission_id[:8]}")


class CodeGenerator:
    def __init__(self, provider: InferenceProvider) -> None:
        self._provider = provider

    @staticmethod
    def valid_algo_keys(algorithm: str) -> set:
        """Return the set of valid SB3 constructor kwargs for an algorithm (empty = unknown)."""
        return _VALID_ALGO_KEYS.get(algorithm.upper(), set())

    async def generate_training_script(
        self,
        mission_id: str,
        plan: dict,
        current_iteration: int = 0,
    ) -> str:
        """
        Generate a training script and write it to data/missions/{id}/train.py.
        Returns the path to the written script.
        """
        task_type = plan.get("task_type", "rl")
        if task_type in _FINETUNE_REMOTE_TASK_TYPES and settings.sandbox_host:
            checkpoint_dir = finetune_checkpoint_dir(task_type, plan, mission_id)
        else:
            checkpoint_dir = os.path.join(settings.data_path, "missions", mission_id, "checkpoints")
            os.makedirs(checkpoint_dir, exist_ok=True)

        system_prompt = _BASE_SYSTEM.format(
            api_url=f"http://{settings.telemetry_host}:{settings.api_port}",
            mission_id=mission_id,
            checkpoint_dir=checkpoint_dir,
        )

        # Inject lessons learned from prior code generation failures
        lessons = self._query_lessons(plan)
        if lessons:
            lesson_block = "\n".join(f"- {l}" for l in lessons)
            system_prompt += f"\n\nLessons learned from prior failures (avoid repeating these):\n{lesson_block}"

        user_prompt = self._build_user_prompt(task_type, mission_id, plan, checkpoint_dir, current_iteration)

        messages = [
            Message(role="system", content=system_prompt),
            Message(role="user", content=user_prompt),
        ]
        code = await self._provider.generate(messages, GenerationConfig(max_tokens=4096, temperature=0.1))
        code = self._strip_fences(code)
        if task_type == "rl":
            code = self._patch_rl_imports(code)
            code = self._patch_undefined_logger(code)
            env_id = plan.get("env_id", "")
            _proj_root = os.path.abspath(os.path.join(settings.data_path, ".."))
            if env_id == "Snake-v0" and "register" not in code:
                code = _SNAKE_SETUP.format(project_root=_proj_root) + "\n" + code
                logger.info("CodeGenerator: injected Snake-v0 registration preamble")
            elif env_id == "Tetris-v0" and "register" not in code:
                code = _TETRIS_SETUP.format(project_root=_proj_root) + "\n" + code
                logger.info("CodeGenerator: injected Tetris-v0 registration preamble")
            # Inject curriculum loop if recipe defines phases
            _algo = plan.get("algorithm", "PPO")
            _recipe = _load_recipe_for_env(env_id, _algo)
            _curriculum_phases = (_recipe.get("curriculum") or {}).get("phases")
            if _curriculum_phases:
                _env_kw = _resolve_env_kwargs(env_id, plan.get("env_kwargs"))
                _metric_name = next(iter(plan.get("target_metric") or {}), "food_eaten")
                code = self._inject_curriculum(code, _curriculum_phases, env_id, _env_kw, _metric_name)
                logger.info("CodeGenerator: injected curriculum (%d phases) for %s/%s", len(_curriculum_phases), env_id, _algo)
        # Fix any relative checkpoint paths the LLM may have substituted for the absolute checkpoint_dir
        code = self._fix_checkpoint_paths(code, checkpoint_dir)

        script_path = os.path.abspath(os.path.join(settings.data_path, "missions", mission_id, "train.py"))
        os.makedirs(os.path.dirname(script_path), exist_ok=True)
        with open(script_path, "w") as f:
            f.write(code)

        # Write train_config.json so the play endpoint knows the algorithm and env_kwargs
        if task_type == "rl":
            import json as _json
            config_path = os.path.join(checkpoint_dir, "train_config.json")
            os.makedirs(checkpoint_dir, exist_ok=True)
            _env_id = plan.get("env_id", "")
            train_cfg = {
                "algorithm": plan.get("algorithm", "PPO"),
                "env_id": _env_id,
                "env_kwargs": _resolve_env_kwargs(_env_id, plan.get("env_kwargs")),
                "trainer_type": plan.get("trainer_type", ""),
            }
            with open(config_path, "w") as f:
                _json.dump(train_cfg, f)
            logger.info("CodeGenerator: wrote train_config.json algorithm=%s env_kwargs=%s trainer_type=%s",
                        train_cfg["algorithm"], train_cfg["env_kwargs"], train_cfg["trainer_type"])

        logger.info("Generated training script: %s (%d chars)", script_path, len(code))
        return script_path

    def _build_user_prompt(self, task_type: str, mission_id: str, plan: dict, checkpoint_dir: str, current_iteration: int = 0) -> str:
        recipe_key = plan.get("env_id", "") if task_type == "rl" else task_type
        _plan_algo = plan.get("algorithm", "PPO") if task_type == "rl" else ""
        hp = _resolve_hyperparams(recipe_key, plan.get("hyperparameters", {}), algorithm=_plan_algo)
        api_url = f"http://127.0.0.1:{settings.api_port}"
        base = {
            "mission_id": mission_id,
            "checkpoint_dir": checkpoint_dir,
            "api_url": api_url,
            "target_metric": json.dumps(plan.get("target_metric", {})),
        }
        if task_type == "rl":
            tm = plan.get("target_metric", {})
            tm_name = next(iter(tm), None) if tm else None
            tm_value = next(iter(tm.values()), 200) if tm else 200
            # Only use the target value as the mean_reward threshold when the
            # target metric IS mean_reward; for custom goals (food_eaten, etc.)
            # mean_reward scale is unrelated to the target — use a sentinel that
            # never triggers so the full timestep budget always runs.
            target_reward = tm_value if tm_name in (None, "mean_reward") else 9999
            env_id = plan.get("env_id", "CartPole-v1")
            # Inject custom env registration preamble when needed
            _project_root = os.path.abspath(os.path.join(settings.data_path, ".."))
            if env_id == "Snake-v0":
                env_setup = _SNAKE_SETUP.format(project_root=_project_root)
            elif env_id == "Tetris-v0":
                env_setup = _TETRIS_SETUP.format(project_root=_project_root)
            else:
                env_setup = ""
            hp = dict(hp)  # copy so we don't mutate the plan's hyperparameters dict
            policy_kwargs = hp.pop("policy_kwargs", None)
            # Build env_kwargs_str: ", key=value, ..." for gym.make() call
            env_kwargs = _resolve_env_kwargs(env_id, plan.get("env_kwargs"))
            if env_kwargs:
                env_kwargs_str = ", " + ", ".join(f"{k}={v!r}" for k, v in env_kwargs.items())
            else:
                env_kwargs_str = ""
            algorithm = plan.get("algorithm", "PPO")
            recipe_trainer_type = _load_recipe_for_env(recipe_key, algorithm).get("trainer_type", "")
            trainer_type = plan.get("trainer_type", "") or recipe_trainer_type
            if trainer_type == "actor_critic":
                tm_lines = next(iter(tm.values()), 20) if tm else 20
                lr = hp.get("learning_rate", 0.0001)
                ctx = {
                    "env_id": env_id,
                    "env_setup": env_setup,
                    "target_lines": tm_lines,
                    "hyperparameters": json.dumps(hp, indent=2),
                    "lr": lr,
                    "total_timesteps": hp.get("total_timesteps", 2000000),
                    "replay_buffer_size": hp.get("replay_buffer_size", 10000),
                    "batch_size": hp.get("batch_size", 64),
                    "gamma": hp.get("gamma", 0.99),
                    "epsilon_min": hp.get("epsilon_min", 0.01),
                    "epsilon_decay": hp.get("epsilon_decay", 0.9995),
                    "ac_telemetry_interval": hp.get("ac_telemetry_interval", 50),
                    "current_iteration": current_iteration,
                    **base,
                }
                return _ACTOR_CRITIC_CONTRACT.format(**ctx)
            if trainer_type in ("lookahead_dqn", "lookahead_ppo", "lookahead_a2c"):
                tm_lines = next(iter(tm.values()), 20) if tm else 20
                lr = hp.get("learning_rate", 0.0001)
                lookahead_ctx = {
                    "env_id": env_id,
                    "env_setup": env_setup,
                    "target_lines": tm_lines,
                    "hyperparameters": json.dumps(hp, indent=2),
                    "lr": lr,
                    "total_timesteps": hp.get("total_timesteps", 2000000),
                    "gamma": hp.get("gamma", 0.99),
                    "epsilon_min": hp.get("epsilon_min", 0.01),
                    "epsilon_decay": hp.get("epsilon_decay", 0.9995),
                    "ac_telemetry_interval": hp.get("ac_telemetry_interval", 50),
                    "current_iteration": current_iteration,
                    **base,
                }
                if trainer_type == "lookahead_dqn":
                    lookahead_ctx.update({
                        "replay_buffer_size": hp.get("replay_buffer_size", 10000),
                        "batch_size": hp.get("batch_size", 64),
                        "target_update_interval": hp.get("target_update_interval", 2000),
                    })
                    return _LOOKAHEAD_DQN_CONTRACT.format(**lookahead_ctx)
                if trainer_type == "lookahead_ppo":
                    lookahead_ctx.update({
                        "n_steps": hp.get("n_steps", 2048),
                        "n_epochs": hp.get("n_epochs", 10),
                        "batch_size": hp.get("batch_size", 64),
                        "gae_lambda": hp.get("gae_lambda", 0.95),
                        "clip_range": hp.get("clip_range", 0.2),
                    })
                    return _LOOKAHEAD_PPO_CONTRACT.format(**lookahead_ctx)
                # lookahead_a2c
                lookahead_ctx.update({"n_steps": hp.get("n_steps", 5)})
                return _LOOKAHEAD_A2C_CONTRACT.format(**lookahead_ctx)
            # Build algorithm-aware valid-keys filter for the model constructor block
            algo_upper = algorithm.upper()
            valid_keys = _VALID_ALGO_KEYS.get(algo_upper, _VALID_ALGO_KEYS["PPO"])
            valid_keys_var = f"_VALID_{algo_upper}_KEYS"
            valid_keys_set = "{" + ", ".join(f'"{k}"' for k in sorted(valid_keys)) + "}"
            ctx = {
                "algorithm": algorithm,
                "valid_keys_var": valid_keys_var,
                "valid_keys_set": valid_keys_set,
                "env_id": env_id,
                "hyperparameters": json.dumps(hp, indent=2),
                "policy_kwargs": json.dumps(policy_kwargs) if policy_kwargs else "None",
                "target_reward": target_reward,
                "env_setup": env_setup,
                "env_kwargs_str": env_kwargs_str,
                "current_iteration": current_iteration,
                "total_timesteps": hp.get("total_timesteps", 2000000),
                "telemetry_interval": hp.get("telemetry_interval", 2048),
                **base,
            }
            return _RL_TEMPLATE.format(**ctx)
        if task_type == "sft":
            ctx = {
                **hp,   # recipe + plan hyperparameters (base_model, lora_r, batch_size, etc.)
                **base,
            }
            return _SFT_TEMPLATE.format(**ctx)
        if task_type == "mlx_lora":
            dataset = plan.get("dataset", {})
            ctx = {
                **hp,
                "train_dataset": dataset.get("train", hp.get("train_dataset", "data/datasets/train.jsonl")),
                "valid_dataset": dataset.get("valid", hp.get("valid_dataset", "data/datasets/valid.jsonl")),
                "mask_prompt_flag": "--mask-prompt" if hp.get("mask_prompt", True) else "",
                "grad_checkpoint_flag": "--grad-checkpoint" if hp.get("grad_checkpoint", True) else "",
                **base,
            }
            return _MLX_LORA_TEMPLATE.format(**ctx)
        if task_type == "dpo":
            ctx = {
                **hp,
                "routing_only_flag": '"--routing-only",' if hp.get("routing_only", True) else "",
                "save_pairs_path": os.path.join(
                    hp.get("finetune_dir", ""), "logs", f"astra_{mission_id[:8]}_pairs.jsonl"
                ),
                **base,
            }
            return _DPO_TEMPLATE.format(**ctx)
        if task_type == "grpo":
            ctx = {
                **hp,
                "routing_only_flag": '"--routing-only",' if hp.get("routing_only", True) else "",
                **base,
            }
            return _GRPO_TEMPLATE.format(**ctx)
        # ml
        ctx = {
            "framework": "sklearn",
            "model_class": "RandomForestClassifier",
            "dataset_path": "dataset.csv",
            "target_column": "label",
            "model_params": json.dumps(hp.get("model_params", {}), indent=2),
            **hp,
            **base,
        }
        return _ML_TEMPLATE.format(**ctx)

    @staticmethod
    def _inject_curriculum(code: str, phases: list, env_id: str, env_kwargs: dict, metric_name: str = "food_eaten") -> str:
        """Replace single model.learn() with a multi-phase curriculum loop.

        Called after LLM generation so the loop is deterministic, not LLM-generated.
        obs_type=features keeps obs at 25D across all grid sizes — weights transfer fine.

        Real incident: this was hardcoded entirely for Snake-v0's curriculum shape
        (every phase dict assumed to have grid_h/grid_w/food_target/timesteps keys,
        tracking hardcoded to info["food_eaten"]) with no env-specific branching —
        it just happened to only ever be called for Snake-v0, since Tetris-v0's
        recipe never had a curriculum section until one was merged in from a
        crystallized recipe. That merge triggered this function for Tetris-v0's
        curriculum shape (max_lines_cleared/max_iterations, no grid_h/grid_w at
        all — the board is a fixed 20x10, not resizable) for the first time,
        crashing every phase transition with KeyError: 'grid_h'. Now: per-phase
        env kwargs (grid_h/grid_w) are only added if actually present in the
        phase dict; the phase-duration key is "timesteps" OR "max_iterations",
        whichever is present; and the tracked per-phase best metric comes from
        the mission's actual target_metric name (passed in as metric_name)
        instead of a hardcoded "food_eaten", read from the env's own info dict
        key of the same name (Snake: info["food_eaten"]; Tetris:
        info["lines_cleared"] — both envs already key their info dict by the
        exact target_metric name).
        """
        import re

        if not phases:
            return code
        phase0 = phases[0]
        duration_key = "timesteps" if "timesteps" in phase0 else (
            "max_iterations" if "max_iterations" in phase0 else None
        )
        if duration_key is None:
            logger.warning(
                "CodeGenerator: curriculum phases have neither 'timesteps' nor "
                "'max_iterations' — skipping curriculum injection: %s", phase0,
            )
            return code
        has_grid_dims = "grid_h" in phase0 and "grid_w" in phase0

        phases_repr = json.dumps(phases, indent=4)
        # env_kwargs without grid dims (those come from each phase, when present)
        base_kw = {k: v for k, v in env_kwargs.items() if k not in ("grid_h", "grid_w")}
        kw_str = (", " + ", ".join(f"{k}={v!r}" for k, v in base_kw.items())) if base_kw else ""
        grid_kw_str = ', grid_h=_ph["grid_h"], grid_w=_ph["grid_w"]' if has_grid_dims else ""

        curriculum_block = (
            f'_CURRICULUM_PHASES = {phases_repr}\n'
            f'for _ph_idx, _ph in enumerate(_CURRICULUM_PHASES):\n'
            f'    _ph_env = gym.make("{env_id}"{kw_str}{grid_kw_str})\n'
            f'    model.set_env(_ph_env)\n'
            f'    callback._phase_best_metric = 0\n'
            f'    logging.info("Curriculum phase %d/%d: %s", _ph_idx + 1, len(_CURRICULUM_PHASES), _ph)\n'
            f'    model.learn(total_timesteps=_ph["{duration_key}"], callback=callback,\n'
            f'                reset_num_timesteps=(_ph_idx == 0))\n'
            f'    logging.info("Phase %d done — best_{metric_name}=%d", _ph_idx + 1, callback._phase_best_metric)'
        )

        # Replace single model.learn(...) call with the curriculum loop
        code = re.sub(
            r'model\.learn\(total_timesteps=\d+,\s*callback=callback\)',
            curriculum_block,
            code,
        )

        # Patch callback __init__: add _phase_best_metric tracker
        code = code.replace(
            'self._best_reward = float("-inf")',
            'self._best_reward = float("-inf")\n        self._phase_best_metric = 0',
        )

        # Patch _on_step: insert per-phase metric tracking before the telemetry block
        metric_tracking = (
            '        for _info, _done in zip(self.locals.get("infos", []), self.locals.get("dones", [])):\n'
            f'            if _done and "{metric_name}" in _info:\n'
            f'                _mv = int(_info["{metric_name}"])\n'
            '                if _mv > self._phase_best_metric:\n'
            '                    self._phase_best_metric = _mv\n'
        )
        code = re.sub(
            r'(    def _on_step\(self\) -> bool:\n)',
            r'\1' + metric_tracking,
            code,
        )

        return code

    @staticmethod
    def _patch_rl_imports(code: str) -> str:
        """Inject any SB3 imports the LLM forgot to include."""
        import re
        # Replace `import stable_baselines3 as sb3` + `sb3.XXX` with direct imports
        if "import stable_baselines3 as sb3" in code:
            code = re.sub(r'\bsb3\.(PPO|SAC|A2C|DQN|TD3)\b', r'\1', code)
            code = re.sub(r'^\s*import stable_baselines3 as sb3\s*\n?', '', code, flags=re.MULTILINE)

        # Re-split after any string-level substitution above
        lines = code.splitlines()
        last_import_idx = 0
        for i, line in enumerate(lines):
            s = line.strip()
            if s.startswith("import ") or s.startswith("from "):
                last_import_idx = i

        import_lines = "\n".join(l for l in lines if l.strip().startswith(("import ", "from ")))
        to_inject = []
        if "import os" not in import_lines:
            to_inject.append("import os")
        for algo in ("PPO", "SAC", "A2C", "DQN", "TD3"):
            if algo in code and f"from stable_baselines3 import {algo}" not in import_lines:
                to_inject.append(f"from stable_baselines3 import {algo}")
        if "BaseCallback" in code and "from stable_baselines3.common.callbacks import BaseCallback" not in import_lines:
            to_inject.append("from stable_baselines3.common.callbacks import BaseCallback")
        if "CheckpointCallback" in code and "CheckpointCallback" not in import_lines:
            to_inject.append("from stable_baselines3.common.callbacks import CheckpointCallback")

        if to_inject:
            insert_at = last_import_idx + 1
            for i, imp in enumerate(to_inject):
                lines.insert(insert_at + i, imp)
            code = "\n".join(lines)
        # Fix callback constructor calls with unsupported kwargs
        code = re.sub(
            r'(\w*[Cc]allback\w*)\s*\(\s*(?:check_freq|checkpoint_freq|save_path)[^)]*\)',
            r'\1()',
            code,
        )
        # Fix callback class that has _on_step but doesn't extend BaseCallback
        code = re.sub(
            r'(class\s+\w*[Cc]allback\w*)\s*:',
            r'\1(BaseCallback):',
            code,
        )
        # Ensure __init__ accepts **kwargs to tolerate extra args
        if "class " in code and "(BaseCallback)" in code:
            if "def __init__(self):" in code:
                code = code.replace(
                    "def __init__(self):",
                    "def __init__(self, verbose=0, **kwargs):\n        super().__init__(verbose=verbose)",
                )
        return code

    @staticmethod
    def _patch_undefined_logger(code: str) -> str:
        """Replace bare logger.warning/error/info calls with logging.* when no logger is defined.
        LLMs frequently emit `logger.warning(...)` without defining `logger = logging.getLogger(...)`.
        """
        import re
        if re.search(r'\blogger\s*=', code):
            return code  # logger is explicitly defined — leave as-is
        # Replace logger.<level>(...) → logging.<level>(...)
        return re.sub(r'\blogger\.(warning|error|info|debug|critical)\b', r'logging.\1', code)

    @staticmethod
    def _fix_checkpoint_paths(code: str, checkpoint_dir: str) -> str:
        """Replace relative ./data/missions/.../checkpoints paths with the absolute checkpoint_dir."""
        import re
        # Match patterns like ./data/missions/<uuid>/checkpoints or data/missions/<uuid>/checkpoints
        pattern = r'\.?/?\bdata/missions/[0-9a-f-]{36}/checkpoints\b'
        fixed = re.sub(pattern, checkpoint_dir, code)
        if fixed != code:
            logger.info("CodeGenerator: replaced relative checkpoint paths with absolute: %s", checkpoint_dir)
        return fixed

    @staticmethod
    def _query_lessons(plan: dict) -> list[str]:
        """Retrieve past code generation failure lessons from vector memory."""
        try:
            from backend.services import vector_memory
            domain = plan.get("domain", "code_generation")
            results = vector_memory.query_lessons(
                f"{plan.get('algorithm', '')} {plan.get('task_type', '')} training script",
                domain=domain,
                n_results=3,
            )
            return [r["text"] for r in results if r.get("text")]
        except Exception:
            return []

    @staticmethod
    def _strip_fences(text: str) -> str:
        import re
        # Remove common LLM artifacts
        text = re.sub(r"<\|im_end\|>|<\|endoftext\|>", "", text)
        # Remove lines that are purely markdown fences or artifacts
        lines = []
        for line in text.splitlines():
            if line.strip().startswith("```"):
                continue
            lines.append(line)
        return "\n".join(lines).strip()
