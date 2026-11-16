"""
CodeGenerator — Step 3.2.

Produces runnable Python training scripts from a plan dict.
The script is written to data/missions/{id}/train.py and executed
by the SandboxManager.
"""
from __future__ import annotations

import json
import os
import sys
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
- For RL (SB3) scripts: create envs with `astragym.make(gym_id)` and learners
  with `astrarl.make(learner_id, gym, ...)`. Do NOT call `gym.make` or
  `import gym`. Do not import `gymnasium` unless you use the Gymnasium API
  directly (AstraGym wraps it).
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
import astragym
import astrarl
from envs.snake_env import register as _register_snake
_register_snake()
"""

_TETRIS_SETUP = """\
import sys as _sys
_sys.path.insert(0, "{project_root}")
import astragym
import astrarl
from envs.tetris_env import register as _register_tetris
_register_tetris()
"""

_GAME2048_SETUP = """\
import sys as _sys
_sys.path.insert(0, "{project_root}")
import astragym
import astrarl
from envs.game2048_env import register as _register_2048
_register_2048()
"""

_MINATAR_SETUP = """\
import sys as _sys
_sys.path.insert(0, "{project_root}")
import astragym
import astrarl
from envs.minatar_env import register as _register_minatar
from envs.minatar_space_invaders_env import register as _register_space_invaders
from envs.minatar_asteroids_env import register as _register_asteroids
from envs.minatar_freeway_env import register as _register_freeway
from envs.minatar_seaquest_env import register as _register_seaquest
from envs.minatar_asterix_env import register as _register_asterix
from envs.grid_pacman_env import register as _register_pacman
_register_minatar()
_register_space_invaders()
_register_asteroids()
_register_freeway()
_register_seaquest()
_register_asterix()
_register_pacman()
"""

_GRIDPACMAN_SETUP = """\
import sys as _sys
_sys.path.insert(0, "{project_root}")
import astragym
import astrarl
from envs.grid_pacman_env import register as _register_pacman
_register_pacman()
"""

_AGENT_GYM_SETUP = """\
import sys as _sys
_sys.path.insert(0, "{project_root}")
import astragym
import astrarl
from envs.agent_gym import register as _register_agent_gym
_register_agent_gym()
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
    import numpy as np
    import requests
    import logging
    import astragym
    import astrarl
    from stable_baselines3 import {algorithm}
    from stable_baselines3.common.callbacks import BaseCallback

The script must:
Epoch 9: gym_id={gym_id} learner_id={learner_id} (AstraGym × AstraRL).
1. Create the environment with EXACTLY these lines — copy verbatim:
       _astra_gym = astragym.make("{gym_id}"{env_kwargs_str})
       env = _astra_gym.raw_env
   Do NOT call gym.make or import gymnasium. The planned env is "{env_id}"
   (gym_id "{gym_id}"). Do NOT use "CartPole-v1" or any other env name.
   Do NOT read it from hyperparameters.
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
       _learner = astrarl.make(
           "{learner_id}",
           _astra_gym,
           policy="MlpPolicy",
           **_filtered,
           **(dict(policy_kwargs=_policy_kwargs) if _policy_kwargs else {{}}),
       )
       model = _learner.model

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
       _algo_tag = "{checkpoint_dir}/best_model_algo.txt"
       _ckpt_algo = ""
       try:
           _ckpt_algo = open(_algo_tag).read().strip()
       except Exception:
           _ckpt_algo = ""
       if os.path.exists(_best_ckpt) and (not _ckpt_algo or _ckpt_algo == "{algorithm}"):
           try:
               _warm = {algorithm}.load(_best_ckpt, env=env)
               _warm_sd = _warm.policy.state_dict()
               _model_sd = model.policy.state_dict()
               _shapes_ok = (
                   set(_warm_sd) == set(_model_sd)
                   and all(_warm_sd[k].shape == _model_sd[k].shape for k in _model_sd)
               )
               if _shapes_ok:
                   model.policy.load_state_dict(_warm_sd)
               else:
                   logging.warning(
                       "Warm-start skipped: policy tensor shapes differ from checkpoint "
                       "(%d checkpoint tensors, %d current) — random init for this net_arch",
                       len(_warm_sd), len(_model_sd),
                   )
               del _warm, _warm_sd, _model_sd
           except Exception as _e:
               logging.warning("Warm-start skipped (architecture mismatch or load error): %s", _e)
       elif os.path.exists(_best_ckpt) and _ckpt_algo and _ckpt_algo != "{algorithm}":
           logging.warning(
               "Warm-start skipped: checkpoint algo %s != current %s",
               _ckpt_algo, "{algorithm}",
           )

   This resumes training from the best previously saved weights while keeping the new hyperparameters.
   Load the full policy only when every tensor name and shape matches. If the checkpoint
   architecture differs (e.g. after a net_arch pivot), skip the load and keep the freshly
   initialized weights — do not copy a subset of tensors. `os` and `logging` are already imported.
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

# AgentGym: same SB3 contract, but best_score / early-stop key off task_success
# (mean terminal info["task_success"]), with mean_reward kept as secondary telemetry.
_AGENT_GYM_RL_TEMPLATE = """\
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
    import numpy as np
    import requests
    import logging
    import astragym
    import astrarl
    from stable_baselines3 import {algorithm}
    from stable_baselines3.common.callbacks import BaseCallback

The script must:
Epoch 9: gym_id={gym_id} learner_id={learner_id} (AstraGym × AstraRL).
1. Create the environment with EXACTLY these lines — copy verbatim:
       _astra_gym = astragym.make("{gym_id}"{env_kwargs_str})
       env = _astra_gym.raw_env
   Do NOT call gym.make or import gymnasium. The planned env is "{env_id}"
   (gym_id "{gym_id}"). Do NOT use "CartPole-v1" or any other env name.
   Do NOT read it from hyperparameters.
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
       _learner = astrarl.make(
           "{learner_id}",
           _astra_gym,
           policy="MlpPolicy",
           **_filtered,
           **(dict(policy_kwargs=_policy_kwargs) if _policy_kwargs else {{}}),
       )
       model = _learner.model

3. Immediately after constructing the model, copy this warm-start block EXACTLY — do not modify:

       _best_ckpt = "{checkpoint_dir}/best_model.zip"
       _algo_tag = "{checkpoint_dir}/best_model_algo.txt"
       _ckpt_algo = ""
       try:
           _ckpt_algo = open(_algo_tag).read().strip()
       except Exception:
           _ckpt_algo = ""
       if os.path.exists(_best_ckpt) and (not _ckpt_algo or _ckpt_algo == "{algorithm}"):
           try:
               _warm = {algorithm}.load(_best_ckpt, env=env)
               _warm_sd = _warm.policy.state_dict()
               _model_sd = model.policy.state_dict()
               _shapes_ok = (
                   set(_warm_sd) == set(_model_sd)
                   and all(_warm_sd[k].shape == _model_sd[k].shape for k in _model_sd)
               )
               if _shapes_ok:
                   model.policy.load_state_dict(_warm_sd)
               else:
                   logging.warning(
                       "Warm-start skipped: policy tensor shapes differ from checkpoint "
                       "(%d checkpoint tensors, %d current) — random init for this net_arch",
                       len(_warm_sd), len(_model_sd),
                   )
               del _warm, _warm_sd, _model_sd
           except Exception as _e:
               logging.warning("Warm-start skipped (architecture mismatch or load error): %s", _e)
       elif os.path.exists(_best_ckpt) and _ckpt_algo and _ckpt_algo != "{algorithm}":
           logging.warning(
               "Warm-start skipped: checkpoint algo %s != current %s",
               _ckpt_algo, "{algorithm}",
           )

4. Implement a custom BaseCallback. Copy this ENTIRE class EXACTLY — do not add, remove, or
   modify any line. AgentGym goals use task_success (mean terminal info), NOT mean_reward,
   for best_model / best_score.txt / early stop. mean_reward is still posted as secondary telemetry.

       class CustomCallback(BaseCallback):
           def __init__(self, verbose=0):
               super().__init__(verbose=verbose)
               try:
                   self._best_score = float(open("{checkpoint_dir}/best_score.txt").read().strip())
               except Exception:
                   self._best_score = float("-inf")

           def _eval_task_success(self, n_episodes=8):
               successes = []
               for _ in range(n_episodes):
                   _obs = self.training_env.reset()
                   if isinstance(_obs, tuple):
                       _obs = _obs[0]
                   _done = False
                   _info = {{}}
                   while not _done:
                       _act, _ = self.model.predict(_obs, deterministic=True)
                       _step = self.training_env.step(_act)
                       if len(_step) == 5:
                           _obs, _r, _term, _trunc, _info = _step
                           _done = bool(_term or _trunc)
                       else:
                           _obs, _r, _done, _info = _step
                       if isinstance(_info, (list, tuple)) and _info:
                           _info = _info[0] if isinstance(_info[0], dict) else {{}}
                   successes.append(float((_info or {{}}).get("task_success", 0.0)))
               return float(np.mean(successes)) if successes else 0.0

           def _on_step(self) -> bool:
               if self.n_calls % {telemetry_interval} == 0 and len(self.model.ep_info_buffer) > 0:
                   mean_reward = float(np.mean([ep["r"] for ep in self.model.ep_info_buffer]))
                   task_success = self._eval_task_success()
                   for _name, _val in (("mean_reward", mean_reward), ("task_success", task_success)):
                       try:
                           response = requests.post(
                               "{api_url}/telemetry/missions/{mission_id}/metrics",
                               json={{"mission_id": "{mission_id}", "name": _name,
                                      "value": _val, "step": self.n_calls,
                                      "iteration": {current_iteration}}},
                               timeout=2,
                           )
                           if not response.ok:
                               logging.warning("Telemetry failed: %s", response.status_code)
                       except Exception as exc:
                           logging.warning("Telemetry error: %s", exc)
                   if task_success > self._best_score:
                       self._best_score = task_success
                       self.model.save("{checkpoint_dir}/best_model")
                       with open("{checkpoint_dir}/best_score.txt", "w") as _f:
                           _f.write(str(task_success))
                       with open("{checkpoint_dir}/best_model_algo.txt", "w") as _f:
                           _f.write(self.model.__class__.__name__)
                   if task_success >= {target_reward}:
                       return False
               return True

   The `self.n_calls % {telemetry_interval} == 0` guard is MANDATORY. Never remove it.
   The best_model save block is MANDATORY — it ensures the peak model is preserved.
   The __init__ loading from best_score.txt is MANDATORY — it preserves peak weights across restarts.
5. Call model.learn(total_timesteps={total_timesteps}, callback=callback).
6. After training, save the final model: model.save("{checkpoint_dir}/last_model")
7. Exit cleanly when target task_success is reached."""

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
_astra_gym = astragym.make("gymnasium:{env_id}")  # MANDATORY — before the training loop
env = _astra_gym.raw_env
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
_astra_gym = astragym.make("gymnasium:{env_id}")  # MANDATORY — before the training loop
env = _astra_gym.raw_env
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
_astra_gym = astragym.make("gymnasium:{env_id}")  # MANDATORY — before the training loop
env = _astra_gym.raw_env
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
_astra_gym = astragym.make("gymnasium:{env_id}")  # MANDATORY — before the training loop
env = _astra_gym.raw_env
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

_GAME2048_LOOKAHEAD_DQN_CONTRACT = """\
Generate a complete custom lookahead-DQN training script for Game2048-v0 using PyTorch.
Do NOT use Stable-Baselines3. Implement everything with PyTorch and plain Python.

This trainer uses Game2048Env.get_next_states() lookahead action-selection: evaluate
every legal slide action's resulting board through the Game2048ValueNet network, and
select the action yielding the highest predicted successor state value. It retains
DQN's core stabilization mechanisms: an off-policy replay buffer plus periodic target
network updates for stable TD targets.

{env_setup}
Mission ID: {mission_id}
Environment: {env_id}
Checkpoint directory: {checkpoint_dir}
Telemetry URL: {api_url}/telemetry/missions/{mission_id}/metrics
Target: score >= {target_score}
Hyperparameters: {hyperparameters}

== Mandatory imports and setup (copy exactly) ==
from envs.actor_critic_net import Game2048ValueNet   # MANDATORY — do NOT redefine inline
model = Game2048ValueNet(input_dim=16)
target_model = Game2048ValueNet(input_dim=16)
target_model.load_state_dict(model.state_dict())
target_model.eval()
optimizer = torch.optim.Adam(model.parameters(), lr={lr})

== Training skeleton (follow exactly — do NOT deviate from these API calls) ==
_astra_gym = astragym.make("gymnasium:{env_id}")  # MANDATORY — before the training loop
env = _astra_gym.raw_env
BUFFER = collections.deque(maxlen={replay_buffer_size})
ep_scores, ep_max_tiles = [], []
_eps_path = "{checkpoint_dir}/epsilon.txt"
epsilon = float(open(_eps_path).read().strip()) if os.path.exists(_eps_path) else 1.0
best_score = float("-inf")
total_steps = 0
episode = 0

while total_steps < {total_timesteps}:
    obs, _ = env.reset()
    ep_score = 0.0
    ep_max_tile = 2
    done = False

    while not done:
        next_states = env.unwrapped.get_next_states()  # dict{{action: np.array(16,)}}
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
        BUFFER.append((obs, action, reward, next_obs, float(done)))
        obs = next_obs
        total_steps += 1
        if done:
            ep_score = float(info.get("score", getattr(env.unwrapped, "_score", 0)))
            ep_max_tile = int(info.get("max_tile", getattr(env.unwrapped, "_max_tile", 2)))

        if len(BUFFER) >= {batch_size}:
            batch = random.sample(BUFFER, {batch_size})
            s, _, r, ns, d = zip(*batch)
            s  = torch.tensor(np.array(s),  dtype=torch.float32)
            ns = torch.tensor(np.array(ns), dtype=torch.float32)
            r  = torch.tensor(r,  dtype=torch.float32).unsqueeze(1)
            d  = torch.tensor(d,  dtype=torch.float32).unsqueeze(1)
            with torch.no_grad():
                td_target = r + {gamma} * target_model(ns) * (1 - d)
            loss = nn.MSELoss()(model(s), td_target)
            optimizer.zero_grad(); loss.backward(); optimizer.step()

        if total_steps % {target_update_interval} == 0:
            target_model.load_state_dict(model.state_dict())

    epsilon = max({epsilon_min}, epsilon * {epsilon_decay})
    open(_eps_path, "w").write(str(epsilon))
    ep_scores.append(ep_score)
    ep_max_tiles.append(ep_max_tile)
    episode += 1

    if len(ep_scores) >= {ac_telemetry_interval} and episode % {ac_telemetry_interval} == 0:
        mean_score_50 = float(np.mean(ep_scores[-{ac_telemetry_interval}:]))
        mean_tile_50  = float(np.mean(ep_max_tiles[-{ac_telemetry_interval}:]))
        try:
            requests.post(
                "{api_url}/telemetry/missions/{mission_id}/metrics",
                json={{"step": total_steps, "metric": "score", "val": mean_score_50, "max_tile": mean_tile_50, "episode": episode}},
                timeout=2.0,
            )
        except Exception:
            pass
        if mean_score_50 > best_score:
            best_score = mean_score_50
            torch.save(model, "{checkpoint_dir}/best_model.pth")
            open("{checkpoint_dir}/best_score.txt", "w").write(str(best_score))
            open("{checkpoint_dir}/best_model_algo.txt", "w").write("LookaheadDQN")

== ASTRA integration contract (ALL items MANDATORY) ==
1. Write this file once at startup:
     open("{checkpoint_dir}/trainer_type.txt", "w").write("lookahead_dqn")
2. Warm-start: if "{checkpoint_dir}/best_model.pth" exists, load with weights_only=False
   into BOTH model and target_model:
     if os.path.exists("{checkpoint_dir}/best_model.pth"):
         checkpoint = torch.load("{checkpoint_dir}/best_model.pth", weights_only=False)
         state = checkpoint.state_dict() if not isinstance(checkpoint, dict) else checkpoint
         model.load_state_dict(state)
         target_model.load_state_dict(state)
3. Step telemetry: every episode, post metric "score" with the step count and current episode score.
4. Checkpoint saving on every new best:
     torch.save(model, "{checkpoint_dir}/best_model.pth")
     open("{checkpoint_dir}/best_score.txt", "w").write(str(best_score))
     open("{checkpoint_dir}/best_model_algo.txt", "w").write("LookaheadDQN")
5. After the timestep budget ({total_timesteps} steps): torch.save(model, "{checkpoint_dir}/last_model.pth")

Return ONLY the raw Python script. No markdown fences, no explanation."""

_CANONICAL_SFT_RUNNER = """\
import os
import sys
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)

# Ensure project root is in sys.path
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from backend.trainers.base import TrainerConfig
from backend.trainers.sft_trainer import SFTTrainer

def main() -> None:
    config = TrainerConfig(
        mission_id="{mission_id}",
        model_record_id="{mission_id}",
        data_dir=os.path.dirname(os.path.abspath(__file__)),
        api_base_url="{api_url}",
        hyperparameters={{
            "base_model": "{base_model}",
            "dataset_path": "{dataset_path}",
            "val_split": {val_split},
            "seed": 42,
            "max_seq_length": {max_seq_length},
            "preserve_reasoning": {preserve_reasoning},
            "lora_r": {lora_r},
            "lora_alpha": {lora_alpha},
            "lora_dropout": {lora_dropout},
            "batch_size": {batch_size},
            "learning_rate": {learning_rate},
            "num_train_epochs": {num_epochs},
            "save_steps": {save_steps},
            "eval_steps": {eval_steps},
            "load_in_4bit": True,
        }},
        target_metric={target_metric},
    )
    logger.info("Starting SFT training for mission %s...", config.mission_id)
    trainer = SFTTrainer(config)
    trainer.run()
    logger.info("SFT training completed. Best checkpoint at %s", trainer.best_checkpoint_path)

if __name__ == "__main__":
    main()
"""

_SFT_REMOTE_WRAPPER = """\
import os
import sys

os.chdir("{finetune_dir}")
sys.path.insert(0, "{finetune_dir}")
sys.path.insert(0, "{astra_pkg_root}")
os.makedirs("{checkpoint_dir}", exist_ok=True)

import astratune

learner = astratune.make(
    "mlx_sft",
    finetune_dir="{finetune_dir}",
    use_case="{use_case}",
    python_bin="{python_bin}",
    use_execv=True,
    model="{base_model}",
    data="{dataset_path}",
    save_dir="{checkpoint_dir}",
    num_layers={num_layers},
    lora_rank={lora_rank},
    lora_dropout={lora_dropout},
    lora_scale={lora_scale},
    batch_size={batch_size},
    learning_rate={learning_rate},
    iters={iters},
    val_split={val_split},
    val_batches={val_batches},
    steps_per_eval={steps_per_eval},
    steps_per_report={steps_per_report},
    save_every={save_every},
    max_seq_length={max_seq_length},
    {resume_adapter_kwarg}
)
learner.learn()
"""

# Canned remote AstraTune facade wrappers (no coder LLM).
# Path/chdir + astra_pkg_root before import (pip install -e; mirrors removed).
_DPO_REMOTE_WRAPPER = """\
import os
import sys

os.chdir("{finetune_dir}")
sys.path.insert(0, "{finetune_dir}")
sys.path.insert(0, "{astra_pkg_root}")
os.makedirs(os.path.dirname("{save_pairs_path}"), exist_ok=True)
os.makedirs("{checkpoint_dir}", exist_ok=True)

import astratune

learner = astratune.make(
    "mlx_dpo",
    finetune_dir="{finetune_dir}",
    use_case="{use_case}",
    python_bin="{python_bin}",
    use_execv=True,
    model="{base_model}",
    adapter="{adapter}",
    save_dir="{checkpoint_dir}",
    num_layers={num_layers},
    lora_rank={lora_rank},
    lora_dropout={lora_dropout},
    lora_scale={lora_scale},
    k_collect={k_collect},
    temp={temp},
    max_tokens={max_tokens},
    email_weight={email_weight},
    routing_only={routing_only_bool},
    {load_pairs_kwarg}
    epochs={epochs},
    beta={beta},
    learning_rate={learning_rate},
    max_grad_norm={max_grad_norm},
    steps_per_eval={steps_per_eval},
    eval_max_tokens={eval_max_tokens},
    prompt_template="{prompt_template}",
    save_pairs="{save_pairs_path}",
)
learner.learn()
"""

_DISTILL_REMOTE_WRAPPER = """\
import os
import sys

os.chdir("{finetune_dir}")
sys.path.insert(0, "{finetune_dir}")
sys.path.insert(0, "{astra_pkg_root}")
os.makedirs("{checkpoint_dir}", exist_ok=True)

import astratune

learner = astratune.make(
    "mlx_distill",
    finetune_dir="{finetune_dir}",
    use_case="{use_case}",
    python_bin="{python_bin}",
    use_execv=True,
    model="{base_model}",
    {adapter_kwarg}
    save_dir="{checkpoint_dir}",
    prompt_template="{prompt_template}",
    teacher_model="{teacher_model}",
    ollama_url="{ollama_url}",
    num_layers={num_layers},
    lora_rank={lora_rank},
    lora_scale={lora_scale},
    lora_dropout={lora_dropout},
    iters={iters},
    batch_size={batch_size},
    learning_rate={learning_rate},
    steps_per_eval={steps_per_eval},
    save_every={save_every},
    max_seq_len={max_seq_len},
    eval_max_tokens={eval_max_tokens},
    mask_prompt={mask_prompt_bool},
    routing_only={routing_only_bool},
    {focus_cases_kwarg}
)
learner.learn()
"""

_RFT_REMOTE_WRAPPER = """\
import os
import sys

os.chdir("{finetune_dir}")
sys.path.insert(0, "{finetune_dir}")
sys.path.insert(0, "{astra_pkg_root}")
os.makedirs("{checkpoint_dir}", exist_ok=True)

import astratune

learner = astratune.make(
    "mlx_rft",
    finetune_dir="{finetune_dir}",
    use_case="{use_case}",
    python_bin="{python_bin}",
    use_execv=True,
    model="{base_model}",
    {adapter_kwarg}
    save_dir="{checkpoint_dir}",
    prompt_template="{prompt_template}",
    k_samples={k_samples},
    temp={temp},
    max_tokens={max_tokens},
    num_layers={num_layers},
    lora_rank={lora_rank},
    lora_scale={lora_scale},
    lora_dropout={lora_dropout},
    iters={iters},
    batch_size={batch_size},
    learning_rate={learning_rate},
    steps_per_eval={steps_per_eval},
    save_every={save_every},
    max_seq_len={max_seq_len},
    eval_max_tokens={eval_max_tokens},
    mask_prompt={mask_prompt_bool},
    routing_only={routing_only_bool},
    {focus_cases_kwarg}
    {force_gold_cases_kwarg}
)
learner.learn()
"""

# OPD stays in AstraRL (online / gym-coupled) — canned astrarl.make facade.
_OPD_REMOTE_WRAPPER = """\
import os
import sys

os.chdir("{finetune_dir}")
sys.path.insert(0, "{finetune_dir}")
sys.path.insert(0, "{astra_pkg_root}")
os.makedirs("{checkpoint_dir}", exist_ok=True)

import astrarl

learner = astrarl.make(
    "mlx_opd",
    finetune_dir="{finetune_dir}",
    use_case="{use_case}",
    python_bin="{python_bin}",
    use_execv=True,
    model="{base_model}",
    {adapter_kwarg}
    save_dir="{checkpoint_dir}",
    prompt_template="{prompt_template}",
    teacher_model="{teacher_model}",
    num_generations={num_generations},
    temp={temp},
    max_tokens={max_tokens},
    kl_coef={kl_coef},
    outcome_bonus={outcome_bonus},
    clip_epsilon={clip_epsilon},
    adv_clip={adv_clip},
    reward_schema="{reward_schema}",
    num_layers={num_layers},
    lora_rank={lora_rank},
    lora_scale={lora_scale},
    lora_dropout={lora_dropout},
    iters={iters},
    batch_size={batch_size},
    learning_rate={learning_rate},
    steps_per_eval={steps_per_eval},
    save_every={save_every},
    max_seq_len={max_seq_len},
    eval_max_tokens={eval_max_tokens},
    mask_prompt={mask_prompt_bool},
    routing_only={routing_only_bool},
    {focus_cases_kwarg}
)
learner.learn()
"""

# Phase 6a — local sentence-transformers train (NOT Mac Mini MLX).
# Runs on the Astra host via SubprocessSandbox; product_root + .venv required.
_EMBED_LOCAL_WRAPPER = """\
import os

os.chdir("{product_root}")
os.makedirs("{checkpoint_dir}", exist_ok=True)

argv = [
    "{python_bin}", "{train_script_abs}",
    "--data", "{data_dir}",
    "--model", "{base_model}",
    "--out", "{checkpoint_dir}",
    "--epochs", "{epochs}",
    "--batch-size", "{batch_size}",
    "--lr", "{learning_rate}",
    "--warmup-ratio", "{warmup_ratio}",
]
os.execv("{python_bin}", argv)
"""

# HSTU — from-scratch PyTorch trainer in-repo (LSTM-style). NOT Mini MLX,
# NOT Meta generative-recommenders. MovieLens-1M auto-downloads on first run.
_HSTU_LOCAL_WRAPPER = """\
import os

os.makedirs("{checkpoint_dir}", exist_ok=True)

argv = [
    "{python_bin}", "-m", "backend.trainers.hstu_train",
    "--data-dir", "{data_dir}",
    "--dataset-id", "{dataset_id}",
    "--dataset-rev", "{dataset_rev}",
    "--out", "{checkpoint_dir}",
    "--epochs", "{epochs}",
    "--batch-size", "{batch_size}",
    "--lr", "{learning_rate}",
    "--d-model", "{d_model}",
    "--n-layers", "{n_layers}",
    "--n-heads", "{n_heads}",
    "--max-len", "{max_len}",
    "--dropout", "{dropout}",
    "--weight-decay", "{weight_decay}",
    "--seed", "{seed}",
]
os.execv("{python_bin}", argv)
"""

_SFT_TEMPLATE = """\
Generate a complete SFT (QLoRA) fine-tuning script using Astra's SFTTrainer.
Use Astra's SFTTrainer (from backend.trainers.sft_trainer import SFTTrainer) to orchestrate training with strict held-out train_dataset and eval_dataset splitting (fixed seed 42), <think>...</think> reasoning trace preservation, and comprehensive telemetry (train_loss, eval_loss, perplexity).
DO NOT instantiate raw AutoModelForCausalLM or raw transformers.Trainer directly.

Mission ID: {mission_id}
Base model: {base_model}
Dataset path: {dataset_path}
Validation split: {val_split} (held-out split ratio, seed=42)
Max sequence length: {max_seq_length}
Preserve reasoning traces: {preserve_reasoning} (keep <think>...</think> tokens)
LoRA config: r={lora_r}, alpha={lora_alpha}, dropout={lora_dropout}
Training args: batch={batch_size}, lr={learning_rate}, epochs={num_epochs}
save_strategy: steps, save_steps: {save_steps}, eval_steps: {eval_steps}
Checkpoint directory: {checkpoint_dir}
Telemetry URL: {api_url}/telemetry/missions/{mission_id}/metrics

The script must:
1. Ensure project root is in sys.path before importing backend modules.
2. Configure TrainerConfig with hyperparameters and target_metric.
3. Instantiate and run Astra's SFTTrainer: trainer = SFTTrainer(config); trainer.run().
4. Report train_loss, eval_loss, and perplexity to {api_url}/telemetry/missions/{mission_id}/metrics via SFTTrainer.
5. Save the best checkpoint adapter to {checkpoint_dir}.
6. Exit cleanly with code 0 on success, code 1 on error with full traceback.

Emit this complete runnable Python script:

""" + _CANONICAL_SFT_RUNNER

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

# Prompt retained for healers / docs; live codegen uses _DPO_REMOTE_WRAPPER.
_DPO_TEMPLATE = """\
Generate a script that runs DPO via AstraTune MLXLearnerFacade — do NOT reimplement
pair collection / preference loss and do NOT hand-build a long os.execv argv list.
The facade maps kwargs to ``python -m astratune.cli dpo`` and uses os.execv so the
sandbox-tracked pid stays identical to the trainer. This script does NOT report
telemetry; Astra tails "Pass rate" lines remotely. No network calls (no requests).

Mission ID: {mission_id}
Finetune dir (remote host): {finetune_dir}
Python interpreter: {python_bin}
Astra pkg root: {astra_pkg_root}
Base model: {base_model}
Warm-start / reference adapter: {adapter}
Prompt template: {prompt_template}
LoRA: rank={lora_rank}, scale={lora_scale}, dropout={lora_dropout}, layers={num_layers}
Collection: K={k_collect}, temp={temp}, max_tokens={max_tokens}, email_weight={email_weight}
Training: epochs={epochs}, beta={beta}, lr={learning_rate}, max_grad_norm={max_grad_norm}
steps_per_eval={steps_per_eval}, eval_max_tokens={eval_max_tokens}
Adapter output: {checkpoint_dir}
Save collected pairs to: {save_pairs_path}
{load_pairs_kwarg}

Emit EXACTLY this script shape (chdir/path triad before import):

       import os, sys
       os.chdir("{finetune_dir}")
       sys.path.insert(0, "{finetune_dir}")
       sys.path.insert(0, "{astra_pkg_root}")
       os.makedirs(os.path.dirname("{save_pairs_path}"), exist_ok=True)
       os.makedirs("{checkpoint_dir}", exist_ok=True)
       import astratune
       learner = astratune.make(
           "mlx_dpo",
           finetune_dir="{finetune_dir}",
           use_case="{use_case}",
           python_bin="{python_bin}",
           use_execv=True,
           model="{base_model}",
           adapter="{adapter}",
           save_dir="{checkpoint_dir}",
           num_layers={num_layers},
           lora_rank={lora_rank},
           lora_dropout={lora_dropout},
           lora_scale={lora_scale},
           k_collect={k_collect},
           temp={temp},
           max_tokens={max_tokens},
           email_weight={email_weight},
           routing_only={routing_only_bool},
           {load_pairs_kwarg}
           epochs={epochs},
           beta={beta},
           learning_rate={learning_rate},
           max_grad_norm={max_grad_norm},
           steps_per_eval={steps_per_eval},
           eval_max_tokens={eval_max_tokens},
           prompt_template="{prompt_template}",
           save_pairs="{save_pairs_path}",
       )
       learner.learn()

   learn() with use_execv=True never returns (process image becomes astratune.cli dpo)."""

# Canned remote GRPO wrapper (no coder LLM). Path/chdir before import astrarl so
# Mini works with pip-installed astrarl (astra_pkg_root) even if launcher
# python is not finetune-env. Do NOT call astragym.make — MLXLearnerFacade needs no gym.
_GRPO_REMOTE_WRAPPER = """\
import os
import sys

os.chdir("{finetune_dir}")
sys.path.insert(0, "{finetune_dir}")
sys.path.insert(0, "{astra_pkg_root}")

import astrarl

learner = astrarl.make(
    "{learner_id}",
    finetune_dir="{finetune_dir}",
    use_case="{use_case}",
    python_bin="{python_bin}",
    use_execv=True,
    model="{base_model}",
    adapter="{adapter}",
    save_dir="{checkpoint_dir}",
    num_layers={num_layers},
    lora_rank={lora_rank},
    lora_dropout={lora_dropout},
    lora_scale={lora_scale},
    iters={iters},
    num_generations={num_generations},
    temp={temp},
    max_tokens={max_tokens},
    eval_max_tokens={eval_max_tokens},
    reward_schema="{reward_schema}",
    email_weight={email_weight},
    focus_weight={focus_weight},
    max_zero_steps={max_zero_steps},
    max_one_steps={max_one_steps},
    learning_rate={learning_rate},
    clip_epsilon={clip_epsilon},
    max_grad_norm={max_grad_norm},
    steps_per_report={steps_per_report},
    steps_per_eval={steps_per_eval},
    save_every={save_every},
    routing_only={routing_only_bool},
    prompt_template="{prompt_template}",
    {facade_dapo_kwargs}
)
learner.learn()
"""

# Prompt retained for healers / docs; live codegen uses _GRPO_REMOTE_WRAPPER.
_GRPO_TEMPLATE = """\
Generate a script that runs GRPO via AstraRL MLXLearnerFacade — do NOT reimplement
rollout/reward/PPO math and do NOT hand-build a long os.execv argv list for every
flag. The facade maps kwargs to ``python -m astrarl.cli grpo`` and uses os.execv so the
sandbox-tracked pid stays identical to the trainer. This script does NOT report
telemetry; Astra tails "Pass rate" lines remotely. No network calls (no requests).

Mission ID: {mission_id}
Finetune dir (remote host): {finetune_dir}
Python interpreter: {python_bin}
learner_id: {learner_id}
gym_id: {gym_id}
Base model: {base_model}
Warm-start adapter: {adapter}
Prompt template: {prompt_template}
LoRA: rank={lora_rank}, scale={lora_scale}, dropout={lora_dropout}, layers={num_layers}
GRPO: iters={iters}, num_generations={num_generations}, temp={temp}, max_tokens={max_tokens}
eval_max_tokens={eval_max_tokens}, reward_schema={reward_schema}
email_weight={email_weight}, focus_weight={focus_weight}
max_zero_steps={max_zero_steps}, max_one_steps={max_one_steps}
Optimizer: lr={learning_rate}, clip_epsilon={clip_epsilon}, max_grad_norm={max_grad_norm}
DAPO-lite / rollout (optional): clip_epsilon_low={clip_epsilon_low},
clip_epsilon_high={clip_epsilon_high}, dynamic_sample_retries={dynamic_sample_retries},
rollout_backend={rollout_backend}, rollout_base_url={rollout_base_url}
Logging: steps_per_report={steps_per_report}, steps_per_eval={steps_per_eval}, save_every={save_every}
Adapter output: {checkpoint_dir}
routing_only: {routing_only}

Emit EXACTLY this script shape (no astragym.make; chdir/path before import):

       import os, sys
       os.chdir("{finetune_dir}")
       sys.path.insert(0, "{finetune_dir}")
       sys.path.insert(0, "{astra_pkg_root}")
       import astrarl
       learner = astrarl.make(
           "{learner_id}",
           finetune_dir="{finetune_dir}",
           use_case="{use_case}",
           python_bin="{python_bin}",
           use_execv=True,
           model="{base_model}",
           adapter="{adapter}",
           save_dir="{checkpoint_dir}",
           num_layers={num_layers},
           lora_rank={lora_rank},
           lora_dropout={lora_dropout},
           lora_scale={lora_scale},
           iters={iters},
           num_generations={num_generations},
           temp={temp},
           max_tokens={max_tokens},
           eval_max_tokens={eval_max_tokens},
           reward_schema="{reward_schema}",
           email_weight={email_weight},
           focus_weight={focus_weight},
           max_zero_steps={max_zero_steps},
           max_one_steps={max_one_steps},
           learning_rate={learning_rate},
           clip_epsilon={clip_epsilon},
           max_grad_norm={max_grad_norm},
           steps_per_report={steps_per_report},
           steps_per_eval={steps_per_eval},
           save_every={save_every},
           routing_only={routing_only_bool},
           prompt_template="{prompt_template}",
           {facade_dapo_kwargs}
       )
       learner.learn()

   learn() with use_execv=True never returns (process image becomes astrarl.cli grpo)."""

# Canned remote Agent GRPO wrapper (no coder LLM). Same triad + astrarl.make as GRPO.
_AGRPO_REMOTE_WRAPPER = """\
import os
import sys

os.chdir("{finetune_dir}")
sys.path.insert(0, "{finetune_dir}")
sys.path.insert(0, "{astra_pkg_root}")

import astrarl

learner = astrarl.make(
    "{learner_id}",
    finetune_dir="{finetune_dir}",
    use_case="{use_case}",
    python_bin="{python_bin}",
    use_execv=True,
    model="{base_model}",
    {adapter_kwarg}
    save_dir="{checkpoint_dir}",
    num_layers={num_layers},
    lora_rank={lora_rank},
    lora_dropout={lora_dropout},
    lora_scale={lora_scale},
    iters={iters},
    num_generations={num_generations},
    temp={temp},
    max_tokens={max_tokens},
    max_turns={max_turns},
    tool_failure_prob={tool_failure_prob},
    reward_mode="{reward_mode}",
    {e2e_cases_kwarg}
    learning_rate={learning_rate},
    clip_epsilon={clip_epsilon},
    max_grad_norm={max_grad_norm},
    steps_per_report={steps_per_report},
    steps_per_eval={steps_per_eval},
    save_every={save_every},
)
learner.learn()
"""

# Prompt retained for healers / docs; live codegen uses _AGRPO_REMOTE_WRAPPER.
_AGRPO_TEMPLATE = """\
Generate a script that runs Agent GRPO (multi-turn tool trajectory RL) via
AstraRL MLXLearnerFacade — do NOT reimplement the rollout loop / reward / GRPO
update and do NOT hand-build a long os.execv argv list. The facade maps kwargs
to ``python -m astrarl.cli agrpo`` (os.execv, same pid). No network calls; Astra
tails "Pass rate" remotely.

Epoch 9+: AstraGym gym_id={gym_id} × AstraRL learner_id={learner_id}
(reward_mode={reward_mode} selects agent_gym_text vs e2e_stubs).

Mission ID: {mission_id}
Finetune dir: {finetune_dir}
Python: {python_bin}
Astra pkg root: {astra_pkg_root}
Base model: {base_model}
Warm-start / cold: {adapter_kwarg}
LoRA: rank={lora_rank}, scale={lora_scale}, dropout={lora_dropout}, layers={num_layers}
Agent GRPO: iters={iters}, num_generations={num_generations}, temp={temp},
max_tokens={max_tokens}, max_turns={max_turns}, reward_mode={reward_mode},
tool_failure_prob={tool_failure_prob}, e2e_cases={e2e_cases}
Optimizer: lr={learning_rate}, clip_epsilon={clip_epsilon}, max_grad_norm={max_grad_norm}
Logging: steps_per_report={steps_per_report}, steps_per_eval={steps_per_eval}, save_every={save_every}
Adapter output: {checkpoint_dir}

Emit EXACTLY this script shape (no astragym.make; chdir/path before import):

       import os, sys
       os.chdir("{finetune_dir}")
       sys.path.insert(0, "{finetune_dir}")
       sys.path.insert(0, "{astra_pkg_root}")
       import astrarl
       learner = astrarl.make(
           "{learner_id}",
           finetune_dir="{finetune_dir}",
           use_case="{use_case}",
           python_bin="{python_bin}",
           use_execv=True,
           model="{base_model}",
           {adapter_kwarg}
           save_dir="{checkpoint_dir}",
           num_layers={num_layers},
           lora_rank={lora_rank},
           lora_dropout={lora_dropout},
           lora_scale={lora_scale},
           iters={iters},
           num_generations={num_generations},
           temp={temp},
           max_tokens={max_tokens},
           max_turns={max_turns},
           tool_failure_prob={tool_failure_prob},
           reward_mode="{reward_mode}",
           {e2e_cases_kwarg}
           learning_rate={learning_rate},
           clip_epsilon={clip_epsilon},
           max_grad_norm={max_grad_norm},
           steps_per_report={steps_per_report},
           steps_per_eval={steps_per_eval},
           save_every={save_every},
       )
       learner.learn()

   learn() with use_execv=True never returns (process image becomes astrarl.cli agrpo)."""

# Prompt retained for healers / docs; live codegen uses _DISTILL_REMOTE_WRAPPER.
_DISTILL_TEMPLATE = """\
Generate a script that runs distillation via AstraTune MLXLearnerFacade — do NOT
reimplement teacher generation / SFT and do NOT hand-build a long os.execv argv.
The facade maps kwargs to astratune.cli distill (os.execv, same pid).
No network calls; Astra tails "Pass rate" remotely.

Mission ID: {mission_id}
Finetune dir: {finetune_dir}
Python: {python_bin}
Astra pkg root: {astra_pkg_root}
Base model: {base_model}
Warm-start: {adapter_kwarg}
Prompt: {prompt_template}
Teacher: {teacher_model} @ {ollama_url}
LoRA: rank={lora_rank}, scale={lora_scale}, dropout={lora_dropout}, layers={num_layers}
iters={iters}, batch={batch_size}, lr={learning_rate}
Adapter output: {checkpoint_dir}

Emit EXACTLY:

       import os, sys
       os.chdir("{finetune_dir}")
       sys.path.insert(0, "{finetune_dir}")
       sys.path.insert(0, "{astra_pkg_root}")
       os.makedirs("{checkpoint_dir}", exist_ok=True)
       import astratune
       learner = astratune.make(
           "mlx_distill",
           finetune_dir="{finetune_dir}",
           use_case="{use_case}",
           python_bin="{python_bin}",
           use_execv=True,
           model="{base_model}",
           {adapter_kwarg}
           save_dir="{checkpoint_dir}",
           prompt_template="{prompt_template}",
           teacher_model="{teacher_model}",
           ollama_url="{ollama_url}",
           num_layers={num_layers},
           lora_rank={lora_rank},
           lora_scale={lora_scale},
           lora_dropout={lora_dropout},
           iters={iters},
           batch_size={batch_size},
           learning_rate={learning_rate},
           steps_per_eval={steps_per_eval},
           save_every={save_every},
           max_seq_len={max_seq_len},
           eval_max_tokens={eval_max_tokens},
           mask_prompt={mask_prompt_bool},
           routing_only={routing_only_bool},
           {focus_cases_kwarg}
       )
       learner.learn()
"""


# Prompt retained for healers / docs; live codegen uses _RFT_REMOTE_WRAPPER.
_RFT_TEMPLATE = """\
Generate a script that runs RFT via AstraTune MLXLearnerFacade — do NOT
reimplement sampling/filter/SFT and do NOT hand-build a long os.execv argv.
Facade → ``python -m astratune.cli rft`` via os.execv (same pid). No network calls.

Mission ID: {mission_id}
Finetune dir: {finetune_dir}
Python: {python_bin}
Astra pkg root: {astra_pkg_root}
Base model: {base_model}
Warm-start: {adapter_kwarg}
Prompt: {prompt_template}
k_samples={k_samples}, temp={temp}, max_tokens={max_tokens}
LoRA: rank={lora_rank}, scale={lora_scale}, dropout={lora_dropout}, layers={num_layers}
Adapter output: {checkpoint_dir}

Emit EXACTLY:

       import os, sys
       os.chdir("{finetune_dir}")
       sys.path.insert(0, "{finetune_dir}")
       sys.path.insert(0, "{astra_pkg_root}")
       os.makedirs("{checkpoint_dir}", exist_ok=True)
       import astratune
       learner = astratune.make(
           "mlx_rft",
           finetune_dir="{finetune_dir}",
           use_case="{use_case}",
           python_bin="{python_bin}",
           use_execv=True,
           model="{base_model}",
           {adapter_kwarg}
           save_dir="{checkpoint_dir}",
           prompt_template="{prompt_template}",
           k_samples={k_samples},
           temp={temp},
           max_tokens={max_tokens},
           num_layers={num_layers},
           lora_rank={lora_rank},
           lora_scale={lora_scale},
           lora_dropout={lora_dropout},
           iters={iters},
           batch_size={batch_size},
           learning_rate={learning_rate},
           steps_per_eval={steps_per_eval},
           save_every={save_every},
           max_seq_len={max_seq_len},
           eval_max_tokens={eval_max_tokens},
           mask_prompt={mask_prompt_bool},
           routing_only={routing_only_bool},
           {focus_cases_kwarg}
           {force_gold_cases_kwarg}
       )
       learner.learn()
"""


# Prompt retained for healers / docs; live codegen uses _OPD_REMOTE_WRAPPER.
_OPD_TEMPLATE = """\
Generate a script that runs OPD via AstraRL MLXLearnerFacade — do NOT
reimplement student/teacher KL and do NOT hand-build a long os.execv argv.
Facade → astrarl.cli opd via os.execv (same pid). No network calls.

Mission ID: {mission_id}
Finetune dir: {finetune_dir}
Python: {python_bin}
Astra pkg root: {astra_pkg_root}
Base model: {base_model}
Warm-start: {adapter_kwarg}
Prompt: {prompt_template}
Teacher: {teacher_model}
Adapter output: {checkpoint_dir}

Emit EXACTLY:

       import os, sys
       os.chdir("{finetune_dir}")
       sys.path.insert(0, "{finetune_dir}")
       sys.path.insert(0, "{astra_pkg_root}")
       os.makedirs("{checkpoint_dir}", exist_ok=True)
       import astrarl
       learner = astrarl.make(
           "mlx_opd",
           finetune_dir="{finetune_dir}",
           use_case="{use_case}",
           python_bin="{python_bin}",
           use_execv=True,
           model="{base_model}",
           {adapter_kwarg}
           save_dir="{checkpoint_dir}",
           prompt_template="{prompt_template}",
           teacher_model="{teacher_model}",
           num_generations={num_generations},
           temp={temp},
           max_tokens={max_tokens},
           kl_coef={kl_coef},
           outcome_bonus={outcome_bonus},
           clip_epsilon={clip_epsilon},
           adv_clip={adv_clip},
           reward_schema="{reward_schema}",
           num_layers={num_layers},
           lora_rank={lora_rank},
           lora_scale={lora_scale},
           lora_dropout={lora_dropout},
           iters={iters},
           batch_size={batch_size},
           learning_rate={learning_rate},
           steps_per_eval={steps_per_eval},
           save_every={save_every},
           max_seq_len={max_seq_len},
           eval_max_tokens={eval_max_tokens},
           mask_prompt={mask_prompt_bool},
           routing_only={routing_only_bool},
           {focus_cases_kwarg}
       )
       learner.learn()
"""


_RLCD_REMOTE_WRAPPER = """\
import os
import sys

os.chdir("{finetune_dir}")
sys.path.insert(0, "{finetune_dir}")
sys.path.insert(0, "{astra_pkg_root}")
os.makedirs("{checkpoint_dir}", exist_ok=True)

import astratune

learner = astratune.make(
    "mlx_rlcd",
    finetune_dir="{finetune_dir}",
    use_case="{use_case}",
    python_bin="{python_bin}",
    use_execv=True,
    model="{base_model}",
    {adapter_kwarg}
    save_dir="{checkpoint_dir}",
    prompt_template="{prompt_template}",
    stage="{stage}",
    score="{score}",
    num_generations={num_generations},
    temp={temp},
    max_tokens={max_tokens},
    eval_max_tokens={eval_max_tokens},
    sf_coef={sf_coef},
    kl_beta={kl_beta},
    num_layers={num_layers},
    lora_rank={lora_rank},
    lora_scale={lora_scale},
    lora_dropout={lora_dropout},
    iters={iters},
    learning_rate={learning_rate},
    max_grad_norm={max_grad_norm},
    steps_per_eval={steps_per_eval},
    steps_per_report={steps_per_report},
    save_every={save_every},
    max_seq_len={max_seq_len},
    routing_only={routing_only_bool},
    empty_rationale={empty_rationale_bool},
    focus_weight={focus_weight},
    {cases_kwarg}
    {focus_cases_kwarg}
    {eval_max_cases_kwarg}
)
learner.learn()
"""


def _format_rlcd_wrapper(hp: dict, checkpoint_dir: str) -> str:
    """Canned astratune.make facade. The coder model does not write the RLCD loop."""
    cold = bool(hp.get("no_adapter", False)) or not hp.get("adapter")
    adapter_kwarg = (
        "no_adapter=True," if cold
        else 'adapter="%s",' % hp.get("adapter", "")
    )
    cases = str(hp.get("cases") or "").strip()
    # Foundation / specialize recipes set cases; default routing-only when unset.
    routing_only = bool(hp["routing_only"]) if "routing_only" in hp else (not cases)
    empty_rationale = bool(hp["empty_rationale"]) if "empty_rationale" in hp else True
    focus = _resolve_focus_cases(hp)
    eval_max = int(hp.get("eval_max_cases") or 0)
    return _RLCD_REMOTE_WRAPPER.format(
        finetune_dir=hp.get("finetune_dir", "/Users/kewang/finetune"),
        use_case=_resolve_use_case(hp),
        python_bin=hp.get("python_bin", "/Users/kewang/finetune-env/bin/python"),
        astra_pkg_root=_resolve_astra_pkg_root(hp),
        checkpoint_dir=checkpoint_dir,
        base_model=hp.get("base_model", ""),
        adapter_kwarg=adapter_kwarg,
        prompt_template=hp.get("prompt_template", "backend/prompts/conductor_min.md"),
        stage=hp.get("stage", "both"),
        score=hp.get("score", "brier"),
        num_generations=hp.get("num_generations", 4),
        temp=hp.get("temp", 1.0),
        max_tokens=hp.get("max_tokens", 256),
        eval_max_tokens=hp.get("eval_max_tokens", 256),
        sf_coef=hp.get("sf_coef", 0.3),
        kl_beta=hp.get("kl_beta", 0.04),
        num_layers=hp.get("num_layers", 8),
        lora_rank=hp.get("lora_rank", 16),
        lora_scale=hp.get("lora_scale", 4.0),
        lora_dropout=hp.get("lora_dropout", 0.0),
        iters=hp.get("iters", 100),
        learning_rate=hp.get("learning_rate", 2e-6),
        max_grad_norm=hp.get("max_grad_norm", 1.0),
        steps_per_eval=hp.get("steps_per_eval", 25),
        steps_per_report=hp.get("steps_per_report", 1),
        save_every=hp.get("save_every", 25),
        max_seq_len=hp.get("max_seq_len", 4096),
        routing_only_bool="True" if routing_only else "False",
        empty_rationale_bool="True" if empty_rationale else "False",
        focus_weight=int(hp.get("focus_weight") or 10),
        cases_kwarg=(f'cases="{cases.replace(chr(34), "")}",' if cases else ""),
        focus_cases_kwarg=(f'focus_cases="{focus}",' if focus else ""),
        eval_max_cases_kwarg=(f"eval_max_cases={eval_max}," if eval_max > 0 else ""),
    )

_STAR_REMOTE_WRAPPER = """\
import os
import sys

os.chdir("{finetune_dir}")
sys.path.insert(0, "{finetune_dir}")
sys.path.insert(0, "{astra_pkg_root}")
os.makedirs("{checkpoint_dir}", exist_ok=True)

import astratune

learner = astratune.make(
    "mlx_star",
    finetune_dir="{finetune_dir}",
    use_case="{use_case}",
    python_bin="{python_bin}",
    use_execv=True,
    model="{base_model}",
    {adapter_kwarg}
    save_dir="{checkpoint_dir}",
    prompt_template="{prompt_template}",
    k_samples={k_samples},
    temp={temp},
    rationalize_temp={rationalize_temp},
    max_tokens={max_tokens},
    eval_max_tokens={eval_max_tokens},
    num_layers={num_layers},
    lora_rank={lora_rank},
    lora_scale={lora_scale},
    lora_dropout={lora_dropout},
    iters={iters},
    batch_size={batch_size},
    learning_rate={learning_rate},
    steps_per_eval={steps_per_eval},
    save_every={save_every},
    max_seq_len={max_seq_len},
    val_split={val_split},
    reward_schema="{reward_schema}",
    routing_only={routing_only_bool},
)
learner.learn()
"""

# Prompt retained for healers / docs; live codegen uses _STAR_REMOTE_WRAPPER.
_STAR_TEMPLATE = """\
Generate a script that runs STaR via AstraTune MLXLearnerFacade — do NOT
reimplement rationalization/SFT and do NOT hand-build a long os.execv argv.
Facade → astratune.cli star via os.execv (same pid). No network calls.

Mission ID: {mission_id}
Finetune dir: {finetune_dir}
Python: {python_bin}
Astra pkg root: {astra_pkg_root}
Base model: {base_model}
Warm-start: {adapter_kwarg}
Prompt: {prompt_template}
Adapter output: {checkpoint_dir}

Emit EXACTLY:

       import os, sys
       os.chdir("{finetune_dir}")
       sys.path.insert(0, "{finetune_dir}")
       sys.path.insert(0, "{astra_pkg_root}")
       os.makedirs("{checkpoint_dir}", exist_ok=True)
       import astratune
       learner = astratune.make(
           "mlx_star",
           finetune_dir="{finetune_dir}",
           use_case="{use_case}",
           python_bin="{python_bin}",
           use_execv=True,
           model="{base_model}",
           {adapter_kwarg}
           save_dir="{checkpoint_dir}",
           prompt_template="{prompt_template}",
           k_samples={k_samples},
           temp={temp},
           rationalize_temp={rationalize_temp},
           max_tokens={max_tokens},
           eval_max_tokens={eval_max_tokens},
           num_layers={num_layers},
           lora_rank={lora_rank},
           lora_scale={lora_scale},
           lora_dropout={lora_dropout},
           iters={iters},
           batch_size={batch_size},
           learning_rate={learning_rate},
           steps_per_eval={steps_per_eval},
           save_every={save_every},
           max_seq_len={max_seq_len},
           val_split={val_split},
           reward_schema="{reward_schema}",
           routing_only={routing_only_bool},
       )
       learner.learn()
"""


# Canned prompt-variant wrapper (shell frozen; LLM fills EXTRA_RULES only).
_PROMPT_REMOTE_WRAPPER = """\
import os
import sys

os.chdir("{finetune_dir}")
sys.path.insert(0, "{finetune_dir}")
sys.path.insert(0, "{astra_pkg_root}")

EXTRA_RULES = {extra_rules_repr}

with open("{base_prompt}") as f:
    base = f.read()
os.makedirs(os.path.dirname("{variant_path}"), exist_ok=True)
with open("{variant_path}", "w") as f:
    f.write(base + "\\n\\n" + EXTRA_RULES)

os.environ["ENSEMBLE_FROZEN_NOW"] = "{frozen_now}"
os.execv(
    "{python_bin}",
    [
        "{python_bin}", "{finetune_dir}/bare_eval.py",
        "--model", "{base_model}",
        "--no-adapter",
        "--prompt-template", "{variant_path}",
        "--max-tokens", "{eval_max_tokens}",
    ],
)
"""

# LLM prompt: emit only the appended EXTRA_RULES text (no Python).
_PROMPT_RULES_PROMPT = """\
Propose additional product routing rules that raise model-routed accuracy.
Emit ONLY the appended rules block as plain text (markdown / examples OK).
Do NOT emit Python, imports, os.execv, or a full script.

You are NOT rewriting the base prompt — you cannot see it and must not try to
reproduce it. It is ~5,100 tokens of existing rules; any attempt to restate it
will silently drop content and score worse for a reason no one will be able to
see. Astra reads the base file at runtime and appends your block (EXTRA_RULES).

Mission ID: {mission_id}
BASE PROMPT path (read at runtime, never rewritten): {base_prompt}
Variant output path: {variant_path}

Target genuine routing ambiguities between canonical skills: "Market Analysis",
"Business Reporting", "Data Services", "System Admin", "Programming",
"Travel Planning", "General Chat", "Calendar", and dynamic MCP tools
("mcp:<Server>:<tool>").

CRITICAL GUIDANCE:
- NEVER write abstract meta-rules (such as "be concise", "avoid repetition", or
  "prioritize clarity"). Small language models (Gemma 3 4B) ignore abstract adages.
- Prefer EXPLICIT NEGATIVE CONSTRAINTS and CONCRETE JSON FEW-SHOT EXAMPLES:
    PLANNING EXAMPLE (<Label>):
    - Query: "<query>"
      - Task 1: {{"title": "<title>", "skill": "<Canonical Skill>", "agent_id": "<agent_id>", "arguments": {{...}}, "depends_on": []}}
- A rule that repeats the base prompt is wasted; a contradicting rule makes the
  model worse. Prefer few strong, specific rules over many weak ones.

Output the EXTRA_RULES body only — nothing else.
"""

# Healer / docs: canned wrapper shape + same rules guidance (EXTRA_RULES filled live).
_PROMPT_TEMPLATE = """\
Generate EXTRA_RULES text for a canned prompt-variant wrapper that appends your
block to the base conductor prompt and os.execv's bare_eval.py. Do NOT
reimplement the eval/oracle. Live codegen freezes the shell; you supply rules.

Mission ID: {mission_id}
Finetune dir: {finetune_dir}
Python: {python_bin}
Astra pkg root: {astra_pkg_root}
Base model: {base_model}
BASE PROMPT (read at runtime, never rewritten): {base_prompt}
Variant output path: {variant_path}
Frozen clock instant: {frozen_now}
eval_max_tokens: {eval_max_tokens}

You are NOT rewriting the base prompt — you cannot see it and must not try to reproduce it.
It is ~5,100 tokens of existing rules. Read-then-append ONLY —
never open the base prompt for writing; it is the product's committed file.

Target skills: "Market Analysis", "Business Reporting", "Data Services",
"System Admin", "Programming", "Travel Planning", "General Chat", "Calendar",
and dynamic MCP tools ("mcp:<Server>:<tool>").

CRITICAL GUIDANCE:
- NEVER write abstract meta-rules (such as "be concise", "avoid repetition", or
  "prioritize clarity").
- Prefer EXPLICIT NEGATIVE CONSTRAINTS and CONCRETE JSON FEW-SHOT EXAMPLES.
- Prefer few strong, specific rules over many weak ones.

The live train.py shape (EXTRA_RULES is your block):

       import os, sys
       os.chdir("{finetune_dir}")
       sys.path.insert(0, "{finetune_dir}")
       sys.path.insert(0, "{astra_pkg_root}")
       EXTRA_RULES = \"\"\"…your rules…\"\"\"
       with open("{base_prompt}") as f:
           base = f.read()
       os.makedirs(os.path.dirname("{variant_path}"), exist_ok=True)
       with open("{variant_path}", "w") as f:
           f.write(base + "\\n\\n" + EXTRA_RULES)
       os.environ["ENSEMBLE_FROZEN_NOW"] = "{frozen_now}"
       os.execv(
           "{python_bin}",
           [
               "{python_bin}", "{finetune_dir}/bare_eval.py",
               "--model", "{base_model}",
               "--no-adapter",
               "--prompt-template", "{variant_path}",
               "--max-tokens", "{eval_max_tokens}",
           ],
       )
"""

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


# Canonical SB3 / custom-trainer labels. The Lead Agent often emits "SB3 PPO"
# which is the same trainer as "PPO"; treating them as distinct made level-2
# "algorithm switches" no-ops (Seaquest missions bef66c1e / 7c567090).
_ALGO_CANONICAL: dict[str, str] = {
    "PPO": "PPO",
    "SB3 PPO": "PPO",
    "STABLE BASELINES3 PPO": "PPO",
    "DQN": "DQN",
    "SB3 DQN": "DQN",
    "A2C": "A2C",
    "SB3 A2C": "A2C",
    "SAC": "SAC",
    "TD3": "TD3",
    "ACTOR CRITIC": "Actor-Critic",
    "LOOKAHEAD DQN": "LOOKAHEAD_DQN",
    "LOOKAHEAD PPO": "LOOKAHEAD_PPO",
    "LOOKAHEAD A2C": "LOOKAHEAD_A2C",
}


def canonicalize_algorithm(name: str) -> str:
    """Map LLM/UI algorithm labels to a single trainer name.

    Empty/unknown strings are returned stripped (or empty) so callers can
    distinguish "not set" from a real algo. Known aliases collapse to the
    SB3 class name (or the custom-trainer label).
    """
    if not name or not str(name).strip():
        return ""
    raw = str(name).strip()
    key = " ".join(raw.upper().replace("-", " ").replace("_", " ").split())
    return _ALGO_CANONICAL.get(key, raw)


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

# Constructor defaults when switching to an algorithm that has no matching env
# recipe (Seaquest yaml is DQN-only). Used by algo-switch search reset, not by
# named-goal seeding (that still strips then keeps plan HPs).
_DEFAULT_SB3_HYPERPARAMETERS: dict[str, dict] = {
    "PPO": {
        "learning_rate": 0.0003,
        "n_steps": 2048,
        "batch_size": 64,
        "n_epochs": 10,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_range": 0.2,
        "ent_coef": 0.0,
        "vf_coef": 0.5,
        "max_grad_norm": 0.5,
        "total_timesteps": 300000,
    },
    "A2C": {
        "learning_rate": 0.0007,
        "n_steps": 5,
        "gamma": 0.99,
        "gae_lambda": 1.0,
        "ent_coef": 0.0,
        "vf_coef": 0.5,
        "max_grad_norm": 0.5,
        "total_timesteps": 300000,
    },
    "DQN": {
        "learning_rate": 0.0001,
        "buffer_size": 100000,
        "learning_starts": 1000,
        "batch_size": 32,
        "gamma": 0.99,
        "train_freq": 4,
        "gradient_steps": 1,
        "target_update_interval": 1000,
        "exploration_fraction": 0.15,
        "exploration_initial_eps": 1.0,
        "exploration_final_eps": 0.02,
        "total_timesteps": 300000,
    },
}


def default_sb3_hyperparameters(algorithm: str) -> dict:
    """SB3 constructor defaults for an algorithm with no matching env recipe."""
    canon = (canonicalize_algorithm(algorithm) or algorithm or "").upper()
    raw = _DEFAULT_SB3_HYPERPARAMETERS.get(canon)
    return dict(raw) if raw else {}

# Canonical recipe file per env_id or task_type — hyperparameters and env_kwargs
# from these files are used as defaults when the LLM plan omits a value.
_ENV_RECIPE: dict = {
    "Snake-v0": "snake_ppo_v1.yaml",
    "Snake-v0/DQN": "snake_dqn_v1.yaml",   # algorithm-specific override
    "Tetris-v0": "tetris_actor_critic_v1.yaml",
    "Game2048-v0": "game2048_dqn_v1.yaml",
    "Game2048-v0/LOOKAHEAD_DQN": "game2048_lookahead_dqn_v1.yaml",
    "MinAtar-Breakout-v0": "minatar_breakout_dqn_v1.yaml",
    "MinAtar-v0": "minatar_breakout_dqn_v1.yaml",
    "MinAtar-SpaceInvaders-v0": "minatar_space_invaders_dqn_v1.yaml",
    "MinAtar-Space-Invaders-v0": "minatar_space_invaders_dqn_v1.yaml",
    "MinAtar-Asteroids-v0": "minatar_asteroids_dqn_v1.yaml",
    "MinAtar-Freeway-v0": "minatar_freeway_dqn_v1.yaml",
    "MinAtar-Seaquest-v0": "minatar_seaquest_dqn_v1.yaml",
    "MinAtar-Asterix-v0": "minatar_asterix_dqn_v1.yaml",
    "GridPacMan-v0": "grid_pacman_dqn_v1.yaml",
    "PacMan-v0": "grid_pacman_dqn_v1.yaml",
    "MultiTurnAgentGym-v0": "agent_gym_ppo_v1.yaml",
    "AgentGym-v0": "agent_gym_ppo_v1.yaml",
    "agent-gym": "agent_gym_ppo_v1.yaml",
    "MultiTurnAgentGymText-v0": "agent_gym_ppo_v1.yaml",
    # Platform task_type defaults (no product name). Product LM defaults use
    # default_recipe_for_task(task_type, use_case) → "{use_case}_{task}_v1".
    "agrpo": "agent_gym_agrpo_v1.yaml",
    "agent_gym_agrpo_v1": "agent_gym_agrpo_v1.yaml",
    "agent_gym_agrpo_smoke_v1": "agent_gym_agrpo_smoke_v1.yaml",
    "hstu_ml1m_v1": "hstu_ml1m_v1.yaml",
    "hstu": "hstu_ml1m_v1.yaml",
    "hstu/ml1m": "hstu_ml1m_v1.yaml",
    "hstu/v1": "hstu_ml1m_v1.yaml",
    "mlx_lora": "mlx_lora_v1.yaml",
    # Explicit product recipe stems (plan.recipe already names the file).
    "ensemble_sft_v1": "ensemble_sft_v1.yaml",
    "ensemble_sft_1b_v5": "ensemble_sft_1b_v5.yaml",
    "sft/1b_v5": "ensemble_sft_1b_v5.yaml",
    "ensemble_sft_specialist_ds_v1": "ensemble_sft_specialist_ds_v1.yaml",
    "sft/specialist_ds": "ensemble_sft_specialist_ds_v1.yaml",
    "ensemble_sft_specialist_secretary_v1": "ensemble_sft_specialist_secretary_v1.yaml",
    "sft/specialist_secretary": "ensemble_sft_specialist_secretary_v1.yaml",
    "ensemble_embed_v1": "ensemble_embed_v1.yaml",
    "embed/v1": "ensemble_embed_v1.yaml",
    "ensemble_rerank_v1": "ensemble_rerank_v1.yaml",
    "rerank/v1": "ensemble_rerank_v1.yaml",
    "dpo/sft_chained": "ensemble_sft_dpo_v1.yaml",
    "dpo/sft_dpo": "ensemble_sft_dpo_v1.yaml",
    "dpo/ensemble_sft_dpo_v1": "ensemble_sft_dpo_v1.yaml",
    "ensemble_sft_dpo_v1": "ensemble_sft_dpo_v1.yaml",
    "ensemble_dpo_v1": "ensemble_dpo_v1.yaml",
    "ensemble_dpo_astratune_smoke": "ensemble_dpo_astratune_smoke.yaml",
    "ensemble_agent_grpo_v1": "ensemble_agent_grpo_v1.yaml",
    "ensemble_grpo_v1": "ensemble_grpo_v1.yaml",
    "ensemble_grpo_1b_v1": "ensemble_grpo_1b_v1.yaml",
    "grpo/1b": "ensemble_grpo_1b_v1.yaml",
    "grpo/1b_v1": "ensemble_grpo_1b_v1.yaml",
    "grpo/chained": "ensemble_sft_dpo_grpo_v1.yaml",
    "grpo/ensemble_sft_dpo_grpo_v1": "ensemble_sft_dpo_grpo_v1.yaml",
    "ensemble_sft_dpo_grpo_v1": "ensemble_sft_dpo_grpo_v1.yaml",
    "ensemble_distill_v1": "ensemble_distill_v1.yaml",
    "ensemble_distill_prerl_v1": "ensemble_distill_prerl_v1.yaml",
    "ensemble_distill_gemma4b_v1": "ensemble_distill_gemma4b_v1.yaml",
    "ensemble_rft_v1": "ensemble_rft_v1.yaml",
    "ensemble_rft_focus_v1": "ensemble_rft_focus_v1.yaml",
    "ensemble_rft_focus_v2": "ensemble_rft_focus_v2.yaml",
    "ensemble_rft_focus_v3": "ensemble_rft_focus_v3.yaml",
    "ensemble_rft_focus_v4": "ensemble_rft_focus_v4.yaml",
    "ensemble_rft_focus_v5": "ensemble_rft_focus_v5.yaml",
    "ensemble_rft_focus_v6": "ensemble_rft_focus_v6.yaml",
    "ensemble_rft_focus_v7": "ensemble_rft_focus_v7.yaml",
    "ensemble_opd_v1": "ensemble_opd_v1.yaml",
    "ensemble_rlcd_v1": "ensemble_rlcd_v1.yaml",
    "ensemble_rlcd_focus_v1": "ensemble_rlcd_focus_v1.yaml",
    "ensemble_decision_foundation_v1": "ensemble_decision_foundation_v1.yaml",
    "ensemble_decision_foundation_continue_v1": "ensemble_decision_foundation_continue_v1.yaml",
    "ensemble_decision_specialize_v1": "ensemble_decision_specialize_v1.yaml",
    "ensemble_decision_specialize_focus_v1": "ensemble_decision_specialize_focus_v1.yaml",
    "ensemble_decision_from_scratch_v1": "ensemble_decision_from_scratch_v1.yaml",
    "ensemble_prompt_v1": "ensemble_prompt_v1.yaml",
    "ensemble_post_training_v1": "ensemble_post_training_v1.yaml",
    "ensemble_star_v1": "ensemble_star_v1.yaml",
}

# Product-neutral stem suffixes for default_recipe_for_task (no product name).
# distill omitted — multiple product variants; require an explicit plan.recipe.
_TASK_RECIPE_SUFFIX: dict[str, str] = {
    "sft": "sft_v1",
    "dpo": "dpo_v1",
    "grpo": "grpo_v1",
    "opd": "opd_v1",
    "rft": "rft_v1",
    "rlcd": "rlcd_v1",
    "star": "star_v1",
    "prompt": "prompt_v1",
    "embed": "embed_v1",
    "rerank": "rerank_v1",
    "post-training": "post_training_v1",
}


def _product_task_stems(task_type: str, use_case: str) -> list[str]:
    """Candidate ``{use_case}_…`` recipe stems for a product LM worker."""
    tt = (task_type or "").strip().lower()
    uc = (use_case or "").strip()
    if not tt or not uc:
        return []
    suffix = _TASK_RECIPE_SUFFIX.get(tt)
    if suffix:
        return [f"{uc}_{suffix}"]
    # distill has several product variants; prefer the canonical ``*_distill_v1``.
    if tt == "distill":
        return [
            f"{uc}_distill_v1",
            f"{uc}_distill_prerl_v1",
            f"{uc}_distill_gemma4b_v1",
        ]
    return []


def _infer_use_case_for_task(task_type: str) -> Optional[str]:
    """Pick a configured product pack that owns a default recipe for ``task_type``.

    Used when plan HP omit ``use_case`` but ``ASTRA_PRODUCT_RECIPES`` is set —
    bare_eval / crystallizer / preflight resolve ``dpo``/``grpo``/``distill``/…
    against pack YAML without hardcoding a brand into ``_ENV_RECIPE``.
    """
    from backend.services.recipe_paths import find_recipe_file, parse_product_recipes

    tt = (task_type or "").strip().lower()
    packs = parse_product_recipes()
    if not packs or not tt:
        return None
    for uc in packs:
        for stem in _product_task_stems(tt, uc):
            if find_recipe_file(stem, use_case=uc):
                return uc
    if len(packs) == 1:
        return next(iter(packs))
    return None


def default_recipe_for_task(
    task_type: str, use_case: Optional[str] = None,
) -> Optional[str]:
    """Resolve a default recipe stem when plan.recipe is unset.

    Platform workers (agrpo/hstu/mlx_lora) use _ENV_RECIPE. Product LM workers
    need an explicit ``use_case`` and become ``{use_case}_{suffix}`` (e.g.
    ensemble + dpo → ensemble_dpo_v1). distill returns None — multiple product
    variants; pick a recipe by name. Never infers a brand from
    ``ASTRA_PRODUCT_RECIPES`` (callers that need pack fallback use
    ``_infer_use_case_for_task`` + ``_product_task_stems``).
    """
    tt = (task_type or "").strip().lower()
    if not tt:
        return None
    # Platform defaults (filename without .yaml)
    plat = _ENV_RECIPE.get(tt)
    if plat and not plat.startswith("ensemble_"):
        return plat.removesuffix(".yaml").removesuffix(".yml")
    uc = (use_case or "").strip()
    suffix = _TASK_RECIPE_SUFFIX.get(tt)
    if not uc or not suffix:
        return None
    return f"{uc}_{suffix}"


def _load_recipe_for_env(env_id: str, algorithm: str = "") -> dict:
    """Return the parsed recipe dict for env_id (+ optional algorithm), or {}."""
    from backend.services.recipe_paths import find_recipe_file

    # Algorithm-specific override takes priority (e.g. Snake-v0/DQN)
    filename = (
        _ENV_RECIPE.get(f"{env_id}/{algorithm}")
        or _ENV_RECIPE.get(algorithm)
        or _ENV_RECIPE.get(env_id)
    )
    path = None
    if filename:
        path = find_recipe_file(filename)
    if not path and env_id:
        path = find_recipe_file(env_id)
    if not path and algorithm:
        path = find_recipe_file(algorithm)
    if not path:
        return {}
    try:
        import yaml as _yaml
        with open(path) as f:
            return _yaml.safe_load(f) or {}
    except Exception:
        return {}


def _recipe_hyperparameters_for(env_id: str, algorithm: str = "") -> Optional[dict]:
    """Recipe HPs when they belong to this algorithm; None if there is no match.

    Env-keyed recipes (Seaquest) always load the DQN yaml even for a named PPO
    goal. Applying those HPs would put buffer_size on a PPO plan. A recipe
    without an `algorithm` field still applies (Snake PPO yaml).
    """
    recipe = _load_recipe_for_env(env_id, algorithm) or {}
    recipe_hp = recipe.get("hyperparameters") or {}
    if not isinstance(recipe_hp, dict) or not recipe_hp:
        return None
    recipe_algo = canonicalize_algorithm(recipe.get("algorithm") or "")
    plan_algo = canonicalize_algorithm(algorithm) if algorithm else ""
    if recipe_algo and plan_algo and recipe_algo != plan_algo:
        return None
    return dict(recipe_hp)


def _strip_foreign_hp_keys(plan_hp: dict, algorithm: str) -> dict:
    """Drop constructor keys that belong to another SB3 algo; keep pivots.

    Real incident: Lead Agent PPO prior (`n_steps`, `n_epochs`) survived on a
    DQN Seaquest plan. Stripping those must not also discard a later DQN pivot
    such as `exploration_fraction=0.4` or `learning_rate=0.0005`.
    """
    canon = (canonicalize_algorithm(algorithm) or algorithm or "").upper()
    own = _VALID_ALGO_KEYS.get(canon, set())
    if not own:
        return dict(plan_hp or {})
    foreign: set = set()
    for name, keys in _VALID_ALGO_KEYS.items():
        if name == canon:
            continue
        foreign |= keys
    foreign -= own
    return {k: v for k, v in (plan_hp or {}).items() if k not in foreign}


def recipe_metric_ceiling(env_id: str, algorithm: str = "") -> dict:
    """Return the recipe's declared `metric_ceiling` map ({metric_name: value}),
    or {} if the recipe doesn't declare one. This is the empirically-observed
    best a recipe+model+eval combo can reach; missions whose target exceeds it
    are rejected at creation time (see backend/routers/missions.py) rather than
    left to grind for weeks against an unreachable goal."""
    ceiling = _load_recipe_for_env(env_id, algorithm).get("metric_ceiling")
    return ceiling if isinstance(ceiling, dict) else {}


# Real incident: with the recipe fully authoritative (see below), a stuck dpo/grpo mission's
# pivots became complete no-ops — every escalation cycle relaunched the identical recipe-locked
# run, with zero directed change, for 47+ consecutive iterations. Preference-pair sampling
# diversity is the one axis safe to vary per pivot: it affects what gets collected, never model/
# training stability, unlike learning_rate/num_layers/adapter path (the actually-dangerous knobs
# from the original incident this lockout exists to prevent). Bounds keep collection cost/shape
# sane: k_collect/num_generations too high blows up wall-clock per case; too low starves DPO/GRPO
# of contrast. num_generations capped at 4 specifically because K=4 regressed in grpo_v10_min
# per the product GRPO recipe changelog.
# distill has no sampling phase — its one safe pivot knob is `iters` (train longer/shorter).
# Scoped per task type so a dpo recipe never picks up a distill key (or vice versa).
_FINETUNE_PIVOT_RANGES_BY_TASK = {
    "dpo":     {"temp": (0.7, 1.5), "k_collect": (4, 16)},
    "grpo":    {"temp": (0.7, 1.5), "num_generations": (2, 4)},
    "agrpo":   {
        "temp": (0.7, 1.5),
        "num_generations": (2, 4),
        "iters": (20, 200),
        "max_turns": (4, 12),
    },
    # distill: NO pivot lever as of 2026-09-05. `iters` was the one safe knob, but
    # it let a plan override the recipe's value — _resolve_hyperparams applies
    # recipe_hp then hp.update(clamped plan keys), so the LeadAgent's proposal
    # wins. Mission 472e226c launched at iters=800 (the clamp ceiling) against a
    # recipe declaring 500, which would have made the warm-start NOT the only
    # changed variable versus its comparator 594322b2 — the entire point of the
    # pre-RL experiment. Empty until that experiment concludes; restore
    # {"iters": (200, 800)} to re-enable pivoting.
    "distill": {},
    # RFT's lever is how many candidates it draws before filtering: more samples
    # means more chances a borderline case yields a correct completion to train
    # on, which is the whole mechanism. temp must stay > 0 or every sample is
    # identical and rejection sampling has nothing to reject.
    # Temp ceiling 1.0: mission 6f4204c7's pivot to 1.2 dropped a 0.926 best
    # to 0.852 — same collapse class as OPD above 1.0.
    "rft":     {"k_samples": (4, 16), "temp": (0.7, 1.0)},
    # OPD samples student rollouts then grades them with teacher reverse KL —
    # num_generations and temp are the diversity levers. Temp ceiling 1.0:
    # 7b6c78a6 pivots to 1.2–1.3 repeatedly collapsed mid-run held-out to 0%.
    "opd":     {"num_generations": (2, 4), "temp": (0.7, 1.0)},
    # RLCD samples M rationales. Temp ceiling 1.0: the paper samples at 1, and
    # the same collapse seen in OPD/RFT above 1.0 applies to rationale sampling.
    # empty_rationale=True (Choice-facing / decision foundation) overrides this
    # to {} in _clamp_finetune_pivot_hp — live /v1/rlcd/readout posts an empty
    # rationale so M must stay 1; mission 2235f69e burned 26 pivots forcing
    # num_generations into 2–4 while held-out stayed flat at 0.576.
    "rlcd":    {"num_generations": (2, 4), "temp": (0.7, 1.0)},
    # prompt missions evaluate candidate rule variants greedily at temperature 0;
    # there are no numerical hyperparameters to tune via pivot.
    "prompt":  {},
    "sft":     {"learning_rate": (1e-5, 5e-4), "iters": (50, 300)},
    # embed/rerank/hstu: recipe-locked external trainers — no pivot levers.
    "embed":   {},
    "rerank":  {},
    "hstu":    {},
    "post-training": {},
    "star":    {"k_samples": (4, 16), "temp": (0.7, 1.5), "rationalize_temp": (0.5, 1.2), "learning_rate": (1e-5, 5e-4)},
}


def _clamp_finetune_pivot_hp(
    task_type: str, plan_hp: dict, *, empty_rationale: bool = False,
) -> dict:
    """Clamp the small safelist of pivot keys valid for this finetune-remote task
    type; silently drop everything else.

    When ``empty_rationale`` is set (Choice-facing RLCD), return {} — sampling
    diversity levers would break the M=1 empty-rationale contract that matches
    live readout, and they do not move held-out pass_rate (2235f69e / 70791e17).
    """
    ranges = _FINETUNE_PIVOT_RANGES_BY_TASK.get(task_type, {})
    if task_type == "rlcd" and empty_rationale:
        ranges = {}
    clamped = {}
    for k, (lo, hi) in ranges.items():
        if k not in plan_hp:
            continue
        try:
            v = type(lo)(plan_hp[k])
        except (TypeError, ValueError):
            continue
        clamped[k] = max(lo, min(hi, v))
    return clamped


def _csv_cases_flag(hp: dict, key: str, cli_flag: str) -> str:
    """Optional ``--{cli_flag} id,id,...`` argv fragment for Mini finetune CLIs.

    Recipe may set ``key`` as a CSV string or a list of case ids. Empty /
    missing → omit the flag entirely (do not emit an empty string list element).
    """
    raw = hp.get(key)
    if not raw:
        return ""
    if isinstance(raw, (list, tuple)):
        csv = ",".join(str(x).strip() for x in raw if str(x).strip())
    else:
        csv = str(raw).strip()
    if not csv:
        return ""
    return f'"{cli_flag}", "{csv}",'



def _csv_cases(raw) -> str:
    """Normalize recipe focus/force-gold cases to a CSV string for facade kwargs."""
    if raw is None or raw is False:
        return ""
    if isinstance(raw, (list, tuple)):
        return ",".join(str(x).strip() for x in raw if str(x).strip())
    return str(raw).strip()


def _focus_cases_from_fail_slice() -> str:
    """Latest AstraData FailSlice case_ids CSV (Eval → Data → train)."""
    try:
        from astradata.fail_slice import FailSliceStore
        from backend.config import settings
        import os

        bank = os.path.join(settings.data_path, "astradata", "fail_slices")
        return FailSliceStore(bank_root=bank).focus_cases_csv(suite_id="routing_bare")
    except Exception:
        return ""


def _resolve_focus_cases(hp: dict) -> str:
    """Recipe focus_cases, else FailSlice CSV when ensemble / explicitly opted in.

    Mission create already stamps ``focus_cases`` for ensemble when a slice
    exists. Auto-fallback here only when ``use_case=ensemble`` or
    ``fail_slice_focus`` is set — never for a bare empty hp dict (keeps unit
    tests and non-ensemble recipes from picking up a stale bank).
    """
    focus = _csv_cases(hp.get("focus_cases"))
    if focus:
        return focus
    if hp.get("fail_slice_focus") or str(hp.get("use_case") or "").strip() == "ensemble":
        return _focus_cases_from_fail_slice()
    return ""


def _focus_cases_flag(hp: dict) -> str:
    """Optional ``--focus-cases id,id,...`` argv fragment for Mini finetune CLIs."""
    csv = _resolve_focus_cases(hp)
    if not csv:
        return ""
    return f'"--focus-cases", "{csv}",'


def _force_gold_cases_flag(hp: dict) -> str:
    """Optional ``--force-gold-cases id,id,...`` argv fragment for RFT.

    Surgical curated-mock injection — must NOT be the whole focus list
    (astra 7cc3beb4: blanket focus-gold → 52/54 below warm-start 53/54).
    """
    return _csv_cases_flag(hp, "force_gold_cases", "--force-gold-cases")


def _sft_resume_adapter_file(adapter_path: str) -> str:
    """Resolve a resume path for ``sft_train.py`` (dir or ``.safetensors``)."""
    path = (adapter_path or "").strip().rstrip("/")
    if not path:
        return ""
    if path.endswith(".safetensors"):
        return path
    return f"{path}/adapters.safetensors"


def _sft_resume_adapter_arg(adapter_path: str) -> str:
    """Optional ``--resume-adapter-file`` argv fragment for ``sft_train.py``.

    Accepts a directory (``adapters/retrain_1b_v4`` or ``…/best``) or a direct
    ``.safetensors`` path. Empty → omit the flag (cold start).
    """
    resume_file = _sft_resume_adapter_file(adapter_path)
    if not resume_file:
        return ""
    return f'"--resume-adapter-file", "{resume_file}",'


def _sft_resume_adapter_kwarg(adapter_path: str) -> str:
    """Optional ``resume_adapter_file=...`` kwarg for astratune.make(mlx_sft)."""
    resume_file = _sft_resume_adapter_file(adapter_path)
    if not resume_file:
        return ""
    return f'resume_adapter_file="{resume_file}",'


def _resolve_hyperparams(
    env_id: str, plan_hp: dict, algorithm: str = "", warm_start_adapter: Optional[str] = None,
) -> dict:
    """Apply recipe hyperparameters as defaults for keys the LLM plan did not set."""
    recipe = _load_recipe_for_env(env_id, algorithm)
    if not recipe:
        from backend.services.recipe_paths import find_recipe_file

        _uc = str((plan_hp or {}).get("use_case") or "").strip()
        if not _uc:
            _uc = _infer_use_case_for_task(env_id) or ""
        _stem = default_recipe_for_task(env_id, _uc or None)
        # distill (and other multi-variant tasks) have no default_recipe stem —
        # walk pack candidates when a use_case is known.
        if not _stem and _uc:
            for cand in _product_task_stems(env_id, _uc):
                if find_recipe_file(cand, use_case=_uc):
                    _stem = cand
                    break
        if _stem:
            recipe = _load_recipe_for_env(_stem, algorithm)
    recipe_hp = recipe.get("hyperparameters", {})
    task_type = recipe.get("task_type", env_id)
    if env_id in ("dpo", "grpo", "agrpo", "distill", "rft", "opd", "rlcd", "prompt", "ensemble_sft_v1", "ensemble_sft_1b_v5", "ensemble_grpo_1b_v1", "ensemble_sft_specialist_ds_v1", "ensemble_sft_specialist_secretary_v1", "ensemble_embed_v1", "embed", "ensemble_rerank_v1", "rerank", "hstu_ml1m_v1", "hstu", "star", "ensemble_star_v1", "ensemble_rlcd_v1", "ensemble_rlcd_focus_v1", "ensemble_decision_foundation_v1", "ensemble_decision_foundation_continue_v1", "ensemble_decision_specialize_v1", "ensemble_decision_specialize_focus_v1", "ensemble_decision_from_scratch_v1", "agent_gym_agrpo_v1", "agent_gym_agrpo_smoke_v1", "ensemble_agent_grpo_v1") or task_type in ("dpo", "grpo", "agrpo", "distill", "rft", "opd", "rlcd", "prompt", "star", "embed", "rerank", "hstu"):
        # Recipe is authoritative for everything except the small sampling-diversity
        # safelist above — no plan/pivot override allowed for anything else. These are
        # LoRA/optimizer settings tuned against a specific warm-start adapter;
        # PIVOT_SYSTEM's hyperparameter guidance is RL-oriented (PPO/DQN ranges) and
        # doesn't know these are recipe-locked.
        hp = dict(recipe_hp)
        hp.update(_clamp_finetune_pivot_hp(
            task_type, plan_hp,
            empty_rationale=bool(recipe_hp.get("empty_rationale")),
        ))
        # Product plugin binds at mission create / Lead Agent plan — not recipe HP.
        # Plan use_case wins; strip stale recipe-stamped use_case when plan omits it
        # only if plan explicitly set empty? Prefer plan when present.
        _plan_uc = str((plan_hp or {}).get("use_case") or "").strip()
        if _plan_uc:
            hp["use_case"] = _plan_uc
        # DAPO-lite / rollout backend: recipe defaults + explicit plan overrides
        # (not in the temp/num_generations pivot safelist — they are opt-in experiment knobs).
        if (task_type or env_id) in ("grpo", "ensemble_grpo_1b_v1") or env_id in (
            "grpo", "ensemble_grpo_1b_v1", "grpo/1b", "grpo/1b_v1",
        ):
            for _k in (
                "clip_epsilon_low", "clip_epsilon_high", "dynamic_sample_retries",
                "rollout_backend", "rollout_base_url", "rollout_api_key", "rollout_model",
            ):
                if _k in plan_hp and plan_hp[_k] is not None:
                    hp[_k] = plan_hp[_k]
        # warm_start_adapter comes from Mission.last_checkpoint_path (state_machine,
        # system-controlled), never from plan_hp/the LLM pivot — deliberately outside
        # the plan_hp merge above so it can't be spoofed through the same channel the
        # incident above was about. Real gap this closes: without it, every dpo/grpo
        # iteration re-trained from the recipe's static warm-start adapter forever,
        # never from a prior iteration's own improved output — 146 iterations across
        # 26 days on mission 7fd88324 never once built on each other as a result.
        if warm_start_adapter:
            hp["adapter"] = warm_start_adapter
            # A cold-start recipe is cold for iteration 0 ONLY. Once an
            # iteration has produced a checkpoint, later iterations chain from
            # it and are ordinary warm starts — leaving no_adapter set would
            # emit --no-adapter alongside a real adapter path, which
            # distill rejects adapter + no_adapter as mutually exclusive,
            # and would silently discard every iteration's progress if it
            # didn't. Clearing it here rather than in the template keeps the
            # rule with the value it depends on.
            hp["no_adapter"] = False
        return hp
    matching_hp = _recipe_hyperparameters_for(env_id, algorithm)
    recipe_hp = matching_hp if matching_hp is not None else {}
    hp = _strip_foreign_hp_keys(plan_hp, algorithm)
    fill = recipe_hp if matching_hp is not None else default_sb3_hyperparameters(algorithm)
    for k, v in fill.items():
        hp.setdefault(k, v)
    # If plan provided a placeholder for base_model (e.g. "model" or product name)
    # that is neither an HF repo ID with a slash nor an existing local path, fall back to recipe's base_model
    base_m = str(hp.get("base_model", ""))
    if base_m and "/" not in base_m and not os.path.exists(base_m) and "base_model" in recipe_hp:
        hp["base_model"] = recipe_hp["base_model"]
    return hp


_DEFAULT_ASTRA_PKG_ROOT = "/Users/kewang/astra-pkg"


def _resolve_astra_pkg_root(hp: Optional[dict] = None) -> str:
    """Absolute path to the editable Astra tree on the sandbox host.

    Same role as ``finetune_dir`` / ``python_bin``: baked into canned wrappers at
    codegen so ``import astrarl`` / ``import astratune`` resolve after mirrors
    were removed. Resolution order:

    1. recipe / plan HP ``astra_pkg_root``
    2. ``ASTRA_PKG_ROOT`` (``settings.pkg_root``)
    3. ``/Users/kewang/astra-pkg``

    ``~`` is expanded at codegen time so generated scripts use absolute paths
    like ``finetune_dir``, not runtime ``expanduser``.
    """
    raw = ""
    if hp:
        raw = str(hp.get("astra_pkg_root") or "").strip()
    if not raw:
        raw = (settings.pkg_root or "").strip()
    if not raw:
        raw = _DEFAULT_ASTRA_PKG_ROOT
    return os.path.expanduser(raw)



def _looks_like_learner_id_use_case(use_case: str) -> bool:
    """``mlx_dpo`` etc. are learner_ids, not UseCasePlugin modules."""
    uc = (use_case or "").strip().lower()
    return bool(uc) and (uc.startswith("mlx_") or uc in {
        "sb3_ppo", "sb3_dqn", "sb3_sac", "sb3_td3", "actor_critic", "hstu",
    })


def _resolve_use_case(hp: Optional[dict] = None) -> str:
    """Importable UseCasePlugin module for codegen facades.

    Resolution order:
    1. Plan / mission-create hyperparameters ``use_case`` (source of truth)
    2. ``ASTRA_USE_CASE`` env — last-resort bare-CLI fallback only, not mission binding

    Empty string means facades omit ``--use-case`` and the CLI must get the
    module from the environment on the train host. Learner_id-shaped values
    (``mlx_dpo``) are ignored — Lead Agent has hallucinated those into use_case.
    """
    import os
    if hp:
        raw = str(hp.get("use_case") or "").strip()
        if raw and not _looks_like_learner_id_use_case(raw):
            return raw
    env = os.environ.get("ASTRA_USE_CASE", "").strip()
    if env and not _looks_like_learner_id_use_case(env):
        return env
    return ""


def finetune_checkpoint_dir(task_type: str, plan: dict, mission_id: str, iteration: int) -> str:
    """Remote checkpoint/adapter output dir for dpo/grpo missions — under
    finetune_dir/adapters/, alongside the product workspace adapter inventory
    (grpo_v<N>_min/, dpo_v<N>_min/, retrain_best/, ...), so
    astra-produced adapters are part of the normal inventory rather than a
    separate mission-scoped location. Shared by CodeGenerator.generate_training_script
    (to bake --save-dir into the wrapper script) and SandboxManager.launch (to know
    where SSHSandbox._sync_back() should rsync the adapter from).

    Iteration-scoped (astra_<mission>_iter<N>), not just mission-scoped — a real
    incident showed a mission-level-only path (every iteration sharing one
    directory) let each iteration's --save-dir write silently clobber whatever
    the previous iteration had left there, including a genuine best result: once
    the Phase 35 checkpoint-chaining fix pointed a later iteration's --adapter
    warm-start at that same shared directory, it was reading back an already-
    overwritten, arbitrary state, not the checkpoint that actually produced the
    best score. Scoping by iteration makes every iteration's output directory
    permanently distinct — nothing can ever overwrite a prior iteration's
    checkpoint again, so chaining from a specific past iteration's directory is
    now structurally safe rather than relying on nothing else having run since."""
    recipe_key = plan.get("recipe") or task_type
    hp = _resolve_hyperparams(recipe_key, plan.get("hyperparameters", {}))
    finetune_dir = hp.get("finetune_dir", "")
    if task_type == "post-training":
        stage_idx = plan.get("stage_index", 0)
        active_task = plan.get("active_task_type", "sft")
        return os.path.join(finetune_dir, "adapters", f"astra_{mission_id[:8]}_stage{stage_idx+1}_{active_task}_iter{iteration}")
    return os.path.join(finetune_dir, "adapters", f"astra_{mission_id[:8]}_iter{iteration}")


def finetune_checkpoint_dir_relative(mission_id: str, iteration: int, stage_index: Optional[int] = None, active_task_type: Optional[str] = None) -> str:
    """The --adapter-compatible /best CANDIDATE path (DPO trainer resolves it
    relative to the process's cwd, which the generated wrapper script chdir's
    to finetune_dir before exec'ing) for chaining a dpo/grpo mission's own
    prior-best adapter into the next iteration's --adapter, instead of always
    re-reading the recipe's static warm-start adapter. Iteration-scoped — see
    finetune_checkpoint_dir()'s docstring for why.

    NOT guaranteed to exist: the DPO trainer writes <save-dir>/best/ only when it
    tracks a best-during-training checkpoint that beats the final epoch's own
    output — when the final epoch's checkpoint is itself the best seen, no
    best/ subdir is written at all (confirmed ~50% of runs via filesystem
    audit on mac-mini). Callers must not use this candidate path directly for
    an actual --adapter warm-start; LoopStateMachine._resolve_adapter_or_bare()
    checks the remote host and falls back to finetune_checkpoint_dir()'s bare
    iteration directory when best/ doesn't exist. Skipping that check crashes
    the trainer instantly with FileNotFoundError (real incident: silently
    looped a mission for ~7h/558 iterations with no checkpoint ever produced).

    Still preferred over the bare directory whenever best/ does exist, though:
    The trainer's <save-dir>/adapters.safetensors is the raw final-epoch
    output, while <save-dir>/best/adapters.safetensors is the actual best
    checkpoint seen at any point during that run — training within an
    iteration has repeatedly been observed to degrade from its own starting
    point. Confirmed live: a mission's reported new best of 0.773 only
    reproduced when loading .../best/adapters.safetensors directly (0.7727,
    matching exactly); loading the bare directory's adapters.safetensors
    instead reproduced 0.2424, a completely different, much worse model state
    that happened to share a directory with the real one."""
    if stage_index is not None and active_task_type is not None:
        return os.path.join("adapters", f"astra_{mission_id[:8]}_stage{stage_index+1}_{active_task_type}_iter{iteration}", "best")
    return os.path.join("adapters", f"astra_{mission_id[:8]}_iter{iteration}", "best")


class CodeGenerator:
    def __init__(self, provider: InferenceProvider) -> None:
        self._provider = provider

    @staticmethod
    def valid_algo_keys(algorithm: str) -> set:
        """Return the set of valid SB3 constructor kwargs for an algorithm (empty = unknown)."""
        canon = canonicalize_algorithm(algorithm)
        return _VALID_ALGO_KEYS.get(canon.upper(), set()) if canon else set()

    async def generate_training_script(
        self,
        mission_id: str,
        plan: dict,
        current_iteration: int = 0,
        warm_start_adapter: Optional[str] = None,
    ) -> str:
        """
        Generate a training script and write it to data/missions/{id}/train.py.
        Returns the path to the written script.
        """
        task_type = plan.get("task_type", "rl")
        if task_type in _FINETUNE_REMOTE_TASK_TYPES and settings.sandbox_host:
            checkpoint_dir = finetune_checkpoint_dir(task_type, plan, mission_id, current_iteration)
        else:
            # Absolute: embed (and other local) wrappers may chdir before exec.
            checkpoint_dir = os.path.abspath(
                os.path.join(settings.data_path, "missions", mission_id, "checkpoints")
            )
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

        user_prompt = self._build_user_prompt(
            task_type, mission_id, plan, checkpoint_dir, current_iteration,
            warm_start_adapter=warm_start_adapter,
        )

        active_task = plan.get("active_task_type", "sft") if task_type == "post-training" else task_type
        if active_task in ("embed", "rerank"):
            # Local-only: sentence-transformers on the Astra host (product .venv).
            # Never SSH to the Mini — Mini finetune-env is MLX, not ST.
            code = _EMBED_LOCAL_WRAPPER.format(
                **self._build_embed_context(mission_id, plan, checkpoint_dir)
            )
        elif active_task == "hstu":
            # Local CUDA: Meta generative-recommenders. Not Mini MLX finetune-remote.
            code = _HSTU_LOCAL_WRAPPER.format(
                **self._build_hstu_context(mission_id, plan, checkpoint_dir)
            )
        elif active_task == "sft":
            # Remote: canned astratune.make("mlx_sft") facade. Local: canonical runner.
            sft_ctx = self._build_sft_context(
                mission_id, plan, checkpoint_dir, current_iteration, warm_start_adapter
            )
            if settings.sandbox_host:
                code = _SFT_REMOTE_WRAPPER.format(**sft_ctx)
            else:
                code = _CANONICAL_SFT_RUNNER.format(**sft_ctx)
        elif active_task == "rlcd":
            # Canned wrapper. _build_user_prompt already formatted it from the
            # recipe; do not ask the coder model to reimplement the readout.
            code = user_prompt
        elif active_task == "grpo":
            # Canned facade wrapper — LLM previously imported astrarl before
            # sys.path/chdir and invented astragym.make("gymnasium:None").
            code = _GRPO_REMOTE_WRAPPER.format(
                **self._build_grpo_context(
                    mission_id, plan, checkpoint_dir, current_iteration,
                    warm_start_adapter=warm_start_adapter,
                )
            )
        elif active_task == "agrpo":
            code = _AGRPO_REMOTE_WRAPPER.format(
                **self._build_agrpo_context(
                    mission_id, plan, checkpoint_dir, current_iteration,
                    warm_start_adapter=warm_start_adapter,
                )
            )
        elif active_task == "prompt":
            prompt_ctx = self._build_prompt_context(
                mission_id, plan, checkpoint_dir, current_iteration,
                warm_start_adapter=warm_start_adapter,
            )
            rules_prompt = _PROMPT_RULES_PROMPT.format(**prompt_ctx)
            messages = [
                Message(role="system", content=(
                    "You propose product routing EXTRA_RULES only. "
                    "No Python, no scripts, no os.execv."
                )),
                Message(role="user", content=rules_prompt),
            ]
            rules = await self._provider.generate(
                messages, GenerationConfig(max_tokens=2048, temperature=0.4),
            )
            rules = self._strip_fences(rules).strip()
            if not rules:
                rules = (
                    "# (empty rules — healer/provider returned nothing)\n"
                    "# Prefer concrete negative constraints over meta-adages."
                )
            code = _PROMPT_REMOTE_WRAPPER.format(
                **prompt_ctx,
                extra_rules_repr=repr(rules),
            )
        elif active_task == "dpo":
            code = _DPO_REMOTE_WRAPPER.format(
                **self._build_dpo_context(
                    mission_id, plan, checkpoint_dir, current_iteration,
                    warm_start_adapter=warm_start_adapter,
                )
            )
        elif active_task == "distill":
            code = _DISTILL_REMOTE_WRAPPER.format(
                **self._build_distill_context(
                    mission_id, plan, checkpoint_dir, current_iteration,
                    warm_start_adapter=warm_start_adapter,
                )
            )
        elif active_task == "rft":
            code = _RFT_REMOTE_WRAPPER.format(
                **self._build_rft_context(
                    mission_id, plan, checkpoint_dir, current_iteration,
                    warm_start_adapter=warm_start_adapter,
                )
            )
        elif active_task == "opd":
            code = _OPD_REMOTE_WRAPPER.format(
                **self._build_opd_context(
                    mission_id, plan, checkpoint_dir, current_iteration,
                    warm_start_adapter=warm_start_adapter,
                )
            )
        elif active_task == "star":
            star_ctx = self._build_star_context(
                mission_id, plan, checkpoint_dir, current_iteration, warm_start_adapter
            )
            if settings.sandbox_host:
                code = _STAR_REMOTE_WRAPPER.format(**star_ctx)
            else:
                messages = [
                    Message(role="system", content=system_prompt),
                    Message(role="user", content=user_prompt),
                ]
                code = await self._provider.generate(messages, GenerationConfig(max_tokens=4096, temperature=0.1))
                code = self._strip_fences(code)
        else:
            messages = [
                Message(role="system", content=system_prompt),
                Message(role="user", content=user_prompt),
            ]
            code = await self._provider.generate(messages, GenerationConfig(max_tokens=4096, temperature=0.1))
            code = self._strip_fences(code)
            if task_type == "rl":
                code = self._patch_undefined_logger(code)
                env_id = plan.get("env_id", "")
                _proj_root = os.path.abspath(os.path.join(settings.data_path, ".."))
                if env_id == "Snake-v0" and "register" not in code:
                    code = _SNAKE_SETUP.format(project_root=_proj_root) + "\n" + code
                    logger.info("CodeGenerator: injected Snake-v0 registration preamble")
                elif env_id == "Tetris-v0" and "register" not in code:
                    code = _TETRIS_SETUP.format(project_root=_proj_root) + "\n" + code
                    logger.info("CodeGenerator: injected Tetris-v0 registration preamble")
                elif env_id in ("Game2048-v0", "2048") and "register" not in code:
                    code = _GAME2048_SETUP.format(project_root=_proj_root) + "\n" + code
                    logger.info("CodeGenerator: injected Game2048-v0 registration preamble")
                elif env_id in (
                    "MinAtar-Breakout-v0", "MinAtar-v0", "minatar", "minatar-breakout",
                    "MinAtar-SpaceInvaders-v0", "MinAtar-Space-Invaders-v0", "minatar-space-invaders",
                    "MinAtar-Asteroids-v0", "minatar-asteroids",
                    "MinAtar-Freeway-v0", "minatar-freeway", "freeway",
                    "MinAtar-Seaquest-v0", "minatar-seaquest", "seaquest",
                    "MinAtar-Asterix-v0", "minatar-asterix", "asterix",
                    "GridPacMan-v0", "grid-pacman", "pacman",
                ) and "register" not in code:
                    code = _MINATAR_SETUP.format(project_root=_proj_root) + "\n" + code
                    logger.info("CodeGenerator: injected MinAtar registration preamble")
                elif env_id in (
                    "MultiTurnAgentGym-v0", "AgentGym-v0", "agent-gym", "MultiTurnAgentGymText-v0",
                ) and "register" not in code:
                    code = _AGENT_GYM_SETUP.format(project_root=_proj_root) + "\n" + code
                    logger.info("CodeGenerator: injected MultiTurnAgentGym-v0 registration preamble")
                # Inject curriculum loop if recipe defines phases
                _algo = canonicalize_algorithm(plan.get("algorithm", "PPO")) or "PPO"
                _recipe = _load_recipe_for_env(env_id, _algo)
                _curriculum_phases = (_recipe.get("curriculum") or {}).get("phases")
                if _curriculum_phases:
                    _env_kw = _resolve_env_kwargs(env_id, plan.get("env_kwargs"))
                    _metric_name = next(iter(plan.get("target_metric") or {}), "food_eaten")
                    code = self._inject_curriculum(code, _curriculum_phases, env_id, _env_kw, _metric_name)
                    logger.info("CodeGenerator: injected curriculum (%d phases) for %s/%s", len(_curriculum_phases), env_id, _algo)
                _env_kw = _resolve_env_kwargs(env_id, plan.get("env_kwargs"))
                _env_kw_str = ""
                if _env_kw:
                    _env_kw_str = ", " + ", ".join(f"{k}={v!r}" for k, v in _env_kw.items())
                _hp = plan.get("hyperparameters") or {}
                _gym_id = (
                    plan.get("gym_id")
                    or _hp.get("gym_id")
                    or (f"gymnasium:{env_id}" if env_id else "")
                )
                # Pin AstraGym env before import patch so unused gymnasium can be stripped.
                code = self._ensure_rl_gym_make(
                    code, env_id, _env_kw_str, gym_id=_gym_id or None,
                )
                code = self._patch_rl_imports(code)
            # Fix any relative checkpoint paths the LLM may have substituted for the absolute checkpoint_dir
            code = self._fix_checkpoint_paths(code, checkpoint_dir)
            # Fix os.execv(*argv) star-unpacking, which crashes instantly at runtime (see docstring)
            code = self._fix_execv_unpacking(code)
            # os.execv(sys.executable) would run the train CLI under the sandbox launcher
            # (often plain python3), which has no mlx. Pin the finetune interpreter.
            code = self._pin_finetune_exec_python(code)

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
                "algorithm": canonicalize_algorithm(plan.get("algorithm", "PPO")) or "PPO",
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

    def _build_hstu_context(
        self,
        mission_id: str,
        plan: dict,
        checkpoint_dir: str,
    ) -> dict:
        recipe_key = plan.get("recipe") or plan.get("task_type") or "hstu"
        hp = _resolve_hyperparams(recipe_key, plan.get("hyperparameters", {}))
        project_root = os.path.abspath(os.path.join(settings.data_path, ".."))
        venv_python = os.path.join(project_root, ".venv", "bin", "python")
        dataset_id = str(
            plan.get("dataset_id")
            or hp.get("dataset_id")
            or "hstu_ml1m"
        )
        dataset_rev = int(
            plan.get("dataset_rev")
            or hp.get("dataset_rev")
            or 1
        )
        data_dir = str(hp.get("data_dir") or f"data/astradata/{dataset_id}/{dataset_rev}")
        if not os.path.isabs(data_dir):
            data_dir = os.path.join(project_root, data_dir)
        ckpt = os.path.abspath(checkpoint_dir)
        return {
            "mission_id": mission_id,
            "checkpoint_dir": ckpt,
            "python_bin": str(hp.get("python_bin", venv_python if os.path.isfile(venv_python) else sys.executable)),
            "data_dir": data_dir,
            "dataset_id": dataset_id,
            "dataset_rev": str(dataset_rev),
            "epochs": int(hp.get("epochs", 5)),
            "batch_size": int(hp.get("batch_size", 256)),
            "learning_rate": float(hp.get("learning_rate", 1e-3)),
            "d_model": int(hp.get("d_model", 64)),
            "n_layers": int(hp.get("n_layers", 2)),
            "n_heads": int(hp.get("n_heads", 2)),
            "max_len": int(hp.get("max_len", 50)),
            "dropout": float(hp.get("dropout", 0.1)),
            "weight_decay": float(hp.get("weight_decay", 0.01)),
            "seed": int(hp.get("seed", 42)),
        }

    def _build_embed_context(
        self,
        mission_id: str,
        plan: dict,
        checkpoint_dir: str,
    ) -> dict:
        recipe_key = plan.get("recipe") or plan.get("task_type") or "embed"
        hp = _resolve_hyperparams(recipe_key, plan.get("hyperparameters", {}))
        # Recipe may use product_root or legacy ensemble_root key.
        product_root = str(
            hp.get("product_root") or hp.get("ensemble_root") or ""
        ).strip()
        default_script = (
            "finetune/train_reranker.py"
            if (plan.get("task_type") == "rerank" or "rerank" in str(recipe_key))
            else "finetune/train_embedding.py"
        )
        default_data = (
            "finetune/data_ft/rerank"
            if default_script.endswith("train_reranker.py")
            else "finetune/data_ft/embeddings"
        )
        default_model = (
            "cross-encoder/ms-marco-MiniLM-L-6-v2"
            if default_script.endswith("train_reranker.py")
            else "sentence-transformers/all-MiniLM-L6-v2"
        )
        train_script = str(hp.get("train_script", default_script))
        train_abs = train_script if os.path.isabs(train_script) else (
            os.path.join(product_root, train_script) if product_root else train_script
        )
        data_dir = str(hp.get("data_dir", default_data))
        if not os.path.isabs(data_dir) and product_root:
            data_dir = os.path.join(product_root, data_dir)
        # Must be absolute: wrapper chdirs into product_root before exec.
        ckpt = os.path.abspath(checkpoint_dir)
        default_py = (
            os.path.join(product_root, ".venv", "bin", "python")
            if product_root else "python3"
        )
        return {
            "mission_id": mission_id,
            "checkpoint_dir": ckpt,
            "product_root": product_root,
            "python_bin": str(hp.get("python_bin", default_py)),
            "train_script_abs": train_abs,
            "data_dir": data_dir,
            "base_model": str(hp.get("base_model", default_model)),
            "epochs": int(hp.get("epochs", 2 if default_script.endswith("reranker") else 3)),
            "batch_size": int(hp.get("batch_size", 16)),
            "learning_rate": float(hp.get("learning_rate", 2e-5)),
            "warmup_ratio": float(hp.get("warmup_ratio", 0.1)),
        }

    def _build_dpo_context(
        self,
        mission_id: str,
        plan: dict,
        checkpoint_dir: str,
        current_iteration: int = 0,
        warm_start_adapter: Optional[str] = None,
    ) -> dict:
        """Hyperparams for _DPO_REMOTE_WRAPPER."""
        recipe_key = plan.get("recipe") or "dpo"
        hp = _resolve_hyperparams(
            recipe_key, plan.get("hyperparameters", {}),
            warm_start_adapter=warm_start_adapter,
        )
        if warm_start_adapter:
            hp["adapter"] = warm_start_adapter
        finetune_dir = hp.get("finetune_dir", "/Users/kewang/finetune")
        load_pairs = hp.get("load_pairs", "") or ""
        return {
            "finetune_dir": finetune_dir,
            "python_bin": hp.get("python_bin", "/Users/kewang/finetune-env/bin/python"),
            "astra_pkg_root": _resolve_astra_pkg_root(hp),
            "use_case": _resolve_use_case(hp),
            "base_model": hp.get("base_model", "mlx-community/gemma-3-12b-it-4bit"),
            "adapter": hp.get("adapter", "adapters/grpo_v9_min/best"),
            "checkpoint_dir": checkpoint_dir,
            "num_layers": hp.get("num_layers", 8),
            "lora_rank": hp.get("lora_rank", 8),
            "lora_dropout": hp.get("lora_dropout", 0.0),
            "lora_scale": hp.get("lora_scale", 20.0),
            "k_collect": hp.get("k_collect", 8),
            "temp": hp.get("temp", 1.2),
            "max_tokens": hp.get("max_tokens", 96),
            "email_weight": hp.get("email_weight", 1),
            "routing_only_bool": "True" if hp.get("routing_only", True) else "False",
            "load_pairs_kwarg": (
                f'load_pairs="{load_pairs}",' if load_pairs else ""
            ),
            "epochs": hp.get("epochs", 1),
            "beta": hp.get("beta", 0.1),
            "learning_rate": hp.get("learning_rate", 3e-7),
            "max_grad_norm": hp.get("max_grad_norm", 1.0),
            "steps_per_eval": hp.get("steps_per_eval", 15),
            "eval_max_tokens": hp.get("eval_max_tokens", 256),
            "prompt_template": hp.get(
                "prompt_template", "backend/prompts/conductor_min.md"
            ),
            "save_pairs_path": os.path.join(
                finetune_dir, "logs", f"astra_{mission_id[:8]}_pairs.jsonl"
            ),
        }

    def _build_distill_context(
        self,
        mission_id: str,
        plan: dict,
        checkpoint_dir: str,
        current_iteration: int = 0,
        warm_start_adapter: Optional[str] = None,
    ) -> dict:
        recipe_key = plan.get("recipe") or "distill"
        hp = _resolve_hyperparams(
            recipe_key, plan.get("hyperparameters", {}),
            warm_start_adapter=warm_start_adapter,
        )
        if warm_start_adapter:
            hp["adapter"] = warm_start_adapter
            hp["no_adapter"] = False
        _cold = bool(hp.get("no_adapter", False)) or not hp.get("adapter")
        _focus = _resolve_focus_cases(hp)
        return {
            "finetune_dir": hp.get("finetune_dir", "/Users/kewang/finetune"),
            "python_bin": hp.get("python_bin", "/Users/kewang/finetune-env/bin/python"),
            "astra_pkg_root": _resolve_astra_pkg_root(hp),
            "use_case": _resolve_use_case(hp),
            "base_model": hp.get("base_model", "mlx-community/gemma-3-1b-it-4bit"),
            "adapter_kwarg": (
                "no_adapter=True," if _cold
                else f'adapter="{hp.get("adapter", "")}",'
            ),
            "checkpoint_dir": checkpoint_dir,
            "prompt_template": hp.get(
                "prompt_template", "backend/prompts/conductor_min.md"
            ),
            "teacher_model": hp.get("teacher_model", "gemma3:12b"),
            "ollama_url": hp.get("ollama_url", "http://127.0.0.1:11434"),
            "num_layers": hp.get("num_layers", -1),
            "lora_rank": hp.get("lora_rank", 16),
            "lora_scale": hp.get("lora_scale", 20.0),
            "lora_dropout": hp.get("lora_dropout", 0.0),
            "iters": hp.get("iters", 600),
            "batch_size": hp.get("batch_size", 1),
            "learning_rate": hp.get("learning_rate", 1e-5),
            "steps_per_eval": hp.get("steps_per_eval", 50),
            "save_every": hp.get("save_every", 50),
            "max_seq_len": hp.get("max_seq_len", 2048),
            "eval_max_tokens": hp.get("eval_max_tokens", 256),
            "mask_prompt_bool": "True" if hp.get("mask_prompt", True) else "False",
            "routing_only_bool": "True" if hp.get("routing_only", True) else "False",
            "focus_cases_kwarg": (
                f'focus_cases="{_focus}",' if _focus else ""
            ),
        }

    def _build_rft_context(
        self,
        mission_id: str,
        plan: dict,
        checkpoint_dir: str,
        current_iteration: int = 0,
        warm_start_adapter: Optional[str] = None,
    ) -> dict:
        recipe_key = plan.get("recipe") or "rft"
        hp = _resolve_hyperparams(
            recipe_key, plan.get("hyperparameters", {}),
            warm_start_adapter=warm_start_adapter,
        )
        if warm_start_adapter:
            hp["adapter"] = warm_start_adapter
            hp["no_adapter"] = False
        _cold = bool(hp.get("no_adapter", False)) or not hp.get("adapter")
        _focus = _resolve_focus_cases(hp)
        _force = _csv_cases(hp.get("force_gold_cases"))
        return {
            "finetune_dir": hp.get("finetune_dir", "/Users/kewang/finetune"),
            "python_bin": hp.get("python_bin", "/Users/kewang/finetune-env/bin/python"),
            "astra_pkg_root": _resolve_astra_pkg_root(hp),
            "use_case": _resolve_use_case(hp),
            "base_model": hp.get("base_model", "mlx-community/gemma-3-12b-it-4bit"),
            "adapter_kwarg": (
                "no_adapter=True," if _cold
                else f'adapter="{hp.get("adapter", "")}",'
            ),
            "checkpoint_dir": checkpoint_dir,
            "prompt_template": hp.get(
                "prompt_template", "backend/prompts/conductor_min.md"
            ),
            "k_samples": hp.get("k_samples", 4),
            "temp": hp.get("temp", 0.8),
            "max_tokens": hp.get("max_tokens", 256),
            "num_layers": hp.get("num_layers", 8),
            "lora_rank": hp.get("lora_rank", 8),
            "lora_scale": hp.get("lora_scale", 20.0),
            "lora_dropout": hp.get("lora_dropout", 0.0),
            "iters": hp.get("iters", 100),
            "batch_size": hp.get("batch_size", 1),
            "learning_rate": hp.get("learning_rate", 1e-5),
            "steps_per_eval": hp.get("steps_per_eval", 25),
            "save_every": hp.get("save_every", 25),
            "max_seq_len": hp.get("max_seq_len", 2048),
            "eval_max_tokens": hp.get("eval_max_tokens", 256),
            "mask_prompt_bool": "True" if hp.get("mask_prompt", True) else "False",
            "routing_only_bool": "True" if hp.get("routing_only", True) else "False",
            "focus_cases_kwarg": (
                f'focus_cases="{_focus}",' if _focus else ""
            ),
            "force_gold_cases_kwarg": (
                f'force_gold_cases="{_force}",' if _force else ""
            ),
        }

    def _build_opd_context(
        self,
        mission_id: str,
        plan: dict,
        checkpoint_dir: str,
        current_iteration: int = 0,
        warm_start_adapter: Optional[str] = None,
    ) -> dict:
        recipe_key = plan.get("recipe") or "opd"
        hp = _resolve_hyperparams(
            recipe_key, plan.get("hyperparameters", {}),
            warm_start_adapter=warm_start_adapter,
        )
        if warm_start_adapter:
            hp["adapter"] = warm_start_adapter
            hp["no_adapter"] = False
        _cold = bool(hp.get("no_adapter", False)) or not hp.get("adapter")
        _focus = _resolve_focus_cases(hp)
        return {
            "finetune_dir": hp.get("finetune_dir", "/Users/kewang/finetune"),
            "python_bin": hp.get("python_bin", "/Users/kewang/finetune-env/bin/python"),
            "astra_pkg_root": _resolve_astra_pkg_root(hp),
            "use_case": _resolve_use_case(hp),
            "base_model": hp.get("base_model", "mlx-community/gemma-3-12b-it-4bit"),
            "adapter_kwarg": (
                "no_adapter=True," if _cold
                else f'adapter="{hp.get("adapter", "")}",'
            ),
            "checkpoint_dir": checkpoint_dir,
            "prompt_template": hp.get(
                "prompt_template", "backend/prompts/conductor_min.md"
            ),
            "teacher_model": hp.get("teacher_model", "mlx-community/gemma-3-12b-it-4bit"),
            "num_generations": hp.get("num_generations", 2),
            "temp": hp.get("temp", 0.8),
            "max_tokens": hp.get("max_tokens", 96),
            "kl_coef": hp.get("kl_coef", 0.1),
            "outcome_bonus": hp.get("outcome_bonus", 0.0),
            "clip_epsilon": hp.get("clip_epsilon", 0.2),
            "adv_clip": hp.get("adv_clip", 5.0),
            "reward_schema": hp.get("reward_schema", "v1"),
            "num_layers": hp.get("num_layers", 8),
            "lora_rank": hp.get("lora_rank", 8),
            "lora_scale": hp.get("lora_scale", 20.0),
            "lora_dropout": hp.get("lora_dropout", 0.0),
            "iters": hp.get("iters", 100),
            "batch_size": hp.get("batch_size", 1),
            "learning_rate": hp.get("learning_rate", 1e-6),
            "steps_per_eval": hp.get("steps_per_eval", 25),
            "save_every": hp.get("save_every", 25),
            "max_seq_len": hp.get("max_seq_len", 2048),
            "eval_max_tokens": hp.get("eval_max_tokens", 256),
            "mask_prompt_bool": "True" if hp.get("mask_prompt", True) else "False",
            "routing_only_bool": "True" if hp.get("routing_only", True) else "False",
            "focus_cases_kwarg": (
                f'focus_cases="{_focus}",' if _focus else ""
            ),
        }

    def _build_grpo_context(
        self,
        mission_id: str,
        plan: dict,
        checkpoint_dir: str,
        current_iteration: int = 0,
        warm_start_adapter: Optional[str] = None,
    ) -> dict:
        """Hyperparams + optional DAPO/rollout kwargs for _GRPO_REMOTE_WRAPPER."""
        recipe_key = plan.get("recipe") or "grpo"
        hp = _resolve_hyperparams(
            recipe_key, plan.get("hyperparameters", {}),
            warm_start_adapter=warm_start_adapter,
        )
        if warm_start_adapter:
            hp["adapter"] = warm_start_adapter
        _facade_dapo: list[str] = []
        if hp.get("clip_epsilon_low") is not None:
            _facade_dapo.append(f'clip_epsilon_low={hp["clip_epsilon_low"]},')
        if hp.get("clip_epsilon_high") is not None:
            _facade_dapo.append(f'clip_epsilon_high={hp["clip_epsilon_high"]},')
        if int(hp.get("dynamic_sample_retries") or 0) > 0:
            _facade_dapo.append(
                f'dynamic_sample_retries={int(hp["dynamic_sample_retries"])},'
            )
        _rb = str(hp.get("rollout_backend") or "mlx").lower()
        if _rb and _rb != "mlx":
            _facade_dapo.append(f'rollout_backend="{_rb}",')
            if hp.get("rollout_base_url"):
                _facade_dapo.append(f'rollout_base_url="{hp["rollout_base_url"]}",')
            if hp.get("rollout_api_key"):
                _facade_dapo.append(f'rollout_api_key="{hp["rollout_api_key"]}",')
            if hp.get("rollout_model"):
                _facade_dapo.append(f'rollout_model="{hp["rollout_model"]}",')
        return {
            "finetune_dir": hp.get("finetune_dir", "/Users/kewang/finetune"),
            "python_bin": hp.get("python_bin", "/Users/kewang/finetune-env/bin/python"),
            "astra_pkg_root": _resolve_astra_pkg_root(hp),
            "use_case": _resolve_use_case(hp),
            "learner_id": hp.get("learner_id") or plan.get("learner_id") or "mlx_grpo_routing",
            "base_model": hp.get("base_model", "mlx-community/gemma-3-12b-it-4bit"),
            "adapter": hp.get("adapter", "adapters/grpo_v9_min/best"),
            "checkpoint_dir": checkpoint_dir,
            "num_layers": hp.get("num_layers", 8),
            "lora_rank": hp.get("lora_rank", 8),
            "lora_dropout": hp.get("lora_dropout", 0.0),
            "lora_scale": hp.get("lora_scale", 20.0),
            "iters": hp.get("iters", 100),
            "num_generations": hp.get("num_generations", 2),
            "temp": hp.get("temp", 1.2),
            "max_tokens": hp.get("max_tokens", 96),
            "eval_max_tokens": hp.get("eval_max_tokens", 256),
            "reward_schema": hp.get("reward_schema", "v1"),
            "email_weight": hp.get("email_weight", 0),
            "focus_weight": hp.get("focus_weight", 10),
            "max_zero_steps": hp.get("max_zero_steps", 3),
            "max_one_steps": hp.get("max_one_steps", 3),
            "learning_rate": hp.get("learning_rate", 1e-6),
            "clip_epsilon": hp.get("clip_epsilon", 0.1),
            "max_grad_norm": hp.get("max_grad_norm", 1.0),
            "steps_per_report": hp.get("steps_per_report", 1),
            "steps_per_eval": hp.get("steps_per_eval", 25),
            "save_every": hp.get("save_every", 25),
            "routing_only_bool": "True" if hp.get("routing_only", True) else "False",
            "prompt_template": hp.get(
                "prompt_template", "backend/prompts/conductor_min.md"
            ),
            "facade_dapo_kwargs": "\n    ".join(_facade_dapo),
        }

    def _build_prompt_context(
        self,
        mission_id: str,
        plan: dict,
        checkpoint_dir: str,
        current_iteration: int = 0,
        warm_start_adapter: Optional[str] = None,
    ) -> dict:
        """Context for _PROMPT_REMOTE_WRAPPER / heal _PROMPT_TEMPLATE."""
        recipe_key = plan.get("recipe") or "prompt"
        hp = _resolve_hyperparams(
            recipe_key, plan.get("hyperparameters", {}),
            warm_start_adapter=warm_start_adapter,
        )
        return {
            "mission_id": mission_id,
            "finetune_dir": hp.get("finetune_dir", "/Users/kewang/finetune"),
            "python_bin": hp.get("python_bin", "/Users/kewang/finetune-env/bin/python"),
            "astra_pkg_root": _resolve_astra_pkg_root(hp),
            "use_case": _resolve_use_case(hp),
            "base_model": hp.get("base_model", "mlx-community/gemma-3-4b-it-4bit"),
            "base_prompt": hp.get("base_prompt", "backend/prompts/conductor_gemma.md"),
            "variant_path": f"{checkpoint_dir}/conductor_variant.md",
            "frozen_now": hp.get("frozen_now", "2026-09-08T09:00:00"),
            "eval_max_tokens": hp.get("eval_max_tokens", 256),
        }

    def _build_agrpo_context(
        self,
        mission_id: str,
        plan: dict,
        checkpoint_dir: str,
        current_iteration: int = 0,
        warm_start_adapter: Optional[str] = None,
    ) -> dict:
        """Hyperparams for _AGRPO_REMOTE_WRAPPER / _AGRPO_TEMPLATE."""
        recipe_key = plan.get("recipe") or "agrpo"
        hp = _resolve_hyperparams(
            recipe_key, plan.get("hyperparameters", {}),
            warm_start_adapter=warm_start_adapter,
        )
        if warm_start_adapter:
            hp["adapter"] = warm_start_adapter
        _reward_mode = hp.get("reward_mode", "agent_gym")
        try:
            from astragym.resolve import infer_gym_id as _infer_gym_id
            _agrpo_gym = _infer_gym_id(
                task_type="agrpo",
                reward_mode=_reward_mode,
                gym_id=hp.get("gym_id") or plan.get("gym_id"),
                hyperparams=hp,
            )
        except ValueError:
            _agrpo_gym = (
                "e2e_stubs" if _reward_mode in ("e2e", "ensemble_e2e") else "agent_gym_text"
            )
        _cold = bool(hp.get("no_adapter", False)) or bool(hp.get("no_warmstart", False))
        adapter_val = (hp.get("adapter") or "").strip()
        if _cold or not adapter_val:
            adapter_kwarg = "no_warmstart=True,"
        else:
            adapter_kwarg = f'adapter="{adapter_val}",'
        _e2e = (hp.get("e2e_cases") or "").strip()
        e2e_cases_kwarg = f'e2e_cases="{_e2e}",' if _e2e else ""
        return {
            "mission_id": mission_id,
            "finetune_dir": hp.get("finetune_dir", "/Users/kewang/finetune"),
            "python_bin": hp.get("python_bin", "/Users/kewang/finetune-env/bin/python"),
            "astra_pkg_root": _resolve_astra_pkg_root(hp),
            "use_case": _resolve_use_case(hp),
            "learner_id": hp.get("learner_id") or plan.get("learner_id") or "mlx_grpo_traj",
            "gym_id": _agrpo_gym,
            "base_model": hp.get("base_model", "mlx-community/gemma-3-12b-it-4bit"),
            "adapter": adapter_val,
            "adapter_kwarg": adapter_kwarg,
            "checkpoint_dir": checkpoint_dir,
            "num_layers": hp.get("num_layers", 8),
            "lora_rank": hp.get("lora_rank", 8),
            "lora_dropout": hp.get("lora_dropout", 0.0),
            "lora_scale": hp.get("lora_scale", 20.0),
            "iters": hp.get("iters", 50),
            "num_generations": hp.get("num_generations", 2),
            "temp": hp.get("temp", 1.0),
            "max_tokens": hp.get("max_tokens", 64),
            "max_turns": hp.get("max_turns", 8),
            "tool_failure_prob": hp.get("tool_failure_prob", 0.0),
            "reward_mode": _reward_mode,
            "e2e_cases": _e2e,
            "e2e_cases_kwarg": e2e_cases_kwarg,
            "learning_rate": hp.get("learning_rate", 1e-6),
            "clip_epsilon": hp.get("clip_epsilon", 0.1),
            "max_grad_norm": hp.get("max_grad_norm", 1.0),
            "steps_per_report": hp.get("steps_per_report", 1),
            "steps_per_eval": hp.get("steps_per_eval", 10),
            "save_every": hp.get("save_every", 25),
        }

    def _build_sft_context(
        self,
        mission_id: str,
        plan: dict,
        checkpoint_dir: str,
        current_iteration: int = 0,
        warm_start_adapter: Optional[str] = None,
    ) -> dict:
        recipe_key = plan.get("recipe") or "sft"
        hp = _resolve_hyperparams(
            recipe_key, plan.get("hyperparameters", {}),
            warm_start_adapter=warm_start_adapter,
        )
        api_url = f"http://127.0.0.1:{settings.api_port}"
        base = {
            "mission_id": mission_id,
            "checkpoint_dir": checkpoint_dir,
            "api_url": api_url,
            "target_metric": json.dumps(plan.get("target_metric", {})),
        }
        raw_model = str(hp.get("base_model", ""))
        base_model = raw_model if "/" in raw_model else "mlx-community/gemma-3-12b-it-4bit"
        resume_path = (
            warm_start_adapter
            or hp.get("adapter")
            or hp.get("resume_adapter_file")
            or ""
        )
        resume_adapter_arg = _sft_resume_adapter_arg(str(resume_path))
        resume_adapter_kwarg = _sft_resume_adapter_kwarg(str(resume_path))
        return {
            "base_model": base_model,
            "dataset_path": hp.get("dataset_path", "data/datasets/train.jsonl"),
            "finetune_dir": hp.get("finetune_dir", "/Users/kewang/finetune"),
            "python_bin": hp.get("python_bin", "/Users/kewang/finetune-env/bin/python"),
            "val_split": hp.get("val_split", 0.1),
            "val_batches": hp.get("val_batches", 25),
            "max_seq_length": hp.get("max_seq_length", 2048),
            "preserve_reasoning": hp.get("preserve_reasoning", True),
            "lora_rank": hp.get("lora_rank", hp.get("lora_r", 16)),
            "lora_scale": hp.get("lora_scale", hp.get("lora_alpha", 32.0)),
            "lora_dropout": hp.get("lora_dropout", 0.05),
            "num_layers": hp.get("num_layers", 16),
            "batch_size": hp.get("batch_size", hp.get("per_device_train_batch_size", 4)),
            "learning_rate": hp.get("learning_rate", 0.0002),
            "iters": hp.get("iters", 300),
            "num_epochs": hp.get("num_epochs", hp.get("epochs", 3)),
            "save_every": hp.get("save_every", hp.get("save_steps", 50)),
            "steps_per_eval": hp.get("steps_per_eval", hp.get("eval_steps", 25)),
            "steps_per_report": hp.get("steps_per_report", 5),
            "save_steps": hp.get("save_steps", hp.get("save_every", 50)),
            "eval_steps": hp.get("eval_steps", hp.get("steps_per_eval", 25)),
            "lora_r": hp.get("lora_rank", hp.get("lora_r", 16)),
            "lora_alpha": hp.get("lora_scale", hp.get("lora_alpha", 32.0)),
            **hp,
            **base,
            # Must win over **hp (recipe may carry adapter path strings).
            "resume_adapter_arg": resume_adapter_arg,
            "resume_adapter_kwarg": resume_adapter_kwarg,
            "astra_pkg_root": _resolve_astra_pkg_root(hp),
            "use_case": _resolve_use_case(hp),
        }

    def _build_star_context(
        self,
        mission_id: str,
        plan: dict,
        checkpoint_dir: str,
        current_iteration: int = 0,
        warm_start_adapter: Optional[str] = None,
    ) -> dict:
        recipe_key = plan.get("recipe") or "star"
        hp = _resolve_hyperparams(
            recipe_key, plan.get("hyperparameters", {}),
            warm_start_adapter=warm_start_adapter,
        )
        api_url = f"http://127.0.0.1:{settings.api_port}"
        base = {
            "mission_id": mission_id,
            "checkpoint_dir": checkpoint_dir,
            "api_url": api_url,
            "target_metric": json.dumps(plan.get("target_metric", {})),
        }
        raw_model = str(hp.get("base_model", ""))
        base_model = raw_model if "/" in raw_model else "mlx-community/gemma-3-12b-it-4bit"
        _cold = bool(hp.get("no_adapter", False))
        adapter_val = hp.get("adapter", "") or warm_start_adapter or ""
        if _cold or not adapter_val:
            adapter_arg = '"--no-adapter",'
            adapter_kwarg = "no_adapter=True,"
        else:
            adapter_arg = f'"--adapter", "{adapter_val}",'
            adapter_kwarg = f'adapter="{adapter_val}",'
        routing_only = bool(hp.get("routing_only", True))

        return {
            "base_model": base_model,
            "adapter_arg": adapter_arg,
            "adapter_kwarg": adapter_kwarg,
            "routing_only": routing_only,
            "routing_only_bool": "True" if routing_only else "False",
            "finetune_dir": hp.get("finetune_dir", "/Users/kewang/finetune"),
            "python_bin": hp.get("python_bin", "/Users/kewang/finetune-env/bin/python"),
            "prompt_template": hp.get("prompt_template", "backend/prompts/conductor_min.md"),
            "k_samples": hp.get("k_samples", 8),
            "temp": hp.get("temp", 1.0),
            "rationalize_temp": hp.get("rationalize_temp", 0.8),
            "max_tokens": hp.get("max_tokens", 256),
            "eval_max_tokens": hp.get("eval_max_tokens", 256),
            "num_layers": hp.get("num_layers", 4),
            "lora_rank": hp.get("lora_rank", 8),
            "lora_scale": hp.get("lora_scale", 5.0),
            "lora_dropout": hp.get("lora_dropout", 0.1),
            "iters": hp.get("iters", 100),
            "batch_size": hp.get("batch_size", 2),
            "learning_rate": hp.get("learning_rate", 0.0001),
            "steps_per_eval": hp.get("steps_per_eval", 25),
            "save_every": hp.get("save_every", 25),
            "max_seq_len": hp.get("max_seq_len", 2048),
            "val_split": hp.get("val_split", 0.15),
            "reward_schema": hp.get("reward_schema", "v1"),
            "routing_only": "True" if hp.get("routing_only", True) else "False",
            **hp,
            **base,
            "astra_pkg_root": _resolve_astra_pkg_root(hp),
            "use_case": _resolve_use_case(hp),
        }

    def _build_user_prompt(
        self, task_type: str, mission_id: str, plan: dict, checkpoint_dir: str, current_iteration: int = 0,
        warm_start_adapter: Optional[str] = None,
    ) -> str:
        effective_task_type = plan.get("active_task_type", "sft") if task_type == "post-training" else task_type
        recipe_key = plan.get("recipe") or (plan.get("env_id", "") if effective_task_type == "rl" else effective_task_type)
        _raw_algo = plan.get("algorithm", "PPO" if effective_task_type == "rl" else "")
        _plan_algo = canonicalize_algorithm(_raw_algo) or _raw_algo
        hp = _resolve_hyperparams(
            recipe_key, plan.get("hyperparameters", {}), algorithm=_plan_algo,
            warm_start_adapter=warm_start_adapter,
        )
        api_url = f"http://127.0.0.1:{settings.api_port}"
        base = {
            "mission_id": mission_id,
            "checkpoint_dir": checkpoint_dir,
            "api_url": api_url,
            "target_metric": json.dumps(plan.get("target_metric", {})),
        }
        if effective_task_type == "rlcd":
            return _format_rlcd_wrapper(hp, checkpoint_dir)
        if effective_task_type == "rl":
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
            elif env_id in ("Game2048-v0", "2048"):
                env_setup = _GAME2048_SETUP.format(project_root=_project_root)
            elif "minatar" in env_id.lower() or env_id.lower() in ("asterix", "freeway", "seaquest"):
                env_setup = _MINATAR_SETUP.format(project_root=_project_root)
            elif env_id in ("GridPacMan-v0", "PacMan-v0") or "pacman" in env_id.lower() or "pac-man" in env_id.lower():
                env_setup = _GRIDPACMAN_SETUP.format(project_root=_project_root)
            elif env_id in (
                "MultiTurnAgentGym-v0", "AgentGym-v0", "agent-gym", "MultiTurnAgentGymText-v0",
            ):
                env_setup = _AGENT_GYM_SETUP.format(project_root=_project_root)
            else:
                env_setup = ""
            hp = dict(hp)  # copy so we don't mutate the plan's hyperparameters dict
            policy_kwargs = hp.pop("policy_kwargs", None)
            # Build env_kwargs_str: ", key=value, ..." for astragym.make() call
            env_kwargs = _resolve_env_kwargs(env_id, plan.get("env_kwargs"))
            if env_kwargs:
                env_kwargs_str = ", " + ", ".join(f"{k}={v!r}" for k, v in env_kwargs.items())
            else:
                env_kwargs_str = ""
            algorithm = canonicalize_algorithm(plan.get("algorithm", "PPO")) or "PPO"
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
            if trainer_type in ("lookahead_dqn", "lookahead_ppo", "lookahead_a2c") or algorithm.upper() in ("LOOKAHEAD_DQN", "LOOKAHEAD-DQN"):
                if env_id in ("Game2048-v0", "2048"):
                    tm_score = next(iter(tm.values()), 2048) if tm else 2048
                    lr = hp.get("learning_rate", 0.0003)
                    lookahead_2048_ctx = {
                        "env_id": env_id,
                        "env_setup": env_setup,
                        "target_score": tm_score,
                        "hyperparameters": json.dumps(hp, indent=2),
                        "lr": lr,
                        "total_timesteps": hp.get("total_timesteps", 500000),
                        "gamma": hp.get("gamma", 0.99),
                        "replay_buffer_size": hp.get("replay_buffer_size", 50000),
                        "batch_size": hp.get("batch_size", 64),
                        "target_update_interval": hp.get("target_update_interval", 1000),
                        "epsilon_min": hp.get("epsilon_min", 0.02),
                        "epsilon_decay": hp.get("epsilon_decay", 0.9997),
                        "ac_telemetry_interval": hp.get("ac_telemetry_interval", 25),
                        "current_iteration": current_iteration,
                        **base,
                    }
                    return _GAME2048_LOOKAHEAD_DQN_CONTRACT.format(**lookahead_2048_ctx)
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
            algo_upper = canonicalize_algorithm(algorithm).upper() or "PPO"
            valid_keys = _VALID_ALGO_KEYS.get(algo_upper, _VALID_ALGO_KEYS["PPO"])
            valid_keys_var = f"_VALID_{algo_upper}_KEYS"
            valid_keys_set = "{" + ", ".join(f'"{k}"' for k in sorted(valid_keys)) + "}"
            is_agent_gym = env_id in (
                "MultiTurnAgentGym-v0", "AgentGym-v0", "agent-gym", "MultiTurnAgentGymText-v0",
            )
            # AgentGym goal metric is task_success — use the real target for early stop.
            if is_agent_gym and tm_name == "task_success":
                target_reward = tm_value
            from astragym.resolve import infer_gym_id as _infer_gym_id
            from astrarl.resolve import infer_learner_id as _infer_learner_id
            try:
                _gym_id = _infer_gym_id(
                    task_type="rl",
                    env_id=env_id,
                    gym_id=hp.get("gym_id") or plan.get("gym_id"),
                    hyperparams=hp,
                )
            except ValueError:
                _gym_id = f"gymnasium:{env_id}"
            try:
                _learner_id = _infer_learner_id(
                    task_type="rl",
                    algorithm=algorithm,
                    learner_id=hp.get("learner_id") or plan.get("learner_id"),
                    hyperparams=hp,
                )
            except ValueError:
                _learner_id = f"sb3_{algorithm.lower()}"
            ctx = {
                "algorithm": algorithm,
                "valid_keys_var": valid_keys_var,
                "valid_keys_set": valid_keys_set,
                "env_id": env_id,
                "gym_id": _gym_id,
                "learner_id": _learner_id,
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
            if is_agent_gym:
                return _AGENT_GYM_RL_TEMPLATE.format(**ctx)
            return _RL_TEMPLATE.format(**ctx)
        if effective_task_type == "sft":
            ctx = self._build_sft_context(
                mission_id, plan, checkpoint_dir, current_iteration, warm_start_adapter
            )
            return _SFT_TEMPLATE.format(**ctx)
        if effective_task_type == "mlx_lora":
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
        if effective_task_type == "dpo":
            ctx = {
                **self._build_dpo_context(
                    mission_id, plan, checkpoint_dir, current_iteration,
                    warm_start_adapter=warm_start_adapter,
                ),
                **base,
            }
            return _DPO_TEMPLATE.format(**ctx)
        if effective_task_type == "grpo":
            # Heal/docs prompt only — live codegen uses _GRPO_REMOTE_WRAPPER via
            # _build_grpo_context (same defaults). Do not **hp alone: incomplete
            # smoke recipes KeyError on reward_schema / max_zero_steps / etc.
            ctx = {
                **self._build_grpo_context(
                    mission_id, plan, checkpoint_dir, current_iteration,
                    warm_start_adapter=warm_start_adapter,
                ),
                "clip_epsilon_low": hp.get("clip_epsilon_low", ""),
                "clip_epsilon_high": hp.get("clip_epsilon_high", ""),
                "dynamic_sample_retries": hp.get("dynamic_sample_retries", 0),
                "rollout_backend": hp.get("rollout_backend", "mlx"),
                "rollout_base_url": hp.get("rollout_base_url", ""),
                "routing_only": hp.get("routing_only", True),
                "gym_id": hp.get("gym_id") or plan.get("gym_id") or "routing_prompt",
                **base,
            }
            return _GRPO_TEMPLATE.format(**ctx)
        if effective_task_type == "agrpo":
            ctx = {
                **self._build_agrpo_context(
                    mission_id, plan, checkpoint_dir, current_iteration,
                    warm_start_adapter=warm_start_adapter,
                ),
                **base,
            }
            return _AGRPO_TEMPLATE.format(**ctx)
        if effective_task_type == "distill":
            ctx = {
                **self._build_distill_context(
                    mission_id, plan, checkpoint_dir, current_iteration,
                    warm_start_adapter=warm_start_adapter,
                ),
                **base,
            }
            return _DISTILL_TEMPLATE.format(**ctx)
        if effective_task_type == "rft":
            ctx = {
                **self._build_rft_context(
                    mission_id, plan, checkpoint_dir, current_iteration,
                    warm_start_adapter=warm_start_adapter,
                ),
                **base,
            }
            return _RFT_TEMPLATE.format(**ctx)
        if effective_task_type == "opd":
            ctx = {
                **self._build_opd_context(
                    mission_id, plan, checkpoint_dir, current_iteration,
                    warm_start_adapter=warm_start_adapter,
                ),
                **base,
            }
            return _OPD_TEMPLATE.format(**ctx)
        if effective_task_type == "star":
            ctx = {
                **self._build_star_context(
                    mission_id, plan, checkpoint_dir, current_iteration,
                    warm_start_adapter=warm_start_adapter,
                ),
                **base,
            }
            return _STAR_TEMPLATE.format(**ctx)
        if effective_task_type == "prompt":
            ctx = {
                **self._build_prompt_context(
                    mission_id, plan, checkpoint_dir, current_iteration,
                    warm_start_adapter=warm_start_adapter,
                ),
                **base,
            }
            return _PROMPT_TEMPLATE.format(**ctx)
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

        _gym_id = env_id if str(env_id).startswith("gymnasium:") else f"gymnasium:{env_id}"
        curriculum_block = (
            f'_CURRICULUM_PHASES = {phases_repr}\n'
            f'for _ph_idx, _ph in enumerate(_CURRICULUM_PHASES):\n'
            f'    _ph_gym = astragym.make("{_gym_id}"{kw_str}{grid_kw_str})\n'
            f'    _ph_env = _ph_gym.raw_env\n'
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
    def _ensure_rl_gym_make(
        code: str,
        env_id: str,
        env_kwargs_str: str = "",
        gym_id: Optional[str] = None,
    ) -> str:
        """Force AstraGym env construction so the LLM cannot omit it or swap games.

        Real incident: GridPacMan-v0 mission 9b49aa78 crashed ``NameError: env``,
        then the healer inserted ``gym.make("Minatar-SpaceInvaders-v0")``.
        Also rewrites legacy ``gym.make`` to ``astragym.make``.
        """
        import re
        if not env_id or not isinstance(code, str):
            return code
        gid = gym_id or (
            env_id if str(env_id).startswith("gymnasium:") else f"gymnasium:{env_id}"
        )
        astra_block = (
            f'_astra_gym = astragym.make("{gid}"{env_kwargs_str})\n'
            f'env = _astra_gym.raw_env'
        )
        astra_pat = re.compile(
            r"_astra_gym\s*=\s*astragym\.make\s*\([^)]*\)\s*\n\s*"
            r"env\s*=\s*_astra_gym\.raw_env"
        )
        gym_pat = re.compile(r"(?m)^[ \t]*env\s*=\s*gym\.make\s*\([^)]*\)[ \t]*\n?")
        if astra_pat.search(code):
            code = astra_pat.sub(astra_block, code, count=1)
            return gym_pat.sub("", code)  # drop leftover legacy gym.make
        if gym_pat.search(code):
            return gym_pat.sub(astra_block + "\n", code, count=1)
        ctor = re.search(
            r"\n(?=model\s*=\s*(?:PPO|DQN|A2C|SAC|TD3)\s*\("
            r"|_learner\s*=\s*astrarl\.make\s*\("
            r"|model\s*=\s*_learner\.model"
            r"|BUFFER\s*=\s*collections\.deque"
            r"|while\s+total_steps\s*<)",
            code,
        )
        if ctor:
            return code[: ctor.start()] + f"\n{astra_block}\n" + code[ctor.start() :]
        return code

    @staticmethod
    def _patch_rl_imports(code: str) -> str:
        """Inject any SB3 / AstraGym imports the LLM forgot; drop unused gymnasium."""
        import re
        # Replace `import stable_baselines3 as sb3` + `sb3.XXX` with direct imports
        if "import stable_baselines3 as sb3" in code:
            code = re.sub(r'\bsb3\.(PPO|SAC|A2C|DQN|TD3)\b', r'\1', code)
            code = re.sub(r'^\s*import stable_baselines3 as sb3\s*\n?', '', code, flags=re.MULTILINE)

        # Drop unused gymnasium alias (AstraGym path does not need it).
        if re.search(r"(?m)^import gymnasium as gym\s*$", code):
            body = re.sub(r"(?m)^import gymnasium as gym\s*\n?", "", code)
            if not re.search(r"\bgym\.", body):
                code = body

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
        if "astragym.make" in code and "import astragym" not in import_lines:
            to_inject.append("import astragym")
        if "astrarl.make" in code and "import astrarl" not in import_lines:
            to_inject.append("import astrarl")
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
        # Relative only. The prompt already asks for an absolute checkpoint_dir,
        # and that absolute path ends in data/missions/<uuid>/checkpoints. A bare
        # \bdata/missions match rewrites that tail and prefixes the repo root a
        # second time, so weights land outside the gitignored data/ directory.
        pattern = (
            r'(?<![A-Za-z0-9_./])'
            r'(?:\./)?'
            r'data/missions/[0-9a-f-]{36}/checkpoints\b'
        )
        fixed = re.sub(pattern, checkpoint_dir, code)
        if fixed != code:
            logger.info("CodeGenerator: replaced relative checkpoint paths with absolute: %s", checkpoint_dir)
        return fixed

    @staticmethod
    def _fix_execv_unpacking(code: str) -> str:
        """Rewrite os.execv(*argv) to os.execv(argv[0], argv).

        Real incident: the dpo/grpo templates require os.execv(path, argv_list)
        (2 arguments — see _DPO_TEMPLATE/_GRPO_TEMPLATE) but the LLM occasionally
        writes os.execv(*argv) instead, star-unpacking a ~40-element argv list
        into that many positional arguments. os.execv() only accepts 2 — this
        raises TypeError immediately, before the Astra train CLI ever
        runs. The wrapper process then exits near-instantly; SandboxManager
        sees it as "done" and the evaluator falls back to re-scoring the
        previous iteration's stale checkpoint — silently wasting a full
        iteration and feeding a duplicate score into the pivot engine's
        plateau detection. Confirmed live across at least 4 iterations in one
        mission's history before being caught. Same auto-fix pattern as
        _fix_checkpoint_paths/_patch_undefined_logger — repair the known-bad
        pattern rather than rejecting/retrying generation.
        """
        import re
        pattern = r'os\.execv\(\*(\w+)\)'
        fixed = re.sub(pattern, r'os.execv(\1[0], \1)', code)
        if fixed != code:
            logger.info("CodeGenerator: fixed os.execv(*argv) star-unpacking to os.execv(argv[0], argv)")
        return fixed

    @staticmethod
    def _pin_finetune_exec_python(code: str) -> str:
        """Force os.execv onto the absolute finetune interpreter named in the script.

        Real incident (bdd6e532): the code model emitted os.execv(sys.executable, argv)
        for an RFT wrapper. The SSH sandbox launches train.py with plain python3, so
        sys.executable has no mlx and the train CLI dies with ModuleNotFoundError. The
        error healer then treated that as a missing import and rewrote the wrapper into
        a gym/PPO script, exhausting sandbox retries. argv[0] already names
        finetune-env/bin/python; exec that path, not the launcher.
        """
        import re
        # Match Astra -m train CLIs (and legacy *_train.py strings if an old
        # wrapper is still on disk).
        markers = (
            "astrarl.cli", "astratune.cli",
            "rft_train.py", "distill_train.py", "opd_train.py", "rlcd_train.py",
            "grpo_train.py", "dpo_train.py", "sft_train.py", "star_train.py",
        )
        if not code or not any(name in code for name in markers):
            return code
        match = re.search(r'["\'](/[^"\']+/python(?:\d+(?:\.\d+)*)?)["\']', code)
        if not match:
            return code
        py = match.group(1)
        fixed = re.sub(
            r'os\.execv\(\s*sys\.executable\s*,',
            f'os.execv("{py}",',
            code,
            count=1,
        )
        if fixed != code:
            logger.info("CodeGenerator: pinned os.execv to %s (was sys.executable)", py)
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
