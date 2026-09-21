# ASTRA: Product Requirements Document (PRD)

**Project Name:** ASTRA (**A**utonomous **S**trategic **Tr**aining **A**gent)  
**Status:** Phase 75 complete  
**Target:** Autonomous Machine Learning Orchestration

---

## 1. Executive Summary
ASTRA is an autonomous agent designed to manage the end-to-end lifecycle of Reinforcement Learning (RL) and Machine Learning (ML) training. It leverages lessons from high-performance Snake, Tetris, 2048, the complete 5-game MinAtar arcade suite (Breakout, Space Invaders, Asteroids, Freeway, Seaquest), and MultiTurnAgentGym AI implementations to automate curriculum shifts, reward shaping, tournament evaluation, and competitive benchmarking.


## 2. Problem Statement
Manual ML training is repetitive and error-prone. Engineers often spend hours:
- Monitoring training logs for convergence or plateaus.
- Manually adjusting grid sizes or difficulty levels (Curriculum Learning).
- Comparing different algorithms (PPO vs DQN) across multiple seeds.
- Writing boilerplate code for model saving and performance plotting.

## 3. Goals & Objectives
- **Autonomy**: Reduce human intervention in the training loop by 80%.
- **Optimization**: Discover better hyperparameter and reward configurations through automated experimentation.
- **Observability**: Provide high-fidelity insights into *why* a model is performing via feature map analysis.
- **Portability**: Create a system that can be easily plugged into different environments (Snake, Tetris, 2048, MinAtar Breakout, Finance, etc.).

## 4. Key Features

### 4.1. Autonomous Curriculum Manager
- Automatically shifts training phases based on success metrics.
- Scales environment complexity (e.g., grid size) without manual restarts.

### 4.2. Multi-Agent Orchestrator
- Dispatches parallel training jobs for different algorithms (PPO, DQN, A2C).
- Implements "Survival of the Fittest" where underperforming experiments are pruned.

### 4.3. Dynamic Reward Evolver
- Tests variations of reward shaping automatically when structural pivots (HP tune, arch change, algo switch) are exhausted.
- Supported game environments expose configurable reward shaping parameters:
  - **Snake-v0**: `food_reward`, `death_penalty`, `survival_bonus`, `distance_weight`.
  - **Game2048-v0**: `merge_multiplier`, `empty_tile_bonus`, `corner_bonus`.
  - **MinAtar Suite** (`MinAtar-Breakout-v0`, `MinAtar-SpaceInvaders-v0`, `MinAtar-Asteroids-v0`, `MinAtar-Freeway-v0`, `MinAtar-Seaquest-v0`): `brick_reward`, `alien_kill_reward`, `asteroid_hit_reward`, `cross_reward`, `enemy_kill_reward`, `diver_rescue_reward`, `death_penalty`.
  At escalation level 3 the pivot agent proposes `env_kwargs` overrides from the **current** env's allowlist (`env_reward_guidance(env_id)` — Seaquest uses diver/oxygen keys, not Snake `food_reward`). These flow through pivot → plan → code generator → `gym.make()` automatically.
- **Algorithm-locked escalation**: when a mission goal explicitly names an algorithm (e.g. "Train a Snake-v0 DQN agent", including Phase 68 canonical recipe-dispatch goals), ASTRA never switches away from that trainer. Escalation level 2 remaps to reward shaping instead. Aliases count as the same trainer (`SB3 PPO` is PPO), so a PPO-titled run cannot slip the lock by renaming.
  - **Unnamed RL missions**: a goal that only says "RL agent" (no PPO/DQN/A2C/…) is unlocked. The first plan is seeded from the env recipe's `algorithm` and matching hyperparameters when present (Seaquest → DQN at `lr=2.5e-4`, not the PPO prior). At escalation level 2, a pivot that does not actually change trainer (`SB3 PPO` → `PPO`) is rewritten to a different discrete SB3 algorithm **once**; at level 3+ the next untried canonical trainer is forced (PPO → A2C). A real switch resets leftover `env_kwargs`, seeds matching-recipe or SB3-default HPs, restarts the stall clock, and does **not** revert to the previous algorithm's best checkpoint. Custom Tetris/2048 trainers are left alone.
- **Escalation persistence**: the consecutive-failed-pivot counter (`pivot_escalation_count`) is saved to the DB after every pivot and restored on server restart, so long-running missions correctly escalate even across process restarts or crashes.
- **Convergence stop**: a mission that has maxed out escalation and still not set a new best for many evaluated iterations is recognized as converged below target — it is stopped and marked `stalled` (terminal) with `error_log` `converged_below_target:…` and its best checkpoint kept. Resume is not offered; start a new mission. User cancel, gate reject, max retries, and a `PATCH` that asks for `stalled` without that prefix are `failed` with a reason (Resume is still available). Process shutdown without a user-cancel flag stays `pending` so the mission can resume. Targets that exceed a recipe's declared empirical ceiling are rejected when the mission is created.

### 4.4. The Registry & Benchmark Suite
- Persistent storage for models, weights, and training metadata.
- Automated "Tournament Mode" to compare new models against the current "Champion."

### 4.5. Smart Visualizer
- Automatically captures video of "Breakthrough Moments."
- Generates CNN activation maps and feature plots.
- **Unified Live Game Players**: Real-time interactive game HUD canvas players with live WebSocket streaming for Snake-v0, Tetris-v0, Game2048-v0, and the complete 5-game MinAtar Suite (Breakout, Space Invaders, Asteroids, Freeway, Seaquest), featuring unified dark slate telemetry cards, live connection indicators, pre-rendered initial board states, and game-tailored speed controls.

### 4.6. Autonomous Iteration Loop
- Continuous "Plan-Train-Evaluate-Refine" cycle.
- System autonomously restarts or pivots training strategies if goals are not met within predicted timelines.
- Learns from failed iterations to adjust future hyperparameters or curriculum steps.

### 4.7. Autonomous Code Implementation
- ASTRA can generate necessary training scripts across various paradigms:
  - **RL**: Environment wrappers, reward functions, and policy networks.
  - **SFT**: Tokenization logic, prompt templates, and supervised loss functions.
  - **ML**: Feature engineering, data loaders, and standard classification/regression architectures.
- Autonomously fixes bugs in training code by analyzing error logs and stack traces.

### 4.8. Secure Sandboxed Execution
- All generated code runs in an isolated sandbox to protect the host system. The sandbox type depends on the hardware target: **Docker/Podman** (cloud or CPU-only workloads) or a **restricted host subprocess** (Apple Silicon, where Metal GPU is not accessible inside Docker).
- Resource constraints (CPU/GPU/RAM) are enforced at the sandbox level to prevent runaway processes.

### 4.9. Training Recipes & Crystallization
- **Recipe Generation**: Upon achieving a goal, ASTRA "crystallizes" the successful strategy into a **Training Recipe** (a package containing the optimized hyperparameters, reward/loss logic, curriculum phases, and model architecture). Fine-tuning task types (`dpo`/`grpo`) are excluded — their training dispatch always loads a fixed canonical recipe, so a per-mission crystallized recipe would be dead weight.
- **Strategy Sharing**: Recipes are stored in a global library, allowing ASTRA to "warm-start" new, similar goals by retrieving and adapting existing recipes.
- **Recipe Evolution**: ASTRA can treat a recipe as a "DNA" strand, mutating and improving it across different training runs to discover universal "Golden Recipes" for specific domains (e.g., "The Golden Snake Recipe").

### 4.10. Predefined "Golden" Recipes
- ASTRA ships with a set of **Predefined Base Recipes** derived from proven, high-performance training runs (e.g., the workspace's existing Snake, Tetris, 2048, and MinAtar Breakout models).
- These recipes serve as the "Initial Knowledge" of the system, allowing users to achieve expert-level results on day one for common tasks.

### 4.11. Crash-Safe Mission Persistence
- **Stateful Resumption**: If the system is interrupted (crash, power loss, restart), ASTRA automatically recovers the state of all "In-Progress" missions.
- **Checkpoint-Aware Training**: All trainers are required to save frequent checkpoints (weights + optimizer state) to the **File Store** (`data/` volume) at regular intervals (target: every 2–5 minutes of wall-clock training time), ensuring no more than a few minutes of progress is lost. Checkpoint paths are registered as metadata in the Model Registry for discovery.
- **Atomic State Transitions**: Loop transitions (e.g., from Training to Eval) are logged as atomic events in the Mission Store to prevent duplicate or inconsistent execution upon resume.

### 4.12. Resilience & Rigor (Harness Principles)
- **Skeptical Peer Review**: Implements a GAN-like architecture where a "Safety Critic" must audit and approve plans before execution, driving higher quality through iterative internal critique.
- **Artifact-Driven Context**: Uses a structured Mission Manifest to maintain a "Single Source of Truth," allowing the agent to reset its context window and avoid the performance degradation associated with long conversation histories.
- **Multi-Dimensional Validation**: Evaluates success using a complex rubric (Validation Contract) rather than a single metric, ensuring model health and robustness.

### 4.13. Lookahead-Augmented DQN for 2048
- Equips DQN with 1-step successor state evaluation via `get_next_states()` on `Game2048-v0`, breaking past blind trial-and-error exploration limits to reach 4096+ tile values.
- Employs a specialized `Game2048ValueNet` architecture with target network stabilization.

### 4.14. Live Policy Audit & Explainability Inspector
- Streams real-time action probabilities, Q-values, and Shannon policy entropy over WebSockets to the Mission HUD.
- Features confidence indicators and high certainty vs. high exploration categorization, providing immediate visibility into model decision dynamics.

### 4.15. MinAtar Arcade Benchmark Suite
- Expands beyond Breakout to include **Space Invaders** (`MinAtar-SpaceInvaders-v0`) and **Asteroids** (`MinAtar-Asteroids-v0`) with 10x10 pure Python/NumPy execution (>50k steps/sec).
- High-fidelity symbolic arcade physics, projectile collision simulations, and dedicated HUD palettes.

### 4.16. Model Registry & Tournament Leaderboard
- Head-to-head multi-model tournaments across fixed deterministic seeds (`2000 + ep`).
- Computes mean, std, min, max, per-seed score arrays, and tie-split win rates.
- Automatic champion detection and crown 👑 promotion in the Model Registry.

### 4.17. Recipe Library, Lineage DAG Visualizer & Unified Mission Dispatch
- Searchable gallery of canonical training blueprints across RL, Fine-tuning, and ML paradigms.
- Lineage tree DAG visualizer tracking genetic evolution, hyperparameter mutations, and generational wins.
- **Converged Mission Creation & Dispatch**: Unified pipeline (`POST /missions` and `POST /recipes/{name}/dispatch`) supporting first-class recipe seeding, target metric overrides, and automated canonical goal generation (`Train a <env_id> <algo> agent to achieve <target_value> <metric_name>`) without mutating canonical recipe YAML files.

### 4.18. SFT Post-Training with Strict Held-Out Splitting & Reasoning Preservation
- End-to-end Supervised Fine-Tuning orchestrator via `SFTTrainer` supporting HuggingFace Transformers, PEFT (LoRA/QLoRA), and TRL.
- **Strict Held-Out Validation**: Enforces deterministic held-out dataset splitting (`val_split`, fixed seed 42) to eliminate in-sample evaluation overfitting and data leakage.
- **Chain-of-Thought (CoT) Preservation**: Automatically extracts, preserves, and formats `<think>...</think>` internal reasoning traces in multi-turn dialogues and completion records.
- **Comprehensive Telemetry & Checkpointing**: Emits live `train_loss`, `eval_loss`, and `perplexity` (`exp(eval_loss)`) metrics to the dashboard, with peak checkpoint tracking in `checkpoints/best`.
- **AST-Guarded Code Generation & Self-Healing**: Resilient code generation and error analyzer self-healing that validate script syntax via Python's `ast` parser and fall back to canonical execution scripts against autoregressive coder degeneration.

### 4.19. Remote SFT Training via MLX on Apple Silicon Cluster
- Apple Silicon Mac Mini compute offload utilizing standalone MLX training runner (`sft_train.py`) without requiring git checkout on remote worker nodes.
- **Zero-Orphan Process Execution**: `CodeGenerator` wraps remote training commands in an `os.execv` wrapper replacing the wrapper process image in place, guaranteeing clean PID tracking and signal termination.
- **Dynamic Validation Batch-Clamping**: Automatically adjusts validation batch sizes so `effective_batch_size <= min(train_size, val_size)`, preventing MLX batch underflow exceptions on small datasets.
- **Live SSH Telemetry Tailing**: Real-time parsing of remote training logs over SSH, streaming loss curves directly to the dashboard HUD.

### 4.20. SFT-to-DPO Multi-Stage Pipeline Chaining & LoRA Auto-Detection
- Multi-stage post-training pipeline connecting SFT syntax learning directly to DPO preference alignment across discrete mission stages.
- **LoRA Auto-Detection**: Standalone trainers dynamically detect LoRA rank, scale, dropout, and layer count from `adapter_config.json` inside warm-start adapter checkpoints.
- **Accelerated Preference Optimization**: Support for `--load-pairs` to bypass redundant on-policy pair generation and directly optimize against verified contrastive datasets.
- **NLP Model Registry Tournament Arena**: Head-to-head language model evaluation with tournament leaderboards and automatic champion crowning.

### 4.21. Pure Routing Post-Training & Recipe Hyperparameter Enforcement
- **Pure Routing Demonstrations**: Specialized dataset architecture (`data_routing`) standardizing on compact `conductor_min.md` prompt headers (~175 tokens) and verified routing schema plans, preventing prompt overflow and token truncation.
- **Strict Recipe Locking**: Hardened code generation and hyperparameter resolution that strictly lock hardware-critical settings (batch size, layer count, dataset path) to recipe specifications, preventing planner hallucinations from triggering Apple Silicon Metal OOM crashes.
- **Lower-is-Better Metric Visualizations**: Comprehensive telemetry and dashboard support for loss minimization goals, displaying accurate remaining metric gaps and progress percentages toward convergence.

### 4.22. Completed Missions Archive & Row-Based Operational Board
- **Dedicated Completed Missions Archive (`/completed`)**: A dedicated historical repository modeled after the Recipe Library, featuring domain category filtering, full-text search, sortable metrics, and verifiable requirement manifests (`score >= target: PASSED`, `clean sandbox exit: PASSED`, `checkpoint saved: PASSED`).
- **Operational Board Row Layout**: The Command Center separates historical completed runs from active workflows, presenting operational tasks in full-width horizontal status rows (`Running / Active`, `Stalled / Paused`, `Failed`) with responsive mission cards, keeping operator focus strictly on in-flight and attention-needed tasks.

## 5. User Experience & Autonomy Model


### 5.1. The "Goal-First" Interface
The user provides a high-level goal and a success threshold.
- **Example**: `{"task": "SFT", "base_model": "Llama-3.1-8B", "dataset": "customer_logs.jsonl", "target_metric": {"eval_loss": 0.05}}`
- **Example**: `{"task": "RL", "env": "Snake-v0", "target_metric": {"mean_reward": 150}}`

### 5.2. Fully Autonomous Execution
Once the goal is set, ASTRA enters a recursive loop:
1. **Plan**: LLM designs the training trajectory.
2. **Implement**: Generates scripts and configs.
3. **Execute**: Runs in sandbox.
4. **Evaluate**: Compares current metrics against the `target_metric`.
5. **Iterate**: If the goal isn't met, ASTRA self-corrects and repeats from Step 1.

### 5.3. Security & Approval Gates
To balance autonomy with safety, ASTRA implements a **Graduated Autonomy** system:
- **Mandatory Approval**: Irreversible or high-risk actions (e.g., accessing external APIs, deploying to production) require user confirmation.
- **Configurable Gates**: Users can set "Approval Flags" for:
  - **Code Generation**: Reviewing scripts before execution.
  - **Resource Usage**: Approving training runs that exceed a specific cost/time budget.
- **Silent Mode**: In **Supervised** mode only, once a specific sub-task strategy is "trusted" (high success rate in previous iterations), ASTRA can bypass approval gates for that sub-task automatically. Silent Mode does not apply in Guided mode (all gates remain active) and is redundant in Full Autonomy mode (all gates are already suppressed by user choice). See DESIGN §4.2 for the full gate-priority model.
