# ASTRA: Design Document

**Architecture Version:** 1.0.0  
**Core Stack:** Python, PyTorch, SQLAlchemy (Registry), FastAPI (Backend API), Next.js 15 (Frontend Dashboard)

---

## 1. System Overview
ASTRA is designed as a modular system where a **Lead Agent** orchestrates several **Specialist Agents**, served via a high-performance **FastAPI** backend and a **Next.js** professional dashboard.

```
                    +-----------------------+
                    |    Next.js Web UI     |
                    +-----------+-----------+
                                |
                                | (HTTP/REST)
                                v
+-------------------+       +-----------+-----------+       +-----------------------+
| Live Training HUD |<-WS-->|  FastAPI Orchestrator | <---> |    Memory/Registry    |
+-------------------+       +-----------+-----------+       +-----------------------+
                                |           |
                                |           +-----------------------+
                                v                                   |
                    +-----------------------+                       v
                    |  Lead Agent / Planner |           +-----------------------+
                    |       (LLM)           |           |  Specialist Trainer   |
                    +-----------------------+           +-----------+-----------+
                                                                    |
                                                                    v
                                                        +-----------------------+
                                                        |    Secure Sandbox     |
                                                        +-----------+-----------+
                                                                    |
                                                                    v
                                                        +-----------------------+
                                                        |      Environment      |
                                                        +-----------------------+
```

## 2. Components

### 2.1. LLM-Driven Orchestrator (Lead Agent)
The "Brain" of ASTRA. While it supports cloud APIs (OpenAI, Gemini), it is optimized for **Local Execution** on Apple Silicon via **MLX**. 

#### 2.1.1. Inference Optimization Strategy
On a 24GB M4 Mac Mini, the landscape is unique. We leverage Apple's **Unified Memory Architecture** and the **Metal** framework to bypass standard bottlenecks.

**What is NOT Worth Optimizing (Already Mastered):**
We do not optimize core math or tensor operations (Matrix Multiplication, Quantization/Dequantization) as these are already perfectly tuned by Apple's **Accelerate** framework and **Metal Performance Shaders (MPS)** in the MLX/llama.cpp engines.

**What IS Worth Optimizing (ASTRA's Value-Add):**
ASTRA builds custom optimization layers on top of MLX to maximize the 24GB footprint:
- **Smart KV Caching**: Standard setups waste RAM with fixed context blocks. ASTRA implements a dynamic cache eviction policy to drop irrelevant conversation history while preserving core system instructions and code context.
- **Speculative Decoding** *(sandbox-idle only)*: Blazing-fast generation by loading a tiny "drafter" model (e.g., 1B/3B) alongside the main model. The tiny model guesses tokens, and the large model validates them in a single mathematical step. On 24GB, the drafter is only loaded when the training sandbox is inactive; the `ModelManager` is responsible for evicting it before launching a training run.
- **Structured Output Parsing**: Uses **Grammar-Based Sampling** to force the model to choose tokens that fit a specific JSON or code schema, eliminating wasted tokens and ensuring valid tool calls.

#### 2.1.2. Memory & Engine Tiers
The choice of inference engine depends on available **Unified Memory**:
- **Standard (24GB RAM)**: **Native MLX (`mlx-lm`)** for local models; **Ollama** for offloading to a second 24GB machine. Provides the lowest memory footprint by dynamically allocating VRAM and allowing for manual garbage collection to prioritize training sandboxes.
- **Advanced (64GB+ RAM)**: **vLLM (Metal)**. Recommended for high-concurrency multi-agent setups. Leverages **PagedAttention** for massive log contexts and **Continuous Batching** for simultaneous specialist reasoning.

Deployed configuration (both machines: Apple M4, 24 GB unified memory):
- **MacBook M4** — MLX inference for both agents. Runs `mlx-community/Meta-Llama-3.1-8B-Instruct-4bit` (~4.5 GB) for planning and `mlx-community/Qwen2.5-Coder-7B-Instruct-4bit` (~4 GB) for codegen/error-fix. Total inference footprint ~8.5 GB, leaving ~15 GB for the OS and orchestration layer.
- **mac-mini.local** — Dedicated training host. Receives training scripts via `SSHSandbox` (scp + nohup), executes with full 24 GB available, and streams checkpoints + logs back via rsync on completion.

*Hardware Note:* Native MLX is preferred on 24GB to avoid the pre-allocation overhead of serving engines. The 24GB unified memory must be shared between the LLM and the active training runs; quantization (Q4/Q8) is mandatory.

### 2.2. Autonomous Training Loop
The execution engine that manages the state machine of training:
- **Phase Management**: Handles transitions between curriculum steps.
- **Retry Logic**: Automatically restarts failed runs with adjusted noise or exploration parameters.
- **Goal Tracking**: Continuous comparison between current performance and target metrics.
- **Pivot Engine** (`backend/loop/pivots.py`): Detects plateaus and drives escalating pivot strategy.
  - `needs_pivot()` triggers when the last `PLATEAU_WINDOW=3` iterations show < 1% relative improvement.
  - `record_pivot()` increments `_pivot_count` unless the all-time best improved by ≥ 5% (`ESCALATION_RESET_THRESHOLD`) since the previous pivot — preventing small oscillations from resetting escalation.
  - `escalation_level()` returns the current aggression tier based on `_pivot_count`:
    - **Level 0** (`count < 2`): tune hyperparameters only.
    - **Level 1** (`count ≥ 2`): change policy network architecture in addition to HP tuning.
    - **Level 2** (`count ≥ 4`): allow algorithm switch (e.g. DQN → PPO).
    - **Level 3** (`count ≥ 6`): reshape reward function via `env_kwargs` (e.g. disable distance shaping, increase food reward).
  - `_pivot_count` is persisted to the `missions.pivot_escalation_count` DB column after every pivot and restored on server restart, so escalation survives process crashes and restarts.
  - No-op pivot detection: if the LLM proposes HP values identical to current values (including string vs. float type mismatches), the change is filtered and `record_pivot()` is called twice (faster escalation) without regenerating code.
  - **Algorithm-locked missions**: `_is_algorithm_locked(goal, algorithm)` detects when the user's goal names a specific algorithm (whole-word, case-insensitive). When locked: (1) `algo_changed` is forced to False even if the LLM proposes a switch; (2) `propose_pivot()` is called with `algorithm_locked=True`, remapping level 2 from "switch algorithm" to "reshape reward function via env_kwargs", and level 3 to more aggressive env_kwargs tuning. This ensures the user's stated algorithm is never silently replaced.
  - **LLM schema normalization**: `_normalize_pivot()` corrects common LLM deviations where adjustments are nested as `{hyperparameters: {...}, env_kwargs: {...}, policy_kwargs: {...}}` instead of the expected flat scalar dict + top-level keys. The normalizer: (1) flattens `adjustments.hyperparameters` into the flat adjustments dict; (2) promotes `adjustments.env_kwargs` to top-level `pivot["env_kwargs"]`; (3) promotes `adjustments.policy_kwargs` to top-level `pivot["policy_kwargs"]` — critical so that arch-change proposals reach the `_proposed_pky` extraction path and can be vetoed by the best-arch guard. Without this promotion, the stray key was merged into `plan["hyperparameters"]` via the adjustments path, bypassing the guard and corrupting `best_model.zip` with a mismatched architecture.
  - **Post-pivot regression detection & checkpoint recovery**: arch and algorithm pivots force a fresh training start (no warm-start, reset `best_score.txt`), which can cause temporary catastrophic regression. The state machine maintains a rolling window of per-iteration checkpoints (`checkpoints/iter/checkpoint_iter_{N}.zip`, last `ITER_CHECKPOINT_WINDOW=10` kept) saved after every evaluation. Before applying any arch/algo pivot it: (1) saves a `_pre_pivot_hps` snapshot in the plan dict; (2) calls `pivot_engine.record_arch_pivot_baseline()` to arm the regression detector. No separate `best_model_pre_pivot.zip` is written — the iter rolling window makes it redundant. After each subsequent iteration, `should_revert_pivot()` compares the post-pivot best against the pre-pivot best; if the new config is still > 20% (`PIVOT_REGRESSION_THRESHOLD`) below the baseline after `PLATEAU_WINDOW=3` iters, the state machine restores `iter/checkpoint_iter_{best_iter}.zip` (the true best-ever iteration), restores the HPs, calls `revert_escalation()` (decrements `_pivot_count` by 1), and emits a named status event identifying the exact iter restored. If the new config recovers within 3 iters, the tracking is cleared silently without reverting.
  - **Best-architecture memory**: `PivotEngine.record()` accepts an optional `policy_kwargs` argument (passed by the state machine from `plan["hyperparameters"]["policy_kwargs"]` after each iteration). Whenever the recorded goal metric equals or exceeds the current all-time best, the associated `policy_kwargs` is saved as `_best_policy_kwargs`. `best_policy_kwargs()` exposes it. The state machine passes `best_policy_kwargs`, `best_metric_value`, and `best_metric_iteration` to `LeadAgent.propose_pivot()`, which injects them into the LLM query as: `"Best performing architecture so far: {"net_arch": [...]} (best <metric>=<value> at iteration <N>) — prefer this at Level 1"`. The Level 1 escalation description in `_PIVOT_SYSTEM` also explicitly instructs the LLM to reuse the best architecture unless it is identical to the current one. This prevents the common failure mode where the LLM randomly cycles between `[256, 256]`, `[400, 300]`, and `[256, 256, 128]` on each Level 1 escalation, breaking warm-starting and erasing training progress each time the architecture changes. `_best_policy_kwargs` is persisted to `missions.best_policy_kwargs` (JSON column, migration `c3d4e5f6a7b8`) after every iteration via `_save_best_policy_kwargs()`, and restored into the engine on server restart via `restore_best_policy_kwargs(mission.best_policy_kwargs)` alongside `pivot_escalation_count` and history replay — so the best-arch hint is available immediately on resume, before any new pivot fires.

### 2.3. Multi-Tier Memory System
- **Structured Registry (SQL)**: Tracks every experiment's DNA—hyperparameters, weights, and results.
- **Vector Memory (Semantic)**: Stores "lessons learned" and semantic patterns. Each lesson must carry structured metadata (hyperparameter name, value, environment config, run ID) to enable reliable regime-specific retrieval — e.g., distinguishing lessons valid for small grids from those valid for large grids.
- **Recipe Library**: A versioned collection of "Crystallized Strategies." Each recipe is a JSON/YAML manifest (with `version` and `created_at` fields) that can be instantly re-injected into the Orchestrator to reproduce or adapt a successful run. Stored in the SQL Registry (metadata + YAML body) and indexed in ChromaDB for semantic warm-start retrieval.
- **Working Memory**: Real-time buffer for current logs and telemetry, actively injected into the Lead Agent's LLM context window to enable real-time pivot decisions.

### 2.4. Specialist Trainer (Execution)
The worker agents that interface with diverse training paradigms:
- **Universal Code Generator**: LLM-driven generation for:
  - **RL**: Gym/PettingZoo environments and policy gradients.
  - **SFT**: HuggingFace Transformers, LoRA/QLoRA configurations, and dataset formatting.
  - **ML**: Scikit-learn, PyTorch Lightning, and XGBoost/LightGBM.
- **Framework Wrappers**: Standardized interfaces for common libraries (Transformers, SB3, PyTorch).
- **Telemetry Producer** (also referred to as the Telemetry Streamer in IMPLEMENT): Streams paradigm-specific metrics via WebSocket (e.g., Reward for RL, Perplexity for SFT, Accuracy/F1 for ML). On recovery, back-fills missed logs from the `data/` volume to the HUD.

### 2.5. Secure Execution Sandbox
The isolation layer where training actually occurs:
- **Runtime Strategy**: Depends on hardware target. On **Apple Silicon (M4)**, Docker/Podman does not support Metal GPU passthrough so training runs in a `SubprocessSandbox` (restricted host subprocess with memory cap via `resource` module). On **cloud/CUDA** targets, a Docker/Podman container with `nvidia-container-toolkit` is used. See §5.2 for full detail.
- **Resource Guard**: Enforces memory and compute limits to ensure system stability.
- **Filesystem Isolation**: Restricts training code access to specific project directories and the Model Registry.

### 2.6. Specialist Evaluator (Validation)
Independent agent that ensures the training isn't just "overfitting" to the environment:
- **Benchmark Suite**: Runs the model against a "Golden Set" of challenges.
- **Stress Tester**: Introduces noise and edge cases to verify robustness.

### 2.7. Analysis & Introspection Suite
Deep-dive tools for "Explainable AI":
- **Spatial Analyzer**: For CNNs, generates saliency maps to see what the agent is "looking at."
- **Policy Auditor**: Visualizes the action distribution to detect mode collapse or bias.

### 2.8. Resilience & Rigor Layer (Harness Principles)
Enhancements for long-running stability:
- **Safety Critic (Skeptical Peer Review)**: A specialized agent that audits the Lead Agent's plans. It uses a "GAN Pattern" to challenge assumptions and force defensive coding/planning.
- **Mission Manifest**: A structured JSON handoff artifact that stores the "Current Source of Truth." It replaces long conversation history as the primary context for each new iteration, preventing "Context Anxiety" and performance drift.
- **Validation Contract**: A multi-dimensional rubric generated during planning that defines "success" across primary metrics (e.g., reward) and secondary health signals (e.g., action entropy, loss stability).

## 3. Data Flow
1. **Initiation**: User sends goal.
2. **Recipe Retrieval**: Lead Agent queries the **Recipe Library** for similar past successes to create a "Warm-Start" plan.
3. **Planning**: Lead Agent refines the retrieved recipe or designs a new DAG from scratch.
4. **Implementation**: Specialist Trainer generates code based on the plan/recipe.
5. **Sandboxing & Execution**: Training runs in the secure environment.
6. **Promotion & Evaluation**: Standard progress tracking.
7. **Crystallization**: If the goal is met, the system distills the final, optimized strategy into a new **Recipe** and saves it to the Library.
8. **Finalization**: Registry update and report generation.

## 4. Security & Autonomy Gates

### 4.1. The Approval Controller
A centralized service that intercepts high-risk transitions in the DAG:
- **Gate: `EXECUTE_CODE`**: Pauses the loop and presents the generated script to the user for a "Safety Check." The `CodeSafetyClassifier` runs a two-stage pre-screen before presenting to the user: (1) a static regex pass that immediately approves scripts whose only network calls target `localhost`/`127.0.0.1` (telemetry), and immediately blocks known-dangerous patterns (subprocess shell injection, broad file deletion, external pip installs); (2) an LLM classification pass for ambiguous cases. Only genuinely risky scripts reach the human approval queue.
- **Gate: `RESOURCE_ALLOCATION`**: Triggers if the planned iteration exceeds the remaining "Quota" (GPU hours or memory).
- **Gate: `DEPLOY_MODEL`**: Requires human sign-off before a champion model is moved from the Registry to a production endpoint.

### 4.2. Autonomy Tiers
ASTRA supports three operating modes:
1. **Guided**: Every iteration step requires an "Approve/Reject" signal. All gates are active; Silent Mode (PRD §5.3) is disabled.
2. **Supervised (Default)**: ASTRA iterates autonomously but pauses for `EXECUTE_CODE` and `RESOURCE_ALLOCATION`. Silent Mode may auto-promote trusted sub-tasks to bypass these specific gates based on accumulated trust score.
3. **Full Autonomy**: ASTRA runs to completion without intervention, governed only by strict Sandbox and Resource constraints. All approval gates are suppressed; Silent Mode is redundant and has no additional effect.

**Gate-priority rule**: the operating tier takes precedence. Silent Mode trust-score bypass only activates within **Supervised** mode and only for the specific gate types (`EXECUTE_CODE`, `RESOURCE_ALLOCATION`). It cannot escalate behavior beyond what the current tier permits.

### 4.3. Monitoring Dashboard (The "HUD")
A real-time interface showing:
- **Loop Status**: Current iteration count and strategic pivot history.
- **Metric Delta**: Visual gap between "Current Best" and "Target Goal."
- **Approval Queue**: Pending security requests with "Diff" views for code changes.

## 5. Runtime Architecture

ASTRA's runtime is split between **Persistent Management** and **Transient Compute**.

### 5.1. Persistent Orchestration Layer
- **Host**: Local Server, Mac Mini, or Cloud Instance (AWS/GCP).
- **Process Manager**: The FastAPI server runs as a persistent service (e.g., via `pm2` or `systemd`).
- **Autonomous Loop**: Handled by background worker processes (e.g., `asyncio` tasks or `Celery/Redis`) to ensure the training logic survives Web UI disconnections.

### 5.2. Transient Compute Layer (The Sandbox)
- **Isolation**: Sandbox strategy depends on the hardware target:
  - **Apple Silicon (M4)**: Docker/Podman does **not** support Metal GPU passthrough. Training that requires the GPU runs in a **restricted host subprocess** with enforced resource limits (memory cap via `resource` module, CPU affinity via `taskset`/`psutil`). Docker is reserved for CPU-only or dependency-isolation tasks.
  - **Cloud / CUDA**: Every training iteration runs inside a **Docker** or **Podman** container with `nvidia-container-toolkit` for GPU access.
- **Lifecycle**: Sandboxes (container or subprocess) are provisioned by the Lead Agent, execute the training code, and are decommissioned once evaluation is complete.
- **GPU Passthrough**: CUDA environments use `nvidia-container-toolkit`. Apple Silicon GPU access is host-native; the `ModelManager` coordinates memory between the LLM and the training subprocess.

### 5.3. State & Persistence
- **Database**: SQLite (local) or PostgreSQL (cloud) for experiment metadata and the Model Registry.
- **Mission Store**: A specialized table tracking the active DAG state, current iteration number, and sandbox PID/ContainerID for recovery.
- **File Store**: A dedicated `data/` volume mounted to sandboxes for weights and logs.
- **Memory**: ChromaDB running as a sidecar process for vector-based semantic retrieval.

### 5.4. API Reference

| Endpoint | Description |
|---|---|
| `GET /health` | System status + memory stats |
| `GET /health/ready` | Readiness probe |
| `GET/POST/PATCH/DELETE /registry/experiments` | Experiment CRUD |
| `GET/POST/PATCH/DELETE /registry/models` | Model record CRUD (`champion_only` filter) |
| `GET/POST/PATCH/DELETE /missions` | Mission CRUD |
| `GET /missions/{id}/manifest` | Live requirement manifest state |
| `POST /agent/missions/{id}/run` | Launch the autonomous loop for a mission |
| `GET/POST /approvals` | Approval gate CRUD |
| `POST /approvals/{id}/approve\|reject` | Approve or reject a pending gate |
| `POST /approvals/{id}/auto-approve` | LLM-classify gate script; auto-approve if safe |
| `POST /telemetry/missions/{id}/metrics` | Sandbox pushes metrics |
| `WS /ws/missions/{id}/telemetry` | Live telemetry WebSocket (back-fills history on connect) |
| `WS /ws/missions/{id}/play?env_id=&fps=` | Live agent viewer — loads `best_model.zip`, streams 16×16 game frames |
| `POST /analysis/missions/{id}/saliency` | Grad-CAM saliency map |
| `POST /analysis/missions/{id}/audit` | Policy audit (action histogram + entropy) |
| `GET /recipes` | List all recipes (disk + DB merged) |
| `GET /recipes/db` | List DB-backed recipes (`domain`, `golden_only` filters) |
| `GET /recipes/search?q=` | Semantic search over recipe library |
| `GET /recipes/{name}` | Fetch a single recipe (DB-first, disk fallback) |
| `POST /recipes/crystallize/{mission_id}` | Distil a completed mission into a recipe |
| `POST /recipes/{id}/evolve` | Spawn a mutated child recipe |
| `GET /recipes/{id}/lineage` | Ancestor chain for an evolved recipe |

Interactive docs available at `http://localhost:8200/docs` when the backend is running.

### 5.5. Recovery & Resumption Logic
1. **Startup Check**: On boot, `recover_interrupted_missions()` queries the **Mission Store** for any tasks in the `RUNNING`, `PAUSED`, `PLANNING`, or `EVALUATING` state. Each query and subsequent state transition executes inside a database transaction (PRD §4.11): read current state, validate, and write new state atomically to prevent duplicate execution on concurrent restarts.
2. **Sandbox Handling**: The Mission Store tracks either a `ContainerID` (cloud/CPU) or a `SubprocessPID` (Apple Silicon GPU). `SandboxManager.recover()` checks whether the sandbox is still alive. **If alive ("reattached")**: a minimal `SubprocessSandbox` is reconstructed with `_reattach_pid` set to the stored PID; `SandboxManager.terminate()` is then called, which kills the process directly by PID (since there is no `Popen` handle after a restart). The stale subprocess would otherwise be unsupervised, so it is safer to restart cleanly from the last checkpoint. **If gone ("reset")**: no action needed. All recovered missions are reset to `PENDING` and their `container_id` / `subprocess_pid` cleared. `SandboxManager.launch()` additionally evicts and terminates any sandbox already registered for the same mission before starting a new one, guarding against leaks during mid-loop error retries.
3. **Loop Auto-Restart**: After `recover_interrupted_missions()` returns the list of recovered mission IDs, the FastAPI lifespan handler builds a fresh `LoopStateMachine` per mission and creates an asyncio task for `loop.run(mission_id)`. The loop resumes from `current_iteration` and the saved pivot plan in the DB — training continues from the last checkpoint without replanning.
   **Shutdown/hot-reload safety**: when `asyncio.CancelledError` is raised (e.g. uvicorn watchfiles hot-reload), the loop calls `SandboxManager.terminate(mission_id)` before resetting the mission to `PENDING`. This prevents the sandbox subprocess from running orphaned after a hot-reload, which would otherwise cause interleaved telemetry writes if the loop is immediately restarted by the new process.
4. **Telemetry Catch-up**: The **Telemetry Producer** back-fills any missed log entries from the `data/` volume to the HUD when the client reconnects, covering the outage window so operators can assess model behaviour during downtime.
