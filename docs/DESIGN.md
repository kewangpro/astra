# astra: Design Document

**Architecture Version:** 1.0.0  
**Core Stack:** Python, PyTorch, SQLAlchemy (Registry), FastAPI (Backend API), Next.js 15 (Frontend Dashboard)

---

## 1. System Overview
astra is designed as a modular system where a **Lead Agent** orchestrates several **Specialist Agents**, served via a high-performance **FastAPI** backend and a **Next.js** professional dashboard.

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
The "Brain" of astra. While it supports cloud APIs (OpenAI, Gemini), it is optimized for **Local Execution** on Apple Silicon via **MLX** or **Ollama**. On **24GB RAM (M4)**, we prioritize models that leave at least 8-12GB free for training sandboxes. Recommended models:
- **Lead Agent (Planning/Reasoning)**: Llama-3.1-8B (Instruct, Q8_0) or Mistral-Nemo-12B (Q4_K_M).
- **Specialist Generator (Coding)**: DeepSeek-Coder-V2-Lite (MoE) or CodeLlama-13B (Q4_K_M).

*Hardware Note:* The 24GB unified memory must be shared between the LLM and the active training runs. Using 4-bit or 8-bit quantization is required to maintain a low memory footprint.

### 2.2. Autonomous Training Loop
The execution engine that manages the state machine of training:
- **Phase Management**: Handles transitions between curriculum steps.
- **Retry Logic**: Automatically restarts failed runs with adjusted noise or exploration parameters.
- **Goal Tracking**: Continuous comparison between current performance and target metrics.

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
- **Telemetry Producer** (also referred to as the Telemetry Streamer in IMPLEMENT): Streams paradigm-specific metrics via WebSocket (e.g., Reward for RL, Perplexity for SFT, Accuracy/F1 for ML). On recovery, back-fills missed logs from the `storage/` volume to the HUD.

### 2.5. Secure Execution Sandbox
The isolation layer where training actually occurs:
- **Containerized Runtime**: Uses Docker or Podman to isolate the training environment.
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
- **Gate: `EXECUTE_CODE`**: Pauses the loop and presents the generated script to the user for a "Safety Check."
- **Gate: `RESOURCE_ALLOCATION`**: Triggers if the planned iteration exceeds the remaining "Quota" (GPU hours or memory).
- **Gate: `DEPLOY_MODEL`**: Requires human sign-off before a champion model is moved from the Registry to a production endpoint.

### 4.2. Autonomy Tiers
Astra supports three operating modes:
1. **Guided**: Every iteration step requires an "Approve/Reject" signal. All gates are active; Silent Mode (PRD §5.3) is disabled.
2. **Supervised (Default)**: Astra iterates autonomously but pauses for `EXECUTE_CODE` and `RESOURCE_ALLOCATION`. Silent Mode may auto-promote trusted sub-tasks to bypass these specific gates based on accumulated trust score.
3. **Full Autonomy**: Astra runs to completion without intervention, governed only by strict Sandbox and Resource constraints. All approval gates are suppressed; Silent Mode is redundant and has no additional effect.

**Gate-priority rule**: the operating tier takes precedence. Silent Mode trust-score bypass only activates within **Supervised** mode and only for the specific gate types (`EXECUTE_CODE`, `RESOURCE_ALLOCATION`). It cannot escalate behavior beyond what the current tier permits.

### 4.3. Monitoring Dashboard (The "HUD")
A real-time interface showing:
- **Loop Status**: Current iteration count and strategic pivot history.
- **Metric Delta**: Visual gap between "Current Best" and "Target Goal."
- **Approval Queue**: Pending security requests with "Diff" views for code changes.

## 5. Runtime Architecture

Astra's runtime is split between **Persistent Management** and **Transient Compute**.

### 5.1. Persistent Orchestration Layer
- **Host**: Local Server, Mac Mini, or Cloud Instance (AWS/GCP).
- **Process Manager**: The FastAPI server runs as a persistent service (e.g., via `pm2` or `systemd`).
- **Autonomous Loop**: Handled by background worker processes (e.g., `asyncio` tasks or `Celery/Redis`) to ensure the training logic survives Web UI disconnections.

### 5.2. Transient Compute Layer (The Sandbox)
- **Isolation**: Every training iteration runs inside a **Docker** or **Podman** container.
- **Lifecycle**: Containers are provisioned by the Lead Agent, execute the training code, and are decommissioned once evaluation is complete.
- **GPU Passthrough**: Supports `nvidia-container-toolkit` for CUDA access within the sandbox.

### 5.3. State & Persistence
- **Database**: SQLite (local) or PostgreSQL (cloud) for experiment metadata and the Model Registry.
- **Mission Store**: A specialized table tracking the active DAG state, current iteration number, and sandbox PID/ContainerID for recovery.
- **File Store**: A dedicated `storage/` volume mounted to sandboxes for weights and logs.
- **Memory**: ChromaDB running as a sidecar process for vector-based semantic retrieval.

### 5.4. Recovery & Resumption Logic
1. **Startup Check**: On boot, the Orchestrator queries the **Mission Store** for any tasks in the `RUNNING` or `PAUSED` state. Each query and subsequent state transition must execute inside a database transaction to satisfy the atomicity guarantee (PRD §4.11): read current state, validate, and write new state atomically to prevent duplicate execution on concurrent restarts.
2. **Sandbox Re-attachment**: Check if the stored `ContainerID` maps to a live, running container (via `docker inspect`). If the container is still running, attach to its stdout/stderr stream for telemetry catch-up. If the container is stopped, not found, or the `ContainerID` is `NULL` (e.g., crash occurred before container launch or after decommission), skip re-attachment and provision a new container from the last known checkpoint path in the Mission Store.
3. **Telemetry Catch-up**: The **Telemetry Producer** back-fills any missed log entries from the `storage/` volume to the HUD, covering the outage window so operators can assess model behavior during downtime.
