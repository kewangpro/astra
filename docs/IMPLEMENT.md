# astra: Implementation Roadmap

This document outlines the phased implementation strategy for `astra`.

---

## Phase 1: The Foundation (Backend & Memory) ✅
*Goal: Establish the core API, database schema, and project structure.*

- [x] **Step 1.1: Project Scaffolding**
    - FastAPI backend at `backend/` with lifespan startup/shutdown hook.
    - `backend/config.py`: Pydantic Settings reading `ASTRA_*` env vars (see `.env.example`).
    - `backend/logging_config.py`: Structured stdout logging; noisy third-party loggers silenced.
- [x] **Step 1.2: Model Registry & Mission Store (SQL Storage)**
    - SQLAlchemy async models: `Experiment`, `ModelRecord`, `Mission`, `Metric` (`backend/models/`).
    - Alembic initialized; initial schema migration at `alembic/versions/`.
    - `Mission` table tracks `status`, `current_iteration`, `container_id`, `subprocess_pid`, `last_checkpoint_path`.
    - `backend/services/state_recovery.py`: queries `RUNNING`/`PAUSED` missions on boot and atomically resets them to `PENDING`. Sandbox re-attachment is deferred to Step 2.1.
    - **Note:** Python 3.9 is in use — all type hints use `Optional[X]` from `typing` (not `X | None`) and files include `from __future__ import annotations`.
- [x] **Step 1.3: Vector Memory (Semantic Storage)**
    - `backend/services/vector_memory.py`: ChromaDB (persistent) + `sentence-transformers` (`all-MiniLM-L6-v2`) for "Lessons Learned."
    - Lessons stored with structured metadata (`run_id`, `domain`, `hyperparameter_name`, `hyperparameter_value`, `environment_config`) enabling regime-specific retrieval (DESIGN §2.3).
    - FAISS was considered but ChromaDB was chosen for its built-in persistence and metadata filtering.
- [x] **Step 1.4: Base API Endpoints**
    - `GET/POST/PATCH/DELETE /registry/experiments` — Experiment CRUD.
    - `GET/POST/PATCH/DELETE /registry/models` — Model Record CRUD (supports `champion_only` filter).
    - `GET/POST/PATCH/DELETE /missions` — Mission CRUD.
    - `GET /recipes`, `GET /recipes/{name}` — Serve YAML recipes from `recipes/` (full DB-backed crystallization in Phase 5).
    - `GET /health`, `GET /health/ready` — Health check with live memory stats.

---

## Phase 2: The Execution (Sandbox & Trainers)
*Goal: Enable safe, containerized code execution and specialized training logic.*

- [ ] **Step 2.1: Sandbox Manager**
    - Implement a unified `SandboxManager` with two backends:
      - **SubprocessSandbox** (Apple Silicon): spawn a restricted host subprocess with memory/CPU limits via `psutil`/`resource`; store `SubprocessPID` in the Mission Store.
      - **ContainerSandbox** (Cloud/CUDA): Docker/Podman orchestration with `nvidia-container-toolkit`; store `ContainerID` in the Mission Store.
    - Define resource limiting policies (CPU/GPU/RAM) for both backends.
    - Extend the State Recovery Manager (Step 1.2) with sandbox re-attachment: for containers, check via `docker inspect`; for subprocesses, check via `psutil.pid_exists()`; restart from last checkpoint if the sandbox is gone.
- [ ] **Step 2.2: Universal Specialist Trainer**
    - Build the base `Trainer` class.
    - Enforce checkpoint cadence: write weights + optimizer state to `data/` volume every 2–5 minutes of wall-clock time; register checkpoint path in the Model Registry as metadata.
    - Implement `RLTrainer` (wrapping SB3/PyTorch).
    - Implement `SFTTrainer` (wrapping HuggingFace/LoRA). Override default `save_strategy` to `steps` with `save_steps` tuned to the 2–5 minute target.
    - Implement `MLTrainer` (wrapping Scikit-learn/Lightning).
- [ ] **Step 2.3: Telemetry Producer**
    - Implement a WebSocket-based metrics exporter from the sandbox to the backend (FastAPI → HUD).
    - Implement back-fill logic: on recovery, replay missed log entries from `data/` volume to the HUD.

## Phase 3: The Brain (LLM & Autonomous Loop)
*Goal: Implement the planning, self-healing, and iteration logic.*

- [ ] **Step 3.1: Lead Agent (The Orchestrator)**
    - Integrate **Native MLX Inference** (via `mlx-lm`) as the primary local provider for 24GB hardware.
    - Implement a `ModelManager` to dynamically adjust LLM memory footprint and trigger garbage collection when training sandboxes require more VRAM.
    - Implement **Smart KV Caching**: custom eviction policy for conversation history vs. system/code context.
    - Implement **Speculative Decoding** *(sandbox-idle only)*: load 1B/3B drafter models when no training run is active; `ModelManager` must evict before sandbox launch.
    - Implement **Structured Output Parsing**: grammar-based sampling for JSON and code blocks.
    - Setup prefix caching for efficient real-time log analysis.
    - (Optional) Build an abstraction layer to support **vLLM (Metal)** for high-memory environments.
    - Setup system prompts for strategic goal decomposition.
- [ ] **Step 3.2: Code Generator & Self-Healer**
    - Create prompt templates for generating training scripts.
    - Implement the "Error Analyzer" that fixes code based on stack traces.
- [ ] **Step 3.3: The Autonomous Loop State Machine**
    - Implement the logic: Plan -> Implement -> Sandbox -> Execute -> Eval -> Refine.
    - Add support for strategic pivots (e.g., hyperparameter adjustments).
- [ ] **Step 3.4: Specialist Evaluator**
    - Build the independent `Evaluator` agent (DESIGN §2.6).
    - Implement `BenchmarkSuite`: runs the model against a fixed "Golden Set" of challenges.
    - Implement `StressTester`: introduces noise and edge cases to verify robustness.
    - Wire the Evaluator as the mandatory actor in the `Eval` phase of the loop state machine.
- [ ] **Step 3.5: Analysis & Introspection Suite**
    - Implement `SpatialAnalyzer`: generates CNN saliency/activation maps and exposes them via API for the Model Registry deep-dive view (DESIGN §2.7).
    - Implement `PolicyAuditor`: computes and logs action-distribution histograms to detect mode collapse or bias.

## Phase 4: Mission Control (Web Dashboard)
*Goal: Build the professional Next.js interface for monitoring and control.*

- [ ] **Step 4.1: Dashboard Scaffolding**
    - Setup Next.js 15 with Tailwind CSS and shadcn/ui.
- [ ] **Step 4.2: The Command Center (Home)**
    - Implement the "Goal Input" bar.
    - Create the "Active Missions" grid.
- [ ] **Step 4.3: Live Training HUD**
    - Implement real-time charts using `recharts`.
    - Build the "Metric Gap" gauge and strategic pivot timeline.
    - Embed the Approval Queue panel (built in Step 4.4) as an inline overlay within the HUD view — the user must never need to navigate away from the HUD to act on a pending gate.
- [ ] **Step 4.4: Approval Controller UI**
    - Build the side-by-side code diff viewer for security gates.
    - Expose as an embeddable panel (not a standalone page) so Step 4.3 can mount it inside the HUD.

## Phase 5: The Wisdom (Recipes & Sharing)
*Goal: Finalize crystallization logic and the strategy sharing library.*

- [ ] **Step 5.1: Recipe Crystallization Logic**
    - Automate the distillation of a successful run into a YAML recipe.
- [ ] **Step 5.2: Recipe Library & Retrieval**
    - Implement semantic search for recipes based on new goals.
    - Build the "Warm-Start" planning logic.
- [ ] **Step 5.3: Strategy Evolution**
    - Implement a mutation operator: produce candidate child recipes by perturbing hyperparameter values within configurable bounds.
    - Implement a selection policy: promote a child recipe to replace its parent only if it achieves a higher score on the Specialist Evaluator's benchmark suite.
    - Build a cross-mission candidate pool: aggregate top-performing recipes across different runs of the same domain into a gene pool.
    - Define Golden Recipe promotion criteria: a recipe earns "Golden" status after N consecutive successful runs above the target threshold.
    - Add a regression test harness: ensure a newly promoted Golden Recipe does not regress on any prior benchmark the domain has solved.

## Phase 6: Validation & Scaling
*Goal: Ensure robustness and prepare for multi-GPU/distributed use.*

- [ ] **Step 6.1: Comprehensive Test Suite**
    - Integration tests for the full loop (using a "Mock" environment).
- [ ] **Step 6.2: Multi-GPU Orchestration**
    - Add support for assigning specific sandboxes to specific GPUs.
- [ ] **Step 6.3: "Golden Set" Benchmarking**
    - Harden and expand the Specialist Evaluator (built in Step 3.4): add domain-specific Golden Set scenarios for Snake, Tetris, and NLP tasks.
    - Stress-test the StressTester itself: verify edge-case coverage, noise injection calibration, and benchmark reproducibility.
