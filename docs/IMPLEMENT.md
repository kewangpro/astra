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

## Phase 2: The Execution (Sandbox & Trainers) ✅
*Goal: Enable safe, containerized code execution and specialized training logic.*

- [x] **Step 2.1: Sandbox Manager**
    - `backend/sandbox/base.py`: `BaseSandbox` ABC + `SandboxConfig` dataclass + `SandboxStatus` enum.
    - `backend/sandbox/subprocess_sandbox.py`: `SubprocessSandbox` — spawns a restricted host subprocess; memory capped via `resource.setrlimit(RLIMIT_AS)`; CPU affinity via `psutil` (no-op on macOS where it's unsupported).
    - `backend/sandbox/container_sandbox.py`: `ContainerSandbox` — Docker SDK orchestration with optional `nvidia-container-toolkit` GPU passthrough; `docker` package is a soft dependency (graceful error if missing).
    - `backend/sandbox/manager.py`: `SandboxManager` singleton — auto-selects `subprocess` on Apple Silicon (`darwin/arm64`), `container` elsewhere; exposes `launch()`, `terminate()`, `is_alive()`, `recover()`.
    - State Recovery Manager extended: `recover()` checks `psutil.pid_exists()` (subprocess) or `docker inspect` (container); reattaches if alive, resets to PENDING if dead.
    - **Note:** `psutil.cpu_affinity()` is not available on macOS — the call is wrapped in a try/except and silently skipped.
- [x] **Step 2.2: Universal Specialist Trainer**
    - `backend/trainers/base.py`: `BaseTrainer` ABC — background checkpoint thread (default 3-minute cadence, within 2–5 min target); `log_metric()` writes to `data/missions/{id}/telemetry.jsonl` AND POSTs to FastAPI; `_register_checkpoint()` PATCHes Model Registry with latest checkpoint path.
    - `backend/trainers/rl_trainer.py`: `RLTrainer` stub — SB3/PyTorch; `_run_training()` injected by Phase 3 Lead Agent.
    - `backend/trainers/sft_trainer.py`: `SFTTrainer` stub — HuggingFace/PEFT; forces `save_strategy="steps"` with `save_steps=200` default (Phase 3 tunes to observed step duration).
    - `backend/trainers/ml_trainer.py`: `MLTrainer` stub — Scikit-learn/Lightning; `_run_training()` injected by Phase 3 Lead Agent.
    - Training libraries (SB3, Transformers, PEFT, Lightning) are installed inside the sandbox, not the host.
- [x] **Step 2.3: Telemetry Producer**
    - `backend/services/connection_manager.py`: `ConnectionManager` singleton — tracks `WebSocket` connections per mission; `broadcast()` fans out to all HUD clients; auto-removes dead connections.
    - `backend/routers/telemetry.py`:
      - `POST /telemetry/missions/{id}/metrics` — sandbox pushes metrics here; appended to JSONL and broadcast to subscribers.
      - `WS /ws/missions/{id}/telemetry` — HUD connects here; on connect, back-fills full JSONL history before streaming live events.

## Phase 3: The Brain (LLM & Autonomous Loop) ✅
*Goal: Implement the planning, self-healing, and iteration logic.*

- [x] **Step 3.1: Lead Agent (The Orchestrator)**
    - `backend/agent/inference/`: `InferenceProvider` ABC + three implementations:
      - `MLXProvider` — native `mlx-lm` (Apple Silicon; lazy-load, `mx.metal.clear_cache()` on unload).
      - `VLLMProvider` — vLLM Metal (optional, 64GB+).
      - `MockProvider` — deterministic scripted responses for testing (no model weights required).
    - `backend/agent/model_manager.py`: `ModelManager` — tracks estimated VRAM usage, evicts speculative drafter before sandbox launch via `before_sandbox_launch()`, restores on `after_sandbox_exit()`, triggers GC + Metal cache clear.
    - `backend/agent/kv_cache.py`: `SmartKVCache` — three buckets (system/pinned, code/pinned-per-iteration, history/sliding-window); evicts oldest history turns when token budget exceeded.
    - `backend/agent/lead_agent.py`: `LeadAgent` — structured JSON output with retry-on-parse-error; `plan()` for goal decomposition; `propose_pivot()` for stalled runs; `analyze_logs()` for prefix-cached log analysis.
    - `backend/agent/`: vLLM abstraction layer provided via `VLLMProvider` (DESIGN §2.1.2).
    - **Production setup**: swap `MockProvider` → `MLXProvider` in `backend/routers/agent.py` after downloading a quantized model.
- [x] **Step 3.2: Code Generator & Self-Healer**
    - `backend/agent/code_generator.py`: `CodeGenerator` — prompt templates for RL (SB3), SFT (HF+PEFT), and ML (sklearn/Lightning); writes generated script to `data/missions/{id}/train.py`.
    - `backend/agent/error_analyzer.py`: `ErrorAnalyzer` — parses stack traces (extracts exception type, truncates to last 50 lines); generates and writes fixed script as `train.py.fixed_{n}.py`.
- [x] **Step 3.3: The Autonomous Loop State Machine**
    - `backend/loop/state_machine.py`: `LoopStateMachine` — full Plan→Implement→Sandbox→Execute→Eval→Refine cycle; atomic DB state transitions; `EXECUTE_CODE` approval gate in supervised mode; max 3 error-fix retries before FAILED.
    - `backend/loop/pivots.py`: `PivotEngine` — detects plateau (< 1% relative improvement over 3 iterations); calls `LeadAgent.propose_pivot()` to get hyperparameter adjustments.
    - `backend/models/approval.py`: `ApprovalGate` table (`pending/approved/rejected`; `execute_code/resource_allocation/deploy_model` gate types).
    - `POST /agent/missions/{id}/run` — launches loop as a FastAPI background task.
    - `GET/POST /approvals`, `POST /approvals/{id}/approve|reject` — approval gate CRUD.
- [x] **Step 3.4: Specialist Evaluator**
    - `backend/evaluator/specialist.py`: `SpecialistEvaluator` — mandatory Eval phase actor; finds latest checkpoint, runs BenchmarkSuite + StressTester, returns verdict.
    - `backend/evaluator/benchmark.py`: `BenchmarkSuite` — domain-keyed Golden Sets (snake, tetris, nlp); `GoldenChallenge` dataclass with `evaluate_fn` + `pass_threshold`; Phase 6 replaces stub eval functions with real env rollouts.
    - `backend/evaluator/stress_tester.py`: `StressTester` — domain-specific noise strategies (RL obs noise, SFT adversarial prompts, ML feature noise); runs across `n_seeds=3`.
- [x] **Step 3.5: Analysis & Introspection Suite**
    - `backend/analysis/spatial_analyzer.py`: `SpatialAnalyzer` — Grad-CAM via forward/backward hooks on last Conv2d layer; exposed via `POST /analysis/missions/{id}/saliency`.
    - `backend/analysis/policy_auditor.py`: `PolicyAuditor` — action-frequency histogram + entropy + mode-collapse detection (> 80% single action); exposed via `POST /analysis/missions/{id}/audit`.

## Phase 4: Mission Control (Web Dashboard) ✅
*Goal: Build the professional Next.js interface for monitoring and control.*

- [x] **Step 4.1: Dashboard Scaffolding**
    - Next.js 15 App Router + Tailwind CSS (Obsidian & Teal dark theme) at `frontend/`.
    - Port 3200 (backend 8200); `/api/*` proxied to backend via `next.config.ts` rewrites.
    - React Query (`@tanstack/react-query`) for polling; recharts for live charts.
    - `make run-frontend` / `make stop-frontend` / `make ports` updated.
- [x] **Step 4.2: The Command Center (Home)**
    - `GoalInput` — textarea + domain selector; launches mission and navigates to HUD on submit.
    - `MissionsGrid` — card-per-mission with status badge, best metric, run button for pending.
    - Global stats bar (total / running / completed / failed).
- [x] **Step 4.3: Live Training HUD**
    - `MetricGap` — current vs. target with progress bar.
    - `MetricChart` — multi-line recharts with target reference line.
    - `LogStream` — WebSocket feed (`ws://localhost:8200/ws/missions/{id}/telemetry`) with JSONL back-fill; colour-coded by log level.
    - `PivotTimeline` — vertical timeline of pivot events extracted from telemetry stream.
- [x] **Step 4.4: Approval Controller UI**
    - `ApprovalPanel` — embedded in the HUD; polls `/approvals/missions/{id}/pending` every 3 s.
    - Shows code block for `execute_code` gates; key-value table for `resource_allocation`.
    - Approve / Reject buttons call `PATCH /approvals/{id}`; toast-free, optimistic invalidation.

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
