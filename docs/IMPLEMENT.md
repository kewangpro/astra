# ASTRA: Implementation Roadmap

This document outlines the phased implementation strategy for `ASTRA`.

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
    - **Note:** Full training stack (`stable-baselines3`, `transformers`, `peft`, `trl`, `torch`, `scikit-learn`, `pytorch-lightning`) is installed in the project environment to support `SubprocessSandbox` on Apple Silicon.

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
    - **Note:** `MLXProvider` is used for both planning (`Llama-3.1-8B-Instruct-4bit`) and coding (`Qwen2.5-Coder-7B-Instruct-4bit`) by default, optimized for 24GB unified memory.

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
    - `make run` / `make stop` / `make ports` — all services managed via Makefile.
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

## Phase 5: The Wisdom (Recipes & Sharing) ✅
*Goal: Finalize crystallization logic and the strategy sharing library.*

- [x] **Step 5.1: Recipe Crystallization Logic**
    - `backend/services/crystallizer.py`: `crystallize(mission_id)` — distils a completed mission's plan, best metric, and lessons into a YAML recipe; persists as a `RecipeRecord` in DB and writes YAML to `recipes/`; indexes in the semantic recipe library.
    - `backend/models/recipe.py`: `RecipeRecord` ORM model — tracks name, domain, task_type, hyperparameters, curriculum, score, generation, consecutive_wins, is_golden, and provenance (mission_id, parent_recipe_id).
    - `alembic/versions/f3a9b2c1d8e7_add_recipe_records.py`: DB migration for the `recipe_records` table.
    - `LoopStateMachine`: automatically calls `crystallize()` after a mission transitions to `COMPLETED`.
- [x] **Step 5.2: Recipe Library & Retrieval**
    - `backend/services/recipe_library.py`: ChromaDB collection `recipe_library` — `index_recipe()`, `search_recipes()` (semantic), `get_warm_start_hint()`.
    - `LeadAgent.plan()`: queries the recipe library before planning; injects best-matching recipe name as a warm-start hint in the planning prompt.
    - `GET /recipes/search?q=...&domain=...`: semantic search endpoint.
- [x] **Step 5.3: Strategy Evolution**
    - `backend/services/evolution.py`:
      - `MutationOperator`: perturbs numeric hyperparameters ±15% within per-param bounds.
      - `SelectionPolicy`: promotes a child only if it beats its parent by ≥1%.
      - `GenePool`: aggregates top-N recipes per domain as evolution candidates.
      - `GoldenPromoter`: awards Golden status after 3 consecutive wins; re-indexes in recipe library.
      - `RegressionChecker`: validates a candidate Golden recipe against the best existing Golden score in the domain.
      - `evolve_recipe(parent_id)`: orchestration helper — mutates, persists child to DB + YAML, indexes.
    - `POST /recipes/{recipe_id}/evolve`: spawns a mutated child recipe.
    - `GET /recipes/{recipe_id}/lineage`: returns the full ancestor chain.
    - `GET /recipes/db`: lists DB-backed records with domain/golden filters.
    - `GET /recipes` + `GET /recipes/{name}`: now DB-aware; DB records take priority over disk on name collisions.

## Phase 6: Validation & Scaling ✅
*Goal: Ensure robustness and prepare for multi-GPU/distributed use.*

- [x] **Step 6.1: Comprehensive Test Suite**
    - `pytest.ini` + `requirements-dev.txt` (pytest, pytest-asyncio, pytest-mock, aiosqlite).
    - `tests/conftest.py`: in-memory SQLite fixtures (StaticPool), `patch_db` monkeypatches `AsyncSessionLocal` in all modules.
    - **223 tests total** across unit and integration suites:
      - `test_pivot_engine.py` (9), `test_benchmark_suite.py` (6), `test_stress_tester.py` (6), `test_manifest.py` (15) — core loop logic.
      - `test_evolution.py` (22) — `MutationOperator` bounds, `SelectionPolicy` threshold logic.
      - `test_kv_cache.py` (17) — `SmartKVCache` eviction, token accounting, message ordering.
      - `test_model_manager.py` (18) — memory estimation, drafter eviction, GC trigger.
      - `test_mission_state.py` (17) — `_primary_score`, `load`, `update` state transitions.
      - `test_crystallizer.py` (28) — `_slugify`, `_next_version`, `_build_recipe_content`.
      - `test_preflight.py` (16) — `PreflightResult.summary`, package checks, dir writability.
      - `test_subprocess_sandbox.py` (13) — resource limits, PID tracking, lifecycle.
      - `test_state_recovery.py` (8) — all recoverable status variants, mixed reattach/reset.
      - `test_error_analyzer.py` (17) — `_extract_error_type`, `_extract_traceback`, `fix_script` with prior errors, `_store_lesson`, fence stripping.
      - `test_code_generator.py` (15) — `_build_user_prompt` per task type, telemetry guard, lesson injection, `_query_lessons` edge cases, `_strip_fences`.
      - `test_missions_router.py` (11) — `_parse_target_metric` for reward/accuracy/loss patterns and no-match fallback.
      - `test_loop_state_machine.py` (5, integration) — happy path, error recovery, max retries, plateau+pivot, supervised gate rejection.
- [x] **Step 6.2: Multi-GPU Orchestration**
    - `SandboxConfig.gpu_index: Optional[int]` — per-sandbox GPU device pinning.
    - `SubprocessSandbox`: injects `CUDA_VISIBLE_DEVICES` and `MPS_DEVICE_INDEX` when `gpu_index` is set.
    - `ContainerSandbox`: passes `DeviceRequest(device_ids=[str(gpu_index)])` to Docker when `gpu_index` is set.
    - `GPUPool` (in `manager.py`): least-loaded GPU assignment, `acquire()`/`release()` per mission.
    - `SandboxManager.launch()`: accepts `gpu_index` param; auto-assigns via `GPUPool` when `ASTRA_GPU_COUNT > 0`.
- [x] **Step 6.3: "Golden Set" Benchmarking**
    - `backend/evaluator/benchmark.py`: added `snake_hard`, `tetris_hard`, `nlp_perplexity` scenarios; lower-is-better metric semantics for loss/perplexity; missing-checkpoint guard in all eval functions.
    - `backend/evaluator/stress_tester.py`: `StressReport` fields (`mean`, `std`, `min`, `max`, `reproducible`); seed-0 reproducibility check; primary metric aggregation per task type.
    - `backend/services/evolution.py` — `GoldenPromoter.record_win()`: calls `RegressionChecker.passes()` before awarding Golden status; blocks promotion on regression.

## Phase 7: Resilience & Rigor (Harness Principles) ✅
*Goal: Apply Anthropic "Harness" principles to maximize long-running reliability.*

- [x] **Step 7.1: The GAN Pattern (Skeptical Peer Review)**
    - \`backend/agent/critic_agent.py\`: \`CriticAgent\` — evaluates plans on three rubric dimensions (Safety, Complexity, Overfitting Risk); returns \`CritiqueResult\` with per-dimension scores, concerns list, and overall score (0–10).
    - \`LeadAgent.revise_plan()\`: revises a plan in response to critic feedback.
    - \`LoopStateMachine\`: after planning, passes plan to Critic; if score < 7.0 (APPROVAL_THRESHOLD), asks LeadAgent to revise (max 2 rounds); proceeds regardless after cap to avoid infinite loops.
    - \`emit_critique()\` in telemetry_emitter broadcasts \`{"type": "critique", ...}\` events to the HUD.
- [x] **Step 7.2: Atomic Requirement Manifests**
    - Replace text-based goals with a structured \`requirements.json\` stored at \`data/missions/{id}/requirements.json\`.
    - Three check types: \`no_sandbox_error\` (stability), \`file_exists\` (artifact), \`metric_threshold\` (performance).
    - \`backend/models/manifest.py\`: \`Requirement\` + \`RequirementManifest\` dataclasses with save/load/is_complete.
    - \`backend/services/manifest_generator.py\`: rule-based generation from goal + target_metric + task_type; lower-is-better detection for loss/perplexity metrics.
    - \`backend/evaluator/manifest_evaluator.py\`: checks each requirement; suffix-match for metric aliases (e.g. \`accuracy\` target matches \`validation_accuracy\`); passed flags are permanent (not re-evaluated).
    - \`LoopStateMachine\`: generates manifest on first iteration; evaluates after every sandbox run; COMPLETED only when \`manifest.is_complete()\`.
    - \`GET /missions/{id}/manifest\`: exposes the live manifest state via API.
    - 19 unit tests covering model, generator, and evaluator.
- [x] **Step 7.3: The "Clean Handoff" Protocol**
    - \`backend/services/session_summary.py\`: \`write_session_summary()\` writes \`SESSION_SUMMARY.md\` to \`data/missions/{id}/\` at the end of every iteration.
    - File captures: last successful action (iteration + metrics + manifest status), current blocker, and exact next step (pivot, next iteration, or completion).
    - Written rule-based (no LLM) for reliability; \`LoopStateMachine\` calls it after each refine step.
- [x] **Step 7.4: Pre-Flight & Post-Flight Verification**
    - \`backend/services/preflight.py\`: \`PreflightChecker.run()\` checks data dir writability, sandbox Python availability, and task-type specific package imports before the loop starts.
    - Results emitted to the HUD event stream; failures are warnings (not fatal) to avoid blocking valid missions.
    - Post-flight LLM-generated test cases omitted (unreliable); manifest requirement flags (Step 7.2) serve as the per-requirement verification gate.
- [x] **Step 7.5: Artifact-Based State Management**
    - \`backend/services/mission_state.py\`: \`update()\` maintains \`MISSION_MANIFEST.json\` in \`data/missions/{id}/\`.
    - Tracks: best hyperparameters, best score, best algorithm, per-iteration history (last 20), and lessons learned.
    - Updated after every evaluation; bounds file size via \`_MAX_HISTORY=20\`.
- [x] **Step 7.6: The Critique HUD**
    - \`frontend/src/components/hud/CritiqueTrace.tsx\`: shows each critic review as a card with overall score, per-dimension rubric scores (colour-coded), concerns list, and feedback text.
    - Conditionally rendered in the HUD sidebar when critique events are present (alongside PivotTimeline).
    - \`LogStream\`: \`"critique"\` events render with purple \`CRT\` label; \`emit_critique()\` persists events to telemetry JSONL for back-fill on reconnect.

## Phase 8: Autonomous Learning & HUD Polish ✅
*Incremental improvements driven by live CartPole-v1 mission runs.*

- [x] **Autonomous error learning (ErrorAnalyzer + CodeGenerator + StateMachine)**
    - `backend/agent/error_analyzer.py`: updated `_SYSTEM_PROMPT` to scan the *entire* script for all instances of an error class per pass (not just the traceback line); extended `fix_script` signature with `prior_errors`, `mission_id`, `domain`; added `_store_lesson()` — persists each fix to ChromaDB via `vector_memory.add_lesson`.
    - `backend/agent/code_generator.py`: `_query_lessons(plan)` retrieves domain-relevant lessons from ChromaDB before generation and injects them into the system prompt; RL template now embeds the exact `n_calls % 2048 == 0` guard code (not prose); `target_reward` is resolved from `plan.target_metric` and substituted directly; `env_id` read from plan instead of hardcoded; prohibition on `stable_baselines3.common.logger.configure()` added to system prompt.
    - `backend/loop/state_machine.py`: accumulates `error_history` across healing retries and passes `prior_errors=error_history[:-1]` to `fix_script` so the healer sees what already failed.

- [x] **HUD metric display fixes**
    - `frontend/src/lib/api.ts`: added `target_metric: Record<string, number> | null` to `Mission` type (backed by existing `MissionRead` schema field).
    - `frontend/src/components/hud/MetricGap.tsx`: reads `target_metric` dict (`{"mean_reward": 475}`) to derive metric name and target value; raw display for RL (reward), percentage display for ML (accuracy); arc pct always `current / target * 100`.
    - `frontend/src/components/hud/MetricChart.tsx`: same `target_metric` logic; y-axis domain and tick formatter switch between raw and fraction modes; reference line label shows `"target 475"` (not `"target 92%"`).
    - `frontend/src/components/hud/LogStream.tsx`: filters out `metric`-type events (shown in MetricHistory instead), eliminating per-step telemetry spam (hundreds of events per run).
    - `backend/routers/missions.py` `_parse_target_metric`: already correctly extracts `{"mean_reward": 475}` from free-text goals — no change needed.

- [x] **Run button navigation + CritiqueTrace height**
    - `frontend/src/components/command-center/MissionsGrid.tsx`: Run button `onSuccess` navigates to `/missions/{id}` via `useRouter`.
    - `frontend/src/components/hud/CritiqueTrace.tsx`: outer container capped at `maxHeight: "24rem"` to match LogStream.
