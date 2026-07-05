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
      - `test_subprocess_sandbox.py` (17) — resource limits, PID tracking, lifecycle, reattach-pid terminate paths.
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

---

## Phase 9: Autonomous Approval & Code Robustness ✅
*Goal: Reduce human friction in supervised mode; make code generation robust against weak LLM output.*

- [x] **Auto-Approve gate (Step 9.1)**
    - `backend/agent/code_safety_classifier.py`: `CodeSafetyClassifier` — two-stage check: (1) static regex pre-filter (subprocess, eval, exec, external HTTP); (2) LLM classification using code inference provider. Returns `SafetyVerdict(safe, reason, classifier)`.
    - `backend/routers/approvals.py`: `POST /approvals/{gate_id}/auto-approve` — reads `train.py` from gate payload, runs classifier; approves gate automatically if safe, leaves PENDING with verdict if unsafe.
    - `frontend/src/lib/api.ts`: `autoApprove()` method + `AutoApproveResult` type.
    - `frontend/src/lib/hooks/useMissions.ts`: `useAutoApprove()` mutation hook.
    - `frontend/src/components/approvals/ApprovalPanel.tsx`: "Auto-Approve" button (sky-blue, only on `execute_code` gates); shows "Classifying…" spinner; renders inline safety verdict card when classifier blocks.

- [x] **Deterministic import/callback patching (Step 9.2)**
    - `backend/agent/code_generator.py`: `_patch_rl_imports()` — post-generation pass injecting missing SB3 imports (`PPO`, `BaseCallback`, `CheckpointCallback`, etc.); fixes `class Foo:` → `class Foo(BaseCallback):`; strips invalid callback constructor kwargs; replaces `import stable_baselines3 as sb3` + `sb3.PPO(...)` alias pattern with direct imports.
    - `backend/agent/error_analyzer.py`: same via `_patch_missing_imports()` + `_patch_callback_init()`; healer now writes fixes back to canonical `train.py` in addition to `.fixed_N.py`.

- [x] **Iteration counter fix (Step 9.3)**
    - `backend/loop/state_machine.py`: introduced local `current_iteration` counter initialized from `mission.current_iteration`; incremented in-memory after each `_increment_iteration()` DB write. Fixes stale iteration number on pivot events and session summaries (was always showing iter 1).

- [x] **Pivot timeline UX (Step 9.4)**
    - `frontend/src/components/hud/PivotTimeline.tsx`: displays `iter N — pivot triggered` using correct iteration from backend event (fixed by Step 9.3).

- [x] **Best-model checkpoint preservation (Step 9.6)**
    - `backend/agent/code_generator.py`: RL template now hardcodes best-model saving in the callback — `self._best_reward` tracking + `model.save("{checkpoint_dir}/best_model")` whenever `mean_reward` improves; final `model.save("{checkpoint_dir}/last_model")` after training. Previously the LLM only saved at end of training (the degraded model after policy collapse).
    - `backend/evaluator/specialist.py`: `_latest_checkpoint()` now prefers `best_model.zip` over the most-recently-modified file; falls back to newest file only if `best_model.zip` is absent.
    - `tests/unit/test_specialist_evaluator.py`: 7 new tests covering best_model preference, mtime fallback, hidden file skipping, empty dir, missing dir.
    - `tests/unit/test_code_generator.py`: 2 new tests verifying `best_model` and `_best_reward` pattern appear in RL prompt.

- [x] **Pivot hardening — clamping & architecture pivots (Step 9.5)**
    - `backend/loop/state_machine.py`: `_clamp_rl_adjustments()` enforces valid PPO hyperparameter ranges before applying pivot adjustments (learning_rate [1e-5, 1e-2], n_steps [512, 4096], n_epochs [3, 20], etc.); also ensures `batch_size <= n_steps`. Logs both raw and clamped values.
    - `backend/agent/lead_agent.py`: pivot system prompt updated with explicit valid ranges and guidance to avoid destabilizing values; `_PIVOT_SCHEMA` extended with optional `policy_kwargs` field.
    - `backend/loop/state_machine.py`: applies `policy_kwargs` (network architecture) from pivot to `plan["hyperparameters"]` before code generation.
    - `backend/agent/code_generator.py`: `_RL_TEMPLATE` passes `policy_kwargs` to LLM; LLM instructed to include `policy_kwargs=<dict>` in PPO constructor when provided (e.g. `{"net_arch": [256, 256]}`).
    - `tests/unit/test_state_machine_helpers.py`: 13 tests for `_clamp_rl_adjustments` (bounds, batch_size cap, passthrough, non-rl noop).
    - `tests/unit/test_code_generator.py`: 4 new tests for `CheckpointCallback` injection, `sb3` alias replacement, `policy_kwargs` in prompt.

- [x] **Warm-start from best checkpoint (Step 9.7)**
    - **Problem**: every iteration generated a fresh PPO model with random weights. The agent would climb to ~180 reward by step 200k, then policy collapse brought it back to negative — and the next iteration started from scratch again, repeating the cycle.
    - `backend/agent/code_generator.py`: `_RL_TEMPLATE` now includes a mandatory warm-start block (hardcoded, verbatim) that runs immediately after model construction. It loads `best_model.zip` with `PPO.load()` and copies its policy weights into the new model via `model.policy.load_state_dict(_warm.policy.state_dict())`. The new model retains the pivot's hyperparameters; only the neural network weights are transferred. If no `best_model.zip` exists (first run), the block is a no-op. Wrapped in `try/except` so an architecture mismatch after a net_arch pivot silently falls back to random weights.
    - `tests/unit/test_code_generator.py`: 1 new test verifying `_best_ckpt`, `best_model.zip`, and `load_state_dict` all appear in the generated RL prompt; 1 test verifying the `except` branch is present.

- [x] **Hardcoded pivot hyperparameters in RL template (Step 9.8)**
    - **Problem**: LLM was ignoring plan hyperparameters and hallucinating its own values (e.g. `n_steps=128` instead of pivot's `n_steps=1024`), making pivots ineffective.
    - `backend/agent/code_generator.py`: `_RL_TEMPLATE` step 2 now contains a mandatory verbatim Python code block embedding the optimizer's hyperparameter values directly — `_hp = {hyperparameters}`, `_filtered = {k: v ...}`, `_policy_kwargs = {policy_kwargs}`, `model = PPO("MlpPolicy", env, **_filtered, ...)`. LLM copies it unchanged. `policy_kwargs` renders as `None` (Python literal) when absent, or as a JSON dict when provided.
    - `tests/unit/test_code_generator.py`: 2 new tests — `_hp` and pivot values appear verbatim in prompt; `_policy_kwargs = None` renders correctly when no policy_kwargs.

- [x] **ML checkpoint path and manifest task_type reconciliation (Step 9.9)**
    - **Problem 1**: ML template said "save the model with joblib" without specifying where, so `model.joblib` landed in the wrong directory and the `file_exists` manifest requirement (`checkpoints/model.*`) never passed.
    - `backend/agent/code_generator.py`: `_ML_TEMPLATE` step 5 now hardcodes `joblib.dump(model, "{checkpoint_dir}/model.joblib")` verbatim.
    - **Problem 2**: manifest was generated at mission start using `mission.task_type` (from the UI dropdown, default `"rl"`), before the LeadAgent had a chance to identify the correct type. A scikit-learn mission created with the default dropdown got `checkpoints/*.zip` as its artifact requirement instead of `checkpoints/model.*`.
    - `backend/loop/state_machine.py`: on iteration 0, after the first plan is ready, if `plan["task_type"]` differs from `mission.task_type`, the manifest is regenerated using the plan's type and saved to disk. The LeadAgent's determination is authoritative.
    - **Problem 3**: MLX SIGABRT crash when auto-approve was clicked during active inference — two concurrent MLX calls raced on the same Metal GPU command buffer.
    - `backend/agent/inference/mlx_provider.py`: module-level `asyncio.Lock` (`_MLX_LOCK`) serializes all `generate()` calls across all `MLXProvider` instances.
    - `tests/unit/test_code_generator.py`: 1 new test verifying `joblib.dump` and checkpoint path appear in ML prompt.
    - `tests/integration/test_loop_state_machine.py`: 1 new integration test (`test_manifest_reconciled_when_plan_task_type_differs`) — mission created with `task_type="rl"`, plan returns `task_type="ml"`, asserts saved manifest uses `checkpoints/model.*`.

- [x] **Mandatory `import os` in RL scripts (Step 9.10)**
    - **Problem**: the warm-start block uses `os.path.exists()` but the LLM sometimes omitted `import os`, causing a `NameError` at runtime.
    - `backend/agent/code_generator.py`: `_RL_TEMPLATE` mandatory imports section now explicitly lists `import os`. `_patch_rl_imports()` also injects `import os` if it is absent from any LLM-generated RL script, as a belt-and-suspenders fix.
    - No new tests — the existing `test_build_user_prompt_rl_includes_warm_start_block` implicitly covers this because the warm-start block references `os.path.exists`.

- [x] **Snake-v0 registration guaranteed via post-generation injection (Step 9.11)**
    - **Problem**: `_RL_TEMPLATE` included a `{snake_setup}` placeholder with a registration preamble, but the LLM would sometimes drop it or move it after the `gym.make()` call, causing `gymnasium.error.NameNotFound: Environment Snake-v0 doesn't exist`.
    - `backend/agent/code_generator.py`: after code generation, `generate_training_script()` checks `if env_id == "Snake-v0" and "register" not in code` and prepends `_SNAKE_SETUP` directly — no LLM cooperation needed. The `{snake_setup}` placeholder is retained in the template as a hint, but the post-generation injection is the reliable guarantee.
    - `tests/unit/test_code_generator.py`: 2 new tests — `test_generate_training_script_injects_snake_preamble` (LLM omits registration → injected post-generation) and `test_generate_training_script_no_snake_preamble_for_non_snake` (CartPole → no snake preamble).

- [x] **Classifier false positive on `del _warm` (Step 9.12)**
    - **Problem**: the `CodeSafetyClassifier` LLM marked scripts as `unsafe` when they contained `del _warm` (used to free the warm-start model from memory), misreading it as a file deletion.
    - `backend/agent/code_safety_classifier.py`: `_SYSTEM` prompt clarified with three explicit rules: (1) `del variable` is Python object deletion (freeing memory), **not** a file operation — SAFE; (2) `requests.post(...)` to 127.0.0.1 or localhost is SAFE telemetry; (3) importing standard libraries (`os`, `sys`, `json`, `logging`, etc.) is SAFE.
    - No new tests — this is a prompt-engineering fix; correctness verified manually by observing auto-approve succeeding after the fix.

- [x] **Absolute checkpoint path enforcement post-generation (Step 9.13)**
    - **Problem**: the LLM substituted the absolute `{checkpoint_dir}` format variable with relative paths (`./data/missions/<uuid>/checkpoints/...`), making warm-start and model saves fragile and dependent on the process working directory.
    - `backend/agent/code_generator.py`: new `_fix_checkpoint_paths()` static method runs after code generation and replaces any relative `data/missions/<uuid>/checkpoints` pattern with the absolute `checkpoint_dir` path. Applied to both RL and ML scripts.
    - `tests/unit/test_code_generator.py`: 2 new tests — `test_fix_checkpoint_paths_replaces_relative_paths` and `test_fix_checkpoint_paths_leaves_absolute_paths_alone`.

- [x] **Classifier false positive on `sys.path.insert` (Step 9.14)**
    - **Problem**: the `CodeSafetyClassifier` LLM flagged Snake-v0 scripts as `unsafe` because they contain `sys.path.insert(0, "/Users/.../astra")` (needed to import `envs.snake_env`), which the classifier misread as a file operation on an external path.
    - `backend/agent/code_safety_classifier.py`: `_SYSTEM` prompt extended with two additional clarifications: (1) `sys.path.insert(...)` is a Python import path modification, NOT a file operation — SAFE; (2) writing files to absolute paths inside the project directory is SAFE.

- [x] **Remove domain dropdown from GoalInput (Step 9.15)**
    - **Problem**: the `domain:` dropdown on the mission creation form (`rl / sft / ml`) was a footgun — users who left it on the default `rl` got a mis-typed manifest for ML missions (the iris incident). Since the LeadAgent infers task_type from the goal text and the manifest is reconciled on iter 0, the dropdown had no functional benefit.
    - `frontend/src/components/command-center/GoalInput.tsx`: dropdown and `DOMAINS` array removed. `taskType` is hardcoded to `"rl"` on mission creation; the backend reconciles it from the plan's `task_type` on the first iteration.

- [x] **Snake-v0 live agent viewer on mission HUD (Step 9.16)**
    - **Feature**: once a Snake-v0 mission has a `best_model.zip`, the mission HUD shows a `▶ watch` button that streams the trained agent playing the game in real time.
    - `backend/routers/play.py`: new WebSocket endpoint `WS /ws/missions/{id}/play?env_id=Snake-v0&fps=12`. Loads `best_model.zip` in a thread-pool executor (SB3 is not async-native), runs PPO inference in a loop, and streams `{"type": "frame", "grid": [...256 floats...], "episode_reward": ..., ...}` JSON frames at the requested fps. Loops episodes continuously until the client disconnects.
    - `backend/main.py`: `play` router registered.
    - `frontend/src/components/hud/SnakePlayer.tsx`: canvas component (320×320px, 16×16 grid at 20px/cell). Head = teal, body = dark teal, food = red circle. Connects to the play WebSocket on button press; shows live episode number, current reward, and best episode reward. Cleans up on unmount.
    - `frontend/src/app/missions/[id]/page.tsx`: `SnakePlayer` rendered below the metric chart when `mission.goal` contains `"Snake-v0"`.

- [x] **Classifier localhost short-circuit (Step 9.17)**
    - **Problem**: the LLM safety classifier kept marking standard training scripts as unsafe because they call `requests.post("http://127.0.0.1:8200/...")` for telemetry — a false positive the prompt clarifications didn't reliably fix.
    - `backend/agent/code_safety_classifier.py`: `_static_check` now includes a positive short-circuit: after all danger patterns pass, if every `requests` call in the script targets `127.0.0.1` or `localhost`, return `safe=True` immediately without invoking the LLM. Mixed scripts (any external URL) still go to the LLM.
    - `tests/unit/test_code_safety_classifier.py`: 12 new tests covering safe/unsafe static-check paths including the localhost short-circuit and the mixed-host edge case.

---

## Phase 10: Pivot Intelligence & Live Agent Viewer ✅
*Goal: Escalating pivot strategy with 4 levels; live agent viewer; MetricChart and play endpoint polish.*

- [x] **MetricChart limited to last 3 runs (Step 10.1)**
    - **Problem**: missions with 50+ iterations accumulated 14k+ datapoints across a huge x-axis range (~25M steps), making the current training run a tiny sliver on the far right of the chart.
    - `frontend/src/components/hud/MetricChart.tsx`: chart now shows only the last 3 iteration runs (current + 2 prior). Run-reset boundaries are detected from step counter drops; the display slice is computed from the last `MAX_RUNS` reset indices. Missions with fewer than 3 runs are unaffected.

- [x] **Escalating pivot strategy (Step 9.19)**
    - **Problem**: the pivot system always proposed minor hyperparameter tweaks regardless of how many consecutive pivots had failed. The `_PIVOT_SYSTEM` prompt said "the algorithm is fixed", preventing algorithm switches even when the current algorithm (e.g. DQN) was clearly not working.
    - `backend/loop/pivots.py`: `PivotEngine` now tracks `_pivot_count` (consecutive pivots that didn't improve the best metric). `record_pivot()` increments/resets this counter. `escalation_level()` returns 0 (tune HPs), 1 (change architecture), or 2 (allow algorithm switch) based on thresholds `ESCALATION_ARCH=2` and `ESCALATION_ALGO=4`. Extended to 4 levels in Step 9.20 (`ESCALATION_REWARD=6` → level 3).
    - `backend/agent/lead_agent.py`: `_PIVOT_SYSTEM` rewritten — removed "algorithm is fixed", added escalation instructions with HP ranges for both PPO and DQN. `propose_pivot()` accepts `escalation_level` and `current_algorithm`.
    - `_PIVOT_SCHEMA`: added optional `"algorithm"` field so the LLM can propose a switch.
    - `backend/loop/state_machine.py`: passes `escalation_level` and `current_algorithm` to `propose_pivot`; calls `pivot_engine.record_pivot()` after each pivot; if the response includes a new `algorithm`, updates `plan["algorithm"]` and resets hyperparameters to the pivot's suggested values.
    - `tests/integration/test_loop_state_machine.py`: `_track_pivot` mock updated to accept new kwargs.

- [x] **Reward shaping as escalation level 3 (Step 9.20)**
    - **Problem**: purely structural pivots (HP tune → arch change → algo switch) are insufficient when the reward function itself is pathological. Distance shaping (±1/step toward food) causes greedy behaviour in Snake that leads to body collisions and a hard ceiling around 50–100.
    - `envs/snake_env.py`: added four configurable constructor params: `food_reward` (default 10.0), `death_penalty` (−10.0), `survival_bonus` (0.1), `distance_weight` (1.0). `step()` uses these instead of hardcoded constants.
    - `backend/loop/pivots.py`: added `ESCALATION_REWARD=6` threshold; `escalation_level()` returns 3 when `_pivot_count >= 6`.
    - `backend/agent/lead_agent.py`: level-3 description in `propose_pivot()` instructs the LLM to set `env_kwargs` (e.g. `distance_weight=0, food_reward=20.0`). `_PIVOT_SCHEMA` includes optional `"env_kwargs"` field.
    - `backend/loop/state_machine.py`: if pivot response includes `env_kwargs`, updates `plan["env_kwargs"]` so subsequent code generation passes them to `gym.make()`.
    - `backend/agent/code_generator.py`: `_build_user_prompt` injects `env_kwargs` into the `gym.make()` call in the RL template.
    - `tests/unit/test_snake_env.py`: 3 new tests for custom reward params.
    - `tests/unit/test_code_generator.py`: 2 new tests for env_kwargs injection.

- [x] **Play endpoint robustness — algorithm and reward config (Step 9.21)**
    - **Problem**: `backend/routers/play.py` hardcoded `PPO.load()` and `gym.make(env_id)` with no env_kwargs. If the pivot engine switches to DQN, the viewer crashes. If reward shaping is active, the displayed episode reward uses wrong defaults.
    - `backend/agent/code_generator.py`: after writing `train.py`, writes `checkpoints/train_config.json` with `{"algorithm": ..., "env_id": ..., "env_kwargs": {...}}`. This is deterministic (not LLM-generated) so it's always accurate.
    - `backend/routers/play.py`: reads `train_config.json` at WebSocket open time. Uses `_get_algo_class(algorithm)` to dispatch to the correct SB3 class (PPO/DQN/SAC/A2C via `importlib`). Passes `env_kwargs` to `gym.make()`. Falls back to PPO + empty kwargs if config file is absent (backward compat with existing missions).
    - `tests/unit/test_play_router.py`: 9 new tests covering config loading defaults/values, algorithm dispatch (PPO/DQN/A2C), unknown-algo fallback, and case-insensitive lookup.
    - `tests/unit/test_code_generator.py`: 2 new tests verifying `train_config.json` is written with correct content including env_kwargs and algorithm name.

- [x] **Telemetry metric tracking uses peak not last (Step 9.22)**
    - **Problem**: `_read_telemetry_metrics` overwrote each metric key with the latest value seen in `telemetry.jsonl`. A training run that peaked at 164 but ended at 116 would record 116 in the pivot engine, persisting a lower-than-actual best to the DB. The gap display then showed the wrong (deflated) metric.
    - `backend/loop/state_machine.py`: `_read_telemetry_metrics` now tracks `max(value)` per metric key so the state machine always records the iteration's true peak performance.
    - `_load_persisted_best`: extended to also scan the full telemetry file (offset=0) for the all-time max so existing missions with stale `best_score.txt` or DB values recover correctly on the next server restart.
    - `tests/unit/test_state_machine_helpers.py`: 5 new tests covering max-wins, multi-metric, offset correctness, empty file, and missing file cases.

- [x] **Pivot event stream shows what changed (Step 9.23)**
    - **Problem**: the pivot log row only showed the LLM's reason text with no indication of what was actually adjusted.
    - `backend/loop/state_machine.py`: after applying pivot changes, builds a `changes_summary` string — algorithm switch shown as `algo: DQN→PPO`; each HP as `key: old→new`; `net_arch` from `policy_kwargs`; reward params from `env_kwargs`. Appended to the emitted value: `"<reason> | changes: algo: DQN→PPO | lr: 1e-4→3e-4"`. Falls back to `"hyperparameter adjustment"` when no changes extracted.

- [x] **Watch endpoint uses best_model_algo.txt to select SB3 class (Step 9.24)**
    - **Problem**: `play.py` hardcoded `PPO.load()`. After a pivot switches algorithms (e.g. DQN→PPO) the previous algorithm's `best_model.zip` often still holds the best score — `train_config.json` would say PPO but the zip was saved by DQN. `PPO.load()` on a DQN zip raises `'ActorCriticPolicy' object has no attribute 'q_net'`.
    - RL template (`backend/agent/code_generator.py`): when saving `best_model.zip`, also writes `checkpoints/best_model_algo.txt` with `self.model.__class__.__name__` so the checkpoint always records which algorithm saved it.
    - `backend/routers/play.py`: `_checkpoint_algorithm()` reads `best_model_algo.txt` first (ground truth), falls back to `train_config.json`. If loading still fails with the detected algorithm, tries all known SB3 classes (PPO/DQN/SAC/A2C) in order — prevents any algorithm mismatch from hard-crashing the viewer.
    - `tests/unit/test_play_router.py`: 4 new tests for `_checkpoint_algorithm` (prefers algo file, fallback to config, empty file, no config).

---

## Phase 11: Resilience, Environments & Dual Metrics ✅
*Goal: Pivot persistence across restarts; Tetris-v0 environment; dual metric tracking (MetricHistory=mean_reward / MetricGap=goal metric).*

- [x] **MetricGap redesign — best vs current iteration (Step 11.1)**
    - **Problem**: the gap widget showed `best_metric_value` labeled with `current_iteration`, making it look like the current iteration achieved the peak score when the peak may have been several iterations earlier.
    - `backend/models/mission.py` + Alembic migration `a1b2c3d4e5f6`: added `best_metric_iteration` (int, nullable) and `current_metric_value` (str, nullable) columns to the `missions` table.
    - `backend/loop/pivots.py`: `best_metric_iteration()` returns the iteration index that achieved the best score (seed entry at −1 maps to `None`); refactored via shared `_best_entry()` helper.
    - `backend/loop/state_machine.py`: persists `best_metric_iteration` alongside `best_metric_value`; saves `current_metric_value` after each eval iteration.
    - `frontend/src/lib/api.ts`: `Mission` type extended with `best_metric_iteration` and `current_metric_value`.
    - `frontend/src/components/hud/MetricGap.tsx`: arc gauge shows best-ever value; gap and percentage sit below the arc; right column shows "best at iter X" and current iteration score separately. Current score hidden when it equals the best (no redundancy).
    - `tests/unit/test_pivot_engine.py`: 4 new tests for `best_metric_iteration` including seed-entry suppression and seed-beaten-by-real-iter cases. Total: 329 tests.

- [x] **Pivot changes display — no-op filtering and correct old→new format (Step 9.26)**
    - **Problem 1 (no-op pivot)**: the LLM frequently returned HP adjustments identical to current values (e.g. `learning_rate: 0.0005→0.0005`). These were applied, re-generating the training script with no actual changes and wasting an iteration.
    - **Problem 2 (display X→X)**: even for real changes, the pivot event stream showed `old→new` where both sides were the new value, because `old_v` was read from `plan["hyperparameters"]` *after* `plan["hyperparameters"].update(real_adjustments)` had already mutated it.
    - **Problem 3 (LLM schema deviation)**: at escalation level 1+ the LLM sometimes returned `adjustments: {hyperparameters: {lr: ...}, env_kwargs: {...}}` (nested) instead of the flat `adjustments: {lr: ...}` + top-level `env_kwargs`. The nested dicts leaked into the event stream as `hyperparameters={...}`.
    - `backend/loop/state_machine.py`: (1) `_hp_changed()` closure filters `real_adjustments` to only keys where proposed ≠ current (with float coercion to handle LLM string types); (2) `old_hps` snapshot captured *before* `plan["hyperparameters"].update()` so display shows true old→new; (3) `_normalize_pivot()` static method flattens nested `adjustments.hyperparameters` and promotes `adjustments.env_kwargs` to top-level before processing.
    - `tests/unit/test_state_machine_helpers.py`: 12 new tests covering `_hp_changed` type coercion, `old_hps` snapshot correctness, and `_normalize_pivot` for flat/nested/mixed/top-level-env-kwargs cases. Total: 336 tests.

- [x] **Persist pivot escalation count across server restarts (Step 9.27)**
    - **Problem**: `PivotEngine._pivot_count` was pure in-memory state. Every server restart reset it to 0, preventing escalation from ever accumulating past level 1 regardless of how many pivots had been attempted. After 23+ pivots across restarts, DQN was still receiving level 0–1 (HP tune + arch change) treatment indefinitely.
    - **Problem 2 (escalation reset threshold too low)**: `PLATEAU_THRESHOLD = 0.01` (1%) meant any run-to-run oscillation in the all-time best (~1–2% variance is typical) would reset the counter after nearly every pivot.
    - `backend/models/mission.py` + Alembic migration `b2c3d4e5f6a7`: added `pivot_escalation_count` (int, nullable, default 0) to the `missions` table.
    - `backend/schemas/mission.py`: `MissionRead` extended with `pivot_escalation_count`.
    - `backend/loop/pivots.py`: added `ESCALATION_RESET_THRESHOLD = 0.05` (5% required to reset escalation); `pivot_count` property; `restore_pivot_count(count)` to seed state from DB on recovery.
    - `backend/loop/state_machine.py`: calls `_save_pivot_count()` after every `record_pivot()` (including no-op double-count); on startup restores `pivot_escalation_count` from DB into the engine via `restore_pivot_count()`.
    - `tests/unit/test_pivot_engine.py`: 11 new tests covering `pivot_count` property, `restore_pivot_count`, escalation level at each threshold, increment vs reset at 2% vs 16% improvement, and level cap at 3. Total: 347 tests.

- [x] **Algorithm-locked missions skip algo switch at escalation level 2 (Step 9.28)**
    - **Problem**: when the user's goal explicitly names an algorithm (e.g. "Train a Snake-v0 DQN agent …"), escalation level 2 was still proposing an algorithm switch to PPO, violating user intent. The mission kept reverting to DQN on the next re-plan anyway, wasting an iteration.
    - `backend/loop/state_machine.py`: `_is_algorithm_locked(goal, current_algorithm)` static method — detects whether the goal names the current algorithm as a whole word (case-insensitive regex). When locked, `algo_changed` is forced to False even if the LLM proposes a switch; a warning is logged. `algorithm_locked` flag passed to `propose_pivot()`.
    - `backend/agent/lead_agent.py`: `propose_pivot()` accepts `algorithm_locked: bool`. When True, level 2 escalation description is remapped to reward shaping ("algorithm is fixed by the user — reshape env_kwargs instead") and level 3 to more aggressive reward shaping. Algorithm-free missions keep the original 0=HPs / 1=arch / 2=algo switch / 3=rewards ladder.
    - `tests/unit/test_state_machine_helpers.py`: 6 new tests for `_is_algorithm_locked` covering explicit name, case-insensitivity, different algo, no algo in goal, partial-word non-match, and PPO. Total: 353 tests.

- [x] **"Best at iter —" fix — telemetry iteration field + startup seed (Step 9.29)**
    - **Problem**: MetricGap showed "best at iter —" for all missions. Root cause: the RL train script POSTed telemetry without an `"iteration"` field, so every telemetry metric entry had `iteration: null` in the DB. Additionally, on restart the pivot engine was always seeded at `iteration=-1` (a sentinel that maps to `None`/`—` in the UI), even when the DB already had a valid `best_metric_iteration`.
    - `backend/agent/code_generator.py`: `generate_training_script()` gains a `current_iteration: int = 0` parameter passed to `_build_user_prompt()`. The `_RL_TEMPLATE` callback block now includes `"iteration": {current_iteration}` in the telemetry POST body, baking the current ASTRA iteration number into each generated train.py so new telemetry entries carry a real iteration.
    - `backend/loop/state_machine.py`: `generate_training_script()` call passes `current_iteration`. Startup pivot engine seed uses `mission.best_metric_iteration` as the seed iteration when it is not `None` (instead of always seeding at `-1`), preserving "best at iter N" across process restarts.
    - `tests/unit/test_code_generator.py`: 2 new tests — `test_build_user_prompt_rl_includes_iteration` (verifies `"iteration": 3` rendered when `current_iteration=3`) and `test_build_user_prompt_rl_iteration_defaults_to_zero`.
    - `tests/unit/test_state_machine_helpers.py`: 2 new tests — `test_seed_uses_best_metric_iteration_when_set` and `test_seed_falls_back_to_minus_one_when_iteration_none`. Total: 363 tests.

- [x] **Callback __init__ loads best_score.txt to protect peak weights across restarts (Step 9.30)**
    - **Problem**: the `_RL_TEMPLATE` only specified `_on_step` EXACTLY; the LLM freely generated `__init__` and always added `self._best_reward = float("-inf")`. This defeated the `hasattr` lazy-init guard in `_on_step` (which was designed to read best_score.txt once), so every fresh train.py immediately overwrote best_model.zip with the first-ever callback score (e.g. 116.89) rather than preserving the previously-best 164.24 checkpoint.
    - `backend/agent/code_generator.py`: template changed from "Copy this _on_step EXACTLY" to "Copy this ENTIRE class EXACTLY". The `__init__` now explicitly loads `_best_reward` from `best_score.txt` (falling back to `-inf`), and the `hasattr` block is removed from `_on_step`. Both methods are now verbatim-locked so the LLM cannot insert a `-inf` initialization.
    - `tests/unit/test_code_generator.py`: 1 new test — `test_build_user_prompt_rl_callback_init_loads_best_score` — asserts `best_score.txt` is read in the template and `not hasattr` is absent. Total: 364 tests.

- [x] **Pivot plan persisted and used across restarts (Step 9.31)**
    - **Problem**: after a pivot modifies `plan` in-memory (new HPs, algorithm, env_kwargs), `_save_plan` was never called again, so the modifications were lost. On the next iteration the loop always called `self._agent.plan(...)` fresh, discarding the pivot. On a service restart, the LLM re-planned from scratch, ignoring the escalated pivot strategy.
    - `backend/loop/state_machine.py`: two changes — (1) after applying all pivot mutations, `_save_plan(mission_id, plan)` is called immediately to persist the modified plan; (2) two flags `skip_replan_in_memory` (in-loop pivot) and `skip_replan_from_db` (restart) replace the LLM planning call when applicable. In-loop: plan is already correct in memory, just skip the LLM call. Restart: reload `current_plan` from DB. The event stream shows "Continuing with pivoted plan" instead of "Generating training plan…" in both cases.
    - `tests/integration/test_loop_state_machine.py`: 2 new tests — `test_pivot_plan_saved_to_db` (verifies `current_plan` in DB contains pivot-modified LR after a plateau sequence) and `test_restart_uses_saved_pivot_plan` (verifies LLM `plan()` is NOT called when a mission restarts with a saved plan at `current_iteration > 0`). Also fixed `_MockLeadAgent.propose_pivot` signature to accept all kwargs. Total: 366 tests.

- [x] **Tetris-v0 custom environment + recipe + code-gen wiring (Step 9.32)**
    - `envs/tetris_env.py`: new Gymnasium-compatible Tetris environment. 20×10 board, 7 tetrominoes (I/O/T/S/Z/J/L) with pre-computed rotation tables. Action space `Discrete(40)` = rotation(0–3) × column(0–9) — placement-based, not step-by-step, which makes the RL credit-assignment problem much easier. Invalid rotations and out-of-bounds columns are clamped silently, so every action is always valid. Observation: 224-element flat float32 (200 board + 7 current one-hot + 7 next one-hot + 10 column heights). Reward is configurable: quadratic line-clear multiplier, piece placement bonus, hole/bumpiness/height penalties, death penalty. `register()` follows the same pattern as `snake_env.py`.
    - `recipes/tetris_cnn_v1.yaml`: rewrote v1.0.0 → v1.1.0. Removed CNN architecture (requires custom `BaseFeaturesExtractor` that LLMs hallucinate incorrectly; flat obs + MLP is sufficient). Changed `task_type: RL` → `rl` (lowercase). Added `env_id`, `target_metric`, full HP set (`n_steps`, `n_epochs`, `gamma`, `gae_lambda`, `vf_coef`, `max_grad_norm`), and missing reward params (`height_penalty`, `death_penalty`). Line-clear reward is quadratic (10 × lines²: 4-line Tetris = 160 vs 4 singles = 40).
    - `backend/agent/code_generator.py`: generalized `{snake_setup}` → `{env_setup}` in `_RL_TEMPLATE`. Added `_TETRIS_SETUP` constant (parallel to `_SNAKE_SETUP`). `_build_user_prompt` now selects preamble by `env_id`: Snake-v0 → `_SNAKE_SETUP`, Tetris-v0 → `_TETRIS_SETUP`, anything else → empty string. Post-generation fallback patcher extended to also inject `_TETRIS_SETUP` if `Tetris-v0` appears in generated code but `register` is absent.
    - `tests/unit/test_tetris_env.py`: 25 new tests (obs shape/bounds, action space, one-hot encoding, placement, line clearing, reward helpers, clamping, death/truncation, registration, custom params).
    - `tests/unit/test_code_generator.py`: 3 new tests (`test_build_user_prompt_injects_tetris_setup`, `test_build_user_prompt_tetris_no_snake_setup`, `test_generate_training_script_injects_tetris_preamble`). Total: 394 tests.

- [x] **Dual metric tracking — MetricHistory vs MetricGap separation (Step 9.33)**
    - **Design**: MetricHistory always shows `mean_reward` (the SB3 training signal, posted every 2048 steps by the callback). MetricGap tracks the goal metric (e.g. `lines_cleared`, `food_eaten`) via a separate post-iteration evaluation pass. This decoupling removes the need for the callback to post custom metrics mid-training.
    - `backend/agent/code_generator.py`: simplified `_RL_TEMPLATE` callback — removed `_ep_metric_buf` and secondary telemetry post entirely. Callback now only posts `mean_reward`. Added `_patch_undefined_logger` post-generation patcher: replaces `logger.warning/error/info` with `logging.*` when no `logger` is defined (LLMs routinely use `logger` without defining it). Removed `target_metric_name` from template ctx (no longer needed).
    - `backend/loop/state_machine.py`:
        - Injects `plan["target_metric"] = mission.target_metric` before calling `generate_training_script` (the `_PLAN_SCHEMA` never includes `target_metric` in LLM output, so it was always `{}` causing `target_reward=200` default).
        - After the EVALUATING step: if goal metric ≠ `mean_reward`, always writes the measured value to `telemetry.jsonl` via `_append_telemetry_metric` (new method — appends JSONL + broadcasts via `connection_manager`). If the benchmark didn't supply the metric, calls `_run_goal_metric_eval` (new method — runs 10 deterministic rollout episodes in a `asyncio.to_thread` threadpool, reads goal metric from episode-end `info` dict).
        - `_load_persisted_best` contamination guard: negative DB `best_metric_value` is discarded for non-`mean_reward` targets (e.g. a contaminated `lines_cleared = -119.67` from a prior `mean_reward` seed). `best_score.txt` guard already existed.
    - `backend/evaluator/benchmark.py`: implemented real rollout helper `_rollout(checkpoint_path, env_id, n_episodes)` — loads checkpoint with each SB3 algo class until one succeeds, runs N deterministic episodes, returns `(mean_reward, mean_info_dict)`. `_tetris_eval` and `_snake_eval` now run real episodes and return actual `lines_cleared` / `mean_reward` / `max_length`.
    - `backend/evaluator/specialist.py`: infers `domain` from `env_id` (`"Tetris-v0"` → `"tetris"`, `"Snake-v0"` → `"snake"`) so golden set challenges actually execute instead of falling through to the empty `"rl"` domain.
    - `tests/unit/test_state_machine_helpers.py`: 5 new tests for `_load_persisted_best` contamination guard (negative DB value discarded for custom targets, accepted for `mean_reward`, telemetry wins over negative DB).
    - `tests/unit/test_code_generator.py`: updated `test_build_user_prompt_rl_callback_init_loads_best_score` — asserts callback posts `mean_reward` only (no `_ep_metric_buf`). Replaced `test_build_user_prompt_rl_injects_target_metric_name` with `test_build_user_prompt_rl_callback_only_posts_mean_reward`.

- [x] **Snake-v0 food_eaten tracking + multi-word goal metric parsing (Step 9.34)**
    - `envs/snake_env.py`: added `_food_eaten` counter (reset on `reset()`, incremented when food is eaten, returned in `info` dict on every step including death). Enables post-iteration eval to measure `food_eaten` directly from episode info.
    - `backend/routers/missions.py`:
        - `_parse_target_metric` generic catch-all upgraded from single-word (`[a-zA-Z][a-zA-Z0-9_]*`) to multi-word (`[\w][\w\s]*?`) with space→underscore conversion (`re.sub(r"\s+", "_", ...)`) so "achieve food eaten of 30" → `{"food_eaten": 30}`.
        - `update_mission` PATCH handler: calls `flag_modified(mission, k)` for dict fields to ensure SQLAlchemy detects JSON column mutations.
    - `tests/unit/test_missions_router.py`: 3 new tests (`test_parse_multi_word_metric_food_eaten`, `test_parse_multi_word_metric_spaces_to_underscores`, `test_parse_multi_word_metric_case_insensitive`). Total: **413 tests**.

---

## Phase 12: Mission Lifecycle & Telemetry Hardening ✅
*Goal: Clean mission deletion; robust sandbox error detection; accurate goal metric telemetry; iteration number polish; resume hardening.*

- [x] **MetricGap iteration number formatting (Step 12.1)**
    - **Problem**: iteration references in MetricGap were shown as raw integers (e.g. `3`), making them indistinguishable from metric values at a glance.
    - `frontend/src/components/hud/MetricGap.tsx`: "current iter:" label uses `.toFixed(1)` (e.g. `4.0`) to visually distinguish it from iteration index references. "best at iter" and "iter N: value" use `Math.round()` (integer) since they refer to a specific iteration number. `bestIter = 0` is treated as unset (shows "—").

- [x] **DELETE /missions/{id} fully stops running missions (Step 12.2)**
    - **Problem**: `DELETE /missions/{id}` only removed the DB record. If the mission's asyncio loop was running, it continued as a zombie — consuming memory, posting telemetry to a now-missing mission, and blocking the event loop.
    - `backend/routers/missions.py`: `delete_mission` now: (1) pops and cancels the asyncio task from `_running_tasks` in the agent router; (2) rejects all `PENDING` approval gates for the mission (so the polling loop sees `REJECTED` and exits); (3) then deletes the DB record.
    - `tests/unit/test_delete_mission.py`: 7 new tests covering task cancellation, finished-task no-op, gate rejection, no-gates, no-task, task removed from registry, and 404. Total: **420 tests**.

- [x] **Sandbox error detection — ignore benign warnings (Step 12.3)**
    - **Problem**: `_wait_for_sandbox` matched `"Error"` anywhere in sandbox output. `"Telemetry error: HTTPConnectionPool(...): Read timed out"` (transient, training continues fine) and `"Warm-start skipped (architecture mismatch or load error)"` (expected after architecture pivots) both triggered false healer invocations.
    - `backend/loop/state_machine.py`: `_wait_for_sandbox` now filters log lines — only counts as fatal if a line contains `"Traceback"` or `"Error"` AND does not match `"Telemetry error"` or `"Warm-start skipped"`. Returns `None` for benign-only output.
    - `tests/unit/test_state_machine_helpers.py`: 5 new tests — real traceback flagged, telemetry timeout ignored, warm-start mismatch ignored, clean exit returns None, mixed benign+fatal still flagged. Total: **425 tests**.

- [x] **Telemetry event cap preserves goal metric events (Step 12.4)**
    - **Problem**: `useTelemetry` capped the WS event buffer at 500 total events. With ~244 `mean_reward` posts per iteration, a 7-iteration mission floods the cap and drops early `food_eaten` / `lines_cleared` entries — the MetricGap sparkline showed only 2 of 7 data points.
    - `frontend/src/lib/hooks/useTelemetry.ts`: when trimming past 500 events, goal metric events (`type === "metric"` and `name !== "mean_reward"`) are always preserved unconditionally. Only high-frequency events (mean_reward, status, info) are trimmed to fill the remaining budget.

- [x] **Pivot prompt includes current plan state to prevent redundant arch pivots (Step 12.5)**
    - **Problem**: `propose_pivot()` sent the LLM only the current algorithm, recent metric history, and escalation level — it had no visibility into the plan's current `policy_kwargs`, `hyperparameters`, or `env_kwargs`. At escalation level 1, the LLM proposed a new `net_arch` every single iteration without knowing one had already been applied, trapping missions in a reset-train-plateau cycle: each new architecture wiped `best_score.txt`, the model trained from scratch in 3 minutes, couldn't beat food_eaten=1.0, plateau fired again, new arch proposed, repeat.
    - `backend/agent/lead_agent.py`: `propose_pivot()` gains three optional parameters — `current_policy_kwargs`, `current_hyperparameters`, `current_env_kwargs`. These are prepended to the query so the LLM can see exactly what is already in place. The prompt instructs it to avoid repeating already-applied changes.
    - `backend/loop/state_machine.py`: call site passes `plan["hyperparameters"]["policy_kwargs"]`, the remaining hyperparameters, and `plan["env_kwargs"]` to `propose_pivot()`.
    - `tests/unit/test_state_machine_helpers.py`: 1 new test — `test_propose_pivot_passes_current_plan_context` verifies that policy_kwargs, hyperparameters, and env_kwargs from the current plan are forwarded to `propose_pivot`. Total: **426 tests**.

- [x] **Goal metric parser handles number-first phrasing (Step 12.6)**
    - **Problem**: `_parse_target_metric` only matched "achieve {metric} of {value}" word order. Goals like "achieve 20 food eaten in one game" (number before metric name) returned `{}`, so `mission.target_metric` was empty and MetricGap fell back to the default 92% accuracy target.
    - `backend/routers/missions.py`: added a second generic pattern `r"achieve\s+(\d+...)\s+([\w\s]+?)(?:\s+(?:in|per|on|within)\b|$)"` that captures number-first goals; the metric name is snake-cased the same way as the existing pattern.
    - `tests/unit/test_missions_router.py`: 4 new tests — "achieve 20 food eaten in one game", "achieve 30 lines cleared per episode", "achieve 50 food eaten" (no trailing clause), and regression check that "achieve food eaten of 30" still uses the metric-first path. Total: **430 tests**.

- [x] **Resume hardening — pivot engine fully restored on restart (Step 12.7)**
    - **Problem (Bug 1)**: `arch_changed = bool(pivot.get("policy_kwargs"))` was `True` whenever the LLM returned any `policy_kwargs`, even if identical to the current plan. This caused `best_score.txt` to reset on every plateau pivot — trapping missions in a reset-train-plateau cycle. `env_kwargs_changed` already compared values correctly; `arch_changed` did not.
    - **Problem (Bug 2)**: `_best_at_last_pivot` was not persisted or restored on restart. The first `record_pivot()` after restart saw `_best_at_last_pivot=None`, hit the `else` branch, and reset `pivot_count` to 0 — silently erasing the restored escalation level.
    - **Problem (Bug 3)**: On restart the pivot engine's `_history` had only one seeded entry. `PLATEAU_WINDOW=3` required 2 more fresh iterations before plateau detection could fire — burning 2 iterations with no pivoting after every restart.
    - **Problem (Bug 4)**: The critic re-ran on every resume and in-loop pivot iteration, costing ~30s of MLX inference for plans that were already reviewed.
    - `backend/loop/pivots.py`: added `restore_best_at_last_pivot(value)` (seeds `_best_at_last_pivot` so `record_pivot()` doesn't reset escalation) and `restore_history(entries)` (replays per-iteration goal metric entries, skipping duplicates).
    - `backend/loop/state_machine.py`: (1) `arch_changed` now compares `pivot["policy_kwargs"] != plan["hyperparameters"].get("policy_kwargs")`; (2) startup calls `restore_best_at_last_pivot(persisted_best)` and `restore_history(_load_goal_metric_history(...))`; (3) critic gated on `did_replan` — skipped on resume (`skip_replan_from_db`) and pivot continuation (`skip_replan_in_memory`); (4) `_load_goal_metric_history` reads `telemetry.jsonl` for all goal metric events, returning one entry per iteration.
    - `tests/unit/test_pivot_engine.py`: 7 new tests — `restore_best_at_last_pivot` prevents reset, None resets (regression), `restore_history` enables immediate plateau detection, skips duplicates, partial below window, `arch_changed` same value is no-op, `arch_changed` different value is True.
    - `tests/unit/test_state_machine_helpers.py`: 3 new tests — `_load_goal_metric_history` reads per-iter values, returns empty for missing file, last-value-per-iter wins. Total: **441 tests**.

---

## Phase 13: Training Continuity & Loop Recovery ✅
*Goal: Eliminate training pathologies on Snake-v0 (env_kwargs destructive replace, navigation shaping disabled by LLM, early stop at wrong scale, arch oscillation, insufficient timesteps); adaptive MetricChart x-axis; robust loop restart on service recovery.*

- [x] **env_kwargs merge (not replace) + distance_weight floor clamp (Step 13.1)**
    - **Problem 1 (destructive replace)**: when a pivot dict included `env_kwargs`, `state_machine.py` did `plan["env_kwargs"] = pivot["env_kwargs"]`, discarding all previously applied reward-shaping keys. E.g. a pivot proposing only `food_reward=20.0` would silently zero out `distance_weight` that had been set by an earlier pivot.
    - **Problem 2 (LLM disabling navigation shaping)**: at escalation level 2 (reward shaping), the LLM set `distance_weight=0.0`, killing the navigation incentive. The agent immediately lost the ability to find food after the first eat, trapping `food_eaten` at 1–2 indefinitely.
    - `backend/loop/state_machine.py`: env_kwargs pivot block now merges instead of replacing — existing keys from `plan["env_kwargs"]` are preserved; only explicitly proposed keys are overwritten. Calls new `_clamp_env_kwargs()` static method after merge.
    - `_clamp_env_kwargs` static method: enforces `distance_weight >= 0.1` (floor) so the LLM can reshape but cannot fully disable navigation shaping. Additional reward params can be clamped here in the future.
    - `tests/unit/test_state_machine_helpers.py`: 7 new tests — 3 for merge semantics (preserves existing keys, only changes proposed keys, merge into empty plan), 4 for `_clamp_env_kwargs` (zero clamped, below floor clamped, above floor preserved, unrelated key untouched). Total: **448 tests**.

- [x] **Early-stop threshold uses correct scale for custom goal metrics (Step 13.2)**
    - **Problem**: `generate_training_script()` extracted `target_reward = next(iter(tm.values()), 200)` from `target_metric`, then used that number as the `mean_reward` stop threshold. For a mission with `target_metric = {"food_eaten": 20}`, the generated train.py stopped training as soon as `mean_reward >= 20` — typically reached within 40 seconds, far too early for the agent to learn. The food_eaten count never had a chance to improve.
    - `backend/agent/code_generator.py`: added `tm_name = next(iter(tm), None)`. `target_reward` uses the actual target value only when `tm_name in (None, "mean_reward")`; for any other goal metric, it falls back to `200` (a sensible default mean_reward threshold for Snake-v0 / Tetris-v0 scale).
    - `tests/unit/test_code_generator.py`: 2 new tests — `test_target_reward_uses_value_when_target_is_mean_reward` and `test_target_reward_uses_200_when_target_is_custom_metric` (asserts `"mean_reward >= 20:"` not in prompt). Total: **450 tests**.

- [x] **Training timesteps increased to 2 M in RL template (Step 13.3)**
    - **Problem**: the RL template instructed the LLM to call `model.learn(total_timesteps=500000)`. For Snake-v0 with a 16×16 grid and `max_steps=500`, this gave the agent fewer than 1000 episodes — insufficient to learn any meaningful policy at escalation level 2+.
    - `backend/agent/code_generator.py`: `_RL_TEMPLATE` instructions updated: `total_timesteps=2000000`; comment explicitly says "Do NOT use 500000 or any smaller number."

- [x] **Arch oscillation detection — suppress cycling net_arch proposals (Step 13.4)**
    - **Problem**: after several failed pivots the LLM alternated between `[256,256]` and `[256,256,128]` on successive iterations. Each arch change reset `best_score.txt` to `-inf`, so the new architecture had to beat the all-time peak just to save a checkpoint. This cycle meant the agent perpetually trained from scratch for 3 minutes, failed to beat peak `food_eaten=1`, triggered another plateau pivot, and repeated.
    - `backend/loop/state_machine.py`: introduced `recent_arches` sliding window (last 3 entries) stored in the plan. Before accepting an arch pivot, checks if `proposed_policy_kwargs` is already in `recent_arches`. If so, logs a warning and forces `arch_changed = False`, suppressing the reset. When an arch change is accepted, the *previous* arch is appended to `recent_arches` (capped at 3).
    - `tests/unit/test_state_machine_helpers.py`: 5 new tests — oscillation suppressed when proposed arch is in recent history, not suppressed when genuinely new, window capped at 3, no-op when arch identical to current regardless of history, recent_arches updated on accepted arch change. Total: **455 tests**.

- [x] **MetricChart adaptive x-axis offset between runs (Step 13.5)**
    - **Problem**: when a run reset, `MetricChart` added a hardcoded `runOffset += 500000` to prevent step-counter collisions between runs. For 2M-timestep runs this offset was 4× too small — the current run's x-axis started inside the prior run's range, causing the chart to jump back and forth visually.
    - `frontend/src/components/hud/MetricChart.tsx`: changed `runOffset += 500000` to `runOffset += prev` where `prev` is the last step value of the preceding run. The offset now scales exactly to however many steps the prior run actually logged.

- [x] **State recovery auto-restarts loop on service restart (Step 13.6)**
    - **Problem**: `recover_interrupted_missions()` reset RUNNING missions to PENDING but never launched their `LoopStateMachine`. On service restart the mission showed as PENDING in the UI but training never resumed — the event stream showed "AWAITING EVENTS" indefinitely.
    - **Problem 2 (reattached sandbox)**: if the sandbox subprocess was still alive after a restart, the mission was left RUNNING with an orphaned subprocess — the loop had no supervisor so no evaluation or pivoting occurred.
    - `backend/services/state_recovery.py`: reattached sandboxes are now explicitly terminated via `sandbox_manager.terminate()` before being reset to PENDING, so the loop can restart cleanly from the last checkpoint.
    - `backend/main.py`: after `recover_interrupted_missions()` returns the list of recovered mission IDs, the lifespan handler builds a `LoopStateMachine` per mission, creates an asyncio task for `loop.run(mission_id)`, registers it in `agent._running_tasks`, and attaches a done-callback for cleanup. Training resumes from `current_iteration` and the saved plan in DB.
    - `tests/unit/test_state_recovery.py`: rewritten to match new behavior — all tests now assert `list` return type; `test_reattached_mission_stays_running` replaced with `test_reattached_mission_terminated_and_reset_to_pending` (asserts `terminate()` called); `test_mixed_outcomes_both_reset_to_pending` asserts both reattached and gone missions end as PENDING with only the reattached one terminated. Total count unchanged at **455 tests**.

- [x] **Plan preserved across iterations — LLM not re-called after first iteration (Step 13.7)**
    - **Problem**: `skip_replan_in_memory` was only set to `True` after a pivot fired. For all non-pivot iterations, `LoopStateMachine` called `LeadAgent.plan()` fresh on every loop cycle. This produced a new plan with empty `env_kwargs`, silently discarding `distance_weight`, `food_reward`, and other shaping values that had been set by earlier pivots.
    - **Evidence**: backend logs showed `env_kwargs={'food_reward': 25.0, 'distance_weight': 0.1}` at iteration 26 (where the service restart used the DB-persisted plan), then `env_kwargs={}` for iterations 27–28 (fresh LLM re-plan).
    - `backend/loop/state_machine.py`: added `if not skip_replan_in_memory: skip_replan_in_memory = True` at the end of each iteration, after the first plan is generated. This ensures LLM planning happens exactly once per run (at iteration 0 or immediately after a pivot), and the plan — including all `env_kwargs` — is reused for all subsequent iterations.
    - `tests/integration/test_loop_state_machine.py`: new test `test_plan_reused_across_iterations_without_pivot` — TrackingAgent counts `plan()` calls; SequenceEvaluator returns 50 → 100 → 200 (goal met, no plateau, no pivot); asserts `len(plan_calls) == 1`. Total: **456 tests** (447 unit + 9 integration).

---

## Phase 14: HUD Polish & Telemetry Performance ✅
*Goal: Eliminate UI lag on long-running missions (7000+ telemetry events); fix event stream height alignment with sidebar; cap visible log rows; clean up metric display labels.*

- [x] **WebSocket backfill as single batch message (Step 14.1)**
    - **Problem**: the backfill loop sent one WebSocket frame per event. A mission with 7000+ events triggered 7000 individual `setEvents` React state updates — several seconds of UI re-renders before the HUD became usable.
    - `backend/routers/telemetry.py`: `_backfill()` now collects all JSONL lines into a list and sends a single `{"type": "backfill_batch", "events": [...]}` frame. The per-event loop is gone.
    - `frontend/src/lib/hooks/useTelemetry.ts`: `onmessage` handler checks `msg.type === "backfill_batch"` and calls `setEvents(msg.events)` once. All subsequent live events still append individually. React processes the entire history in a single render cycle.

- [x] **Event stream capped at last 100 non-metric events (Step 14.2)**
    - **Problem**: with 7000+ events backfilled, `LogStream` rendered all of them — slow initial paint and overwhelming scroll depth.
    - `frontend/src/components/hud/LogStream.tsx`: `allVisible` filters out `metric` and `backfill_complete` events; `visible = allVisible.slice(-100)` caps the rendered list. The header badge shows the raw count up to 100, then "100+" — indicating more events exist without displaying a distracting total.

- [x] **Event stream height aligned to sidebar panels (Step 14.3)**
    - **Problem**: `LogStream` had a fixed `h-96` height regardless of whether the sidebar (CritiqueTrace + PivotTimeline) was taller or shorter, causing vertical misalignment and unwanted whitespace.
    - `frontend/src/app/missions/[id]/page.tsx`: computed `logMaxH` from sidebar visibility — `max-h-[45rem]` when both CritiqueTrace (24rem) and PivotTimeline (20rem + 1rem gap) are present; `max-h-[24rem]` when only CritiqueTrace is visible; `max-h-[20rem]` when only PivotTimeline; falls back to `h-96` when no sidebar. `LogStream` accepts a `className` prop so the parent controls height.

- [x] **Pivot History scrollable with max height (Step 14.4)**
    - **Problem**: missions with many pivots caused `PivotTimeline` to grow without bound, pushing `LogStream` far below the fold.
    - `frontend/src/components/hud/PivotTimeline.tsx`: pivot list wrapped in `<div className="overflow-y-auto max-h-64 pr-1">` — scrolls internally at 16rem, never expanding the sidebar beyond its design height.

- [x] **MetricChart x-axis: limited tick count and M/K formatter (Step 14.5)**
    - **Problem**: with 2M-step runs, Recharts generated ~20 ticks in the visible x range, all rounding to "2.0M" — visually indistinguishable and space-wasting.
    - `frontend/src/components/hud/MetricChart.tsx`: `<XAxis tickCount={6} />` limits the axis to 6 labels. `tickFormatter` formats values as `1.7M`, `200K`, or raw integer depending on magnitude.

- [x] **MetricGap displays integer iteration numbers (Step 14.6)**
    - **Problem**: `currentIter` was displayed with `.toFixed(1)`, showing "27.0" instead of "27".
    - `frontend/src/components/hud/MetricGap.tsx`: changed to `Math.round(Number(currentIter))` so iteration labels read "current iter: 27" rather than "current iter: 27.0".

---

## Phase 15: Sandbox Lifecycle Hardening ✅
*Goal: Eliminate orphaned sandbox subprocesses that survive `make stop` + `make run` cycles, causing two training processes to write interleaved telemetry to the same file.*

- [x] **Reattached sandbox terminate-by-pid (Step 15.1)**
    - **Root cause**: on service restart, `SandboxManager.recover()` constructed a fresh `SubprocessSandbox` with no `_process` handle (only the stored PID). When `terminate()` was subsequently called, its `if self._process` guard was `False`, so the process was never actually killed — it kept running and writing telemetry concurrently with the newly launched sandbox for the same mission. Two processes writing `iteration=N` events with different step sequences caused 566 false "run resets" in telemetry, making MetricChart display a zigzag line.
    - `backend/sandbox/subprocess_sandbox.py`: added `self._reattach_pid: Optional[int] = None` field. `terminate()` gains an `elif self._reattach_pid is not None` branch that kills by `psutil.Process(pid)` (SIGTERM → 10 s wait → SIGKILL) and clears `_reattach_pid`. `psutil.NoSuchProcess` is swallowed (process already gone).
    - `backend/sandbox/manager.py`: `recover()` sets `sandbox._reattach_pid = subprocess_pid` on the reconstructed sandbox so `terminate()` has the pid available.

- [x] **Stale sandbox eviction before launch (Step 15.2)**
    - **Problem**: if a mid-loop error retry called `launch()` while an existing sandbox was still registered in `self._sandboxes`, the old entry was silently overwritten without termination — leaking the old subprocess.
    - `backend/sandbox/manager.py`: `launch()` now pops any existing sandbox for the mission from `self._sandboxes` before creating the new one; calls `terminate()` on it if `is_alive()` returns True.

- [x] **Sandbox terminate on shutdown cancel (Step 15.4)**
    - **Problem**: when uvicorn's watchfiles hot-reload triggers a `CancelledError` in the mission loop, the state machine caught it and reset the mission to `PENDING` — but never terminated the running sandbox subprocess. On the next server start, `recover_interrupted_missions()` looks for `RUNNING`/`PAUSED`/`PLANNING`/`EVALUATING` missions, so it skipped the `PENDING` mission entirely. The sandbox continued running orphaned, writing telemetry without any evaluation or pivot supervision.
    - `backend/loop/state_machine.py`: `CancelledError` handler now calls `self._sandbox.terminate(mission_id)` before transitioning to `PENDING`. Exceptions from terminate are caught and logged as warnings (don't block the reset).

- [x] **Test coverage (Step 15.3)**
    - `tests/unit/test_subprocess_sandbox.py`: 4 new tests — `test_terminate_via_reattach_pid_kills_process` (psutil.Process called with stored pid), `test_terminate_via_reattach_pid_force_kills_on_timeout` (SIGKILL on TimeoutExpired), `test_terminate_via_reattach_pid_handles_already_gone` (NoSuchProcess swallowed), `test_terminate_via_reattach_pid_clears_pid` (field reset to None after kill).
    - `tests/unit/test_sandbox_manager.py` (new file): 5 tests — `test_reattach_sets_reattach_pid` (recover sets field), `test_reattach_returns_dead_when_pid_gone`, `test_recover_no_pid_no_container_returns_dead`, `test_launch_terminates_alive_existing_sandbox`, `test_launch_skips_terminate_when_existing_sandbox_dead`.
    - Total: **464 tests** (455 unit + 9 integration).

---

## Phase 16: Post-Pivot Regression Detection & Checkpoint Recovery ✅
*Goal: Prevent arch/algo pivots from permanently abandoning a good checkpoint. Detect regression automatically, maintain a rolling per-iteration checkpoint window, and revert to the true best-ever iteration.*

- [x] **Regression detector in PivotEngine (Step 16.1)**
    - **Root cause**: after an arch or algorithm pivot, the new configuration starts with randomised weights (warm-start is skipped due to shape mismatch). If the new arch trains poorly for several iterations, the system keeps escalating (proposing yet more arch changes) without ever reverting to the best-known checkpoint from before the pivot.
    - `backend/loop/pivots.py`: added `PIVOT_REGRESSION_THRESHOLD = 0.20` constant. New fields: `_pivot_applied: bool`, `_pre_pivot_best: Optional[float]`, `_post_pivot_best: Optional[float]`, `_iters_since_pivot: int`. `record()` now increments `_iters_since_pivot` and tracks `_post_pivot_best` whenever `_pivot_applied` is True. New methods:
        - `record_arch_pivot_baseline()`: arms the detector — saves current best as `_pre_pivot_best`, resets post-pivot tracking.
        - `should_revert_pivot()`: after `PLATEAU_WINDOW` post-pivot iters, returns True if `_post_pivot_best < _pre_pivot_best * (1 - PIVOT_REGRESSION_THRESHOLD)`. Clears tracking silently if the new config recovered adequately.
        - `revert_escalation()`: decrements `_pivot_count` by 1 (clamped to 0) and clears all regression state.

- [x] **Per-iteration checkpoint rolling window (Step 16.2)**
    - **Problem**: `best_model.zip` is a single file; an arch pivot immediately resets `best_score.txt` to `-inf`, causing the new arch's first training run to overwrite the previous best. A single pre-pivot backup (`best_model_pre_pivot.zip`) only captures whatever was best at the moment of the last pivot — not necessarily the true best-ever iteration.
    - `backend/loop/state_machine.py`: added `ITER_CHECKPOINT_WINDOW = 10` constant. New method `_save_iteration_checkpoint(mission_id, iteration)` called after every evaluation: copies `best_model.zip` → `checkpoints/iter/checkpoint_iter_{N}.zip`, creates the `iter/` subdirectory on first call, then prunes any files beyond the rolling window. New checkpoint layout:
      ```
      checkpoints/
        best_model.zip          ← best mean_reward across all iters
        best_model_algo.txt
        best_score.txt
        last_model.zip
        train_config.json
        iter/
          checkpoint_iter_57.zip  ← rolling window, oldest pruned
          ...
          checkpoint_iter_66.zip
      ```

- [x] **Checkpoint backup and smart revert in LoopStateMachine (Step 16.3)**
    - Before applying any arch or algo pivot: saves `plan["_pre_pivot_hps"]` and `plan["_pre_pivot_best_score"]`; calls `pivot_engine.record_arch_pivot_baseline()`. No separate `best_model_pre_pivot.zip` is written — the iter rolling window makes it redundant.
    - After `pivot_engine.record()` each iteration, checks `pivot_engine.should_revert_pivot()`. On True: restores `iter/checkpoint_iter_{best_iter}.zip` (the true best-ever iter); restores `best_score.txt`; restores `plan["hyperparameters"]`; calls `pivot_engine.revert_escalation()`; saves plan to DB; emits a named `warn` status event (`"Pivot reverted — restored checkpoint from iter 47, resuming HP tuning"`); sets `_pivot_reverted = True` to skip `needs_pivot()` on this iteration.
    - `import shutil` added for file copy operations.

- [x] **Test coverage (Step 16.4)**
    - `tests/unit/test_pivot_engine.py`: 8 new tests — `test_should_revert_pivot_detects_regression`, `test_should_revert_pivot_false_before_window`, `test_should_revert_pivot_false_when_not_armed`, `test_should_revert_pivot_false_when_recovering`, `test_should_revert_pivot_clears_state_on_recovery`, `test_revert_escalation_decrements_pivot_count`, `test_revert_escalation_clamps_at_zero`, `test_revert_escalation_clears_regression_state`.
    - `tests/unit/test_state_machine_helpers.py`: 4 new tests — `test_save_iteration_checkpoint_creates_iter_subdir`, `test_save_iteration_checkpoint_content_matches_best_model`, `test_save_iteration_checkpoint_prunes_beyond_window`, `test_save_iteration_checkpoint_noop_when_best_model_missing`.
    - Total: **476 tests** (467 unit + 9 integration).

- [x] **Best-architecture memory (Step 16.5)**
    - **Root cause**: at Level 1 escalation the LLM cycled between `[256, 256]`, `[400, 300]`, and `[256, 256, 128]` on successive pivots. Each arch change resets `best_score.txt` to `-inf` (warm-start fails on shape mismatch), so the new arch trains from scratch every 3 minutes, can't beat the all-time peak, plateau fires again, and the LLM picks another arch — permanent thrash with no learning accumulation.
    - `backend/loop/pivots.py`: added `self._best_policy_kwargs: Optional[dict] = None` to `__init__`. `record()` now accepts an optional `policy_kwargs` kwarg; when a new peak goal metric is reached *and* `policy_kwargs` is provided, saves it as `_best_policy_kwargs`. New method `best_policy_kwargs() -> Optional[dict]` returns the stored value.
    - `backend/loop/state_machine.py`: before calling `pivot_engine.record()`, captures `_current_policy_kwargs = plan.get("hyperparameters", {}).get("policy_kwargs")` and passes it as `policy_kwargs=_current_policy_kwargs`. After `record()`, passes three new args to `propose_pivot()`: `best_policy_kwargs=pivot_engine.best_policy_kwargs()`, `best_metric_value=pivot_engine.best_metric_value()`, `best_metric_iteration=pivot_engine.best_metric_iteration()`.
    - `backend/agent/lead_agent.py`:
        - Level 1 in `_PIVOT_SYSTEM` tightened: `net_arch` must come from the allowed set `[256, 256]`, `[400, 300]`, `[256, 256, 128]`; if a "Best performing architecture" is listed in the prompt, reuse it — only deviate when the best arch is identical to the current one.
        - `propose_pivot()` gains three new optional params: `best_policy_kwargs`, `best_metric_value`, `best_metric_iteration`. When `best_policy_kwargs` is not `None`, a line `"Best performing architecture so far: <json> (best <metric>=<value> at iteration <N>) — prefer this at Level 1"` is prepended to the query so the LLM can see the proven arch before proposing.
        - Added `_metric_name_from_history(history)` static helper (returns the first non-`"iteration"` key from history entries) for labelling the best-metric context line.
    - `tests/unit/test_pivot_engine.py`: 6 new tests — `test_best_policy_kwargs_none_initially`, `test_best_policy_kwargs_none_when_no_kwargs_passed`, `test_best_policy_kwargs_set_on_first_record`, `test_best_policy_kwargs_updates_to_arch_at_new_best`, `test_best_policy_kwargs_not_overwritten_by_lower_metric`, `test_best_policy_kwargs_mixed_kwargs_and_no_kwargs`.
    - `tests/unit/test_state_machine_helpers.py`: updated `test_propose_pivot_passes_current_plan_context` — fake pivot engine records `food_eaten=16` with `{"net_arch": [256,256,128]}` at iter 0; asserts `captured["best_policy_kwargs"] == {"net_arch": [256,256,128]}`, `captured["best_metric_value"] == 16.0`, and `captured["best_metric_iteration"] == 0` are forwarded to `propose_pivot`.
    - `tests/integration/test_loop_state_machine.py`: updated 3 mock `propose_pivot` signatures to accept the new optional kwargs (`best_policy_kwargs`, `best_metric_value`, `best_metric_iteration`).

- [x] **Best-architecture memory persisted across restarts (Step 16.6)**
    - **Gap**: `_best_policy_kwargs` lived only in `PivotEngine` memory. On service restart it reset to `None`, so the first Level 1 pivot after restart had no best-arch hint and could thrash again.
    - `backend/models/mission.py`: added `best_policy_kwargs: Mapped[Optional[dict]] = mapped_column(JSON)`.
    - `alembic/versions/c3d4e5f6a7b8_add_best_policy_kwargs.py`: migration adding the nullable JSON column.
    - `backend/schemas/mission.py`: `MissionRead` extended with `best_policy_kwargs: Optional[dict]`.
    - `backend/loop/pivots.py`: added `restore_best_policy_kwargs(kwargs)` to seed `_best_policy_kwargs` from DB on restart.
    - `backend/loop/state_machine.py`: (1) new `_save_best_policy_kwargs()` async helper writes to DB; called immediately after `pivot_engine.record()` each iteration; (2) startup block restores `mission.best_policy_kwargs` into the engine via `restore_best_policy_kwargs()` alongside the existing escalation-count and history restore.
    - `tests/unit/test_pivot_engine.py`: 2 new tests — `test_restore_best_policy_kwargs_seeds_value` and `test_restore_best_policy_kwargs_not_overwritten_by_lower_metric_after_restore`.
    - Total: **485 tests** (476 unit + 9 integration).

- [x] **`_normalize_pivot` policy_kwargs promotion (Step 16.7)**
    - **Root cause**: the LLM sometimes returns `policy_kwargs` nested inside the `adjustments` dict (e.g. `{"adjustments": {"policy_kwargs": {"net_arch": [512, 512]}, "learning_rate": 0.0001}}`) rather than as a top-level key. `_normalize_pivot` already handled `adjustments.hyperparameters` and `adjustments.env_kwargs` but missed `adjustments.policy_kwargs`. The stray key was left in `adjustments`, passed through `_clamp_rl_adjustments`, and then merged into `plan["hyperparameters"]` via the adjustments path — effectively applying the arch change while bypassing the `best_policy_kwargs` guard in `propose_pivot`. The corrupted plan then generated a train.py with the wrong arch (e.g. `[512, 512]`), which overwrote `best_model.zip` with a model of the wrong shape before the warm-start protection could fire.
    - `backend/loop/state_machine.py` — `_normalize_pivot()`: detect `adjustments.policy_kwargs` dict; strip it from `adjustments`; promote to top-level `pivot["policy_kwargs"]` (only if top-level is not already set). This ensures arch-change proposals from the LLM always reach the `_proposed_pky` extraction path and can be vetoed or overridden by the best-arch context injected in `propose_pivot`.
    - Recovery: once the bug triggered for mission `be61cbe2`, `best_model.zip` was restored from the last clean `[256, 256, 128]` checkpoint (`checkpoint_iter_82.zip`); `best_score.txt` was reset to the historical peak (246.35) to prevent immediate overwrite.
    - `tests/unit/test_state_machine_helpers.py`: 2 new tests — `test_normalize_pivot_promotes_nested_policy_kwargs` and `test_normalize_pivot_does_not_overwrite_existing_top_level_policy_kwargs`.
    - Total: **487 tests** (478 unit + 9 integration).

- [x] **Stop button on mission cards (Step 16.8)**
    - **Feature**: mission cards on the home page now show a stop button for running/planning/evaluating missions, matching the existing run button for pending missions.
    - `backend/routers/agent.py`: new `POST /agent/missions/{id}/cancel` endpoint — checks the mission is in a cancellable state (running/planning/evaluating), calls `task.cancel()` on the asyncio loop task, returns 202. The existing `CancelledError` handler in `LoopStateMachine` terminates the sandbox and transitions the mission to pending.
    - `frontend/src/lib/api.ts`: `cancelMission(id)` calling the new endpoint.
    - `frontend/src/lib/hooks/useMissions.ts`: `useCancelMission()` mutation hook.
    - `frontend/src/components/command-center/MissionsGrid.tsx`: converted card wrapper from `<Link>` to `<div onClick>` (so `stopPropagation` on the button reliably intercepts before card navigation fires); added `pointer-events-none` to the running shimmer overlay (it was an `absolute inset-0` div that intercepted all pointer events including the stop button click); stop button uses a direct `fetch()` call to avoid react-query cancellation interference.
    - `tests/unit/test_cancel_mission.py`: 6 new tests — running task cancelled, finished task skipped, no tracked task (idempotent), 404 on missing mission, 409 on non-cancellable status, planning mission cancellable.

- [x] **MLX Metal crash on task cancel (Step 16.9)**
    - **Root cause**: `task.cancel()` raised `CancelledError` at the `await run_in_executor(...)` line in `MLXProvider.complete()` while Metal GPU ops were mid-flight in the thread pool. The thread continued running but its Python-level asyncio future was cancelled, causing a `_MTLCommandBuffer addCompletedHandler` assertion that hung the backend event loop.
    - `backend/agent/inference/mlx_provider.py`: wrapped `run_in_executor` with `asyncio.shield()` so Metal operations complete before `CancelledError` propagates to the caller.
    - Total: **493 tests** (484 unit + 9 integration).

- [x] **Auto-approve classifier: variable URL resolution (Step 16.10)**
    - **Root cause**: `_static_check` only matched `requests.post("http://...")` with a literal string. Generated train.py scripts use `TELEMETRY_URL = "http://127.0.0.1:..."` then `requests.post(TELEMETRY_URL, ...)`. No literal URL found → fell through to `static_ambiguous` → LLM called → LLM hallucinated a file-read risk from `_sys.path.insert(0, "/Users/.../astra")` and flagged the script as unsafe.
    - `backend/agent/code_safety_classifier.py`: (1) `_static_check` now resolves URL variable definitions (`VAR = "http://..."`) into `localhost_url_vars` and `external_url_vars` sets; (2) fails fast if any `requests.post(VAR, ...)` uses a variable resolving to a non-localhost URL; (3) short-circuits to safe if all request calls (literal or variable) resolve to localhost; (4) LLM system prompt clarifications strengthened — explicitly states `_sys.path.insert` is not a file read, absolute paths in strings are not inherently unsafe.
    - `tests/unit/test_code_safety_classifier.py`: 3 new tests — `test_variable_url_localhost_is_safe`, `test_variable_url_with_sys_path_insert_is_safe` (full ASTRA template pattern), `test_variable_url_external_is_unsafe`.
    - Total: **496 tests** (487 unit + 9 integration).

- [x] **Competitive-dip pivot suppression (Step 16.11)**
    - **Problem**: `needs_pivot()` triggered on a 3-iteration dip (151→127→126) while the all-time best was 164.24, causing a destructive architecture pivot that tanked performance to ~40 mean_reward.
    - **Root cause**: the plateau check (`values[-1] <= values[0]`) correctly detected no improvement in the window, but did not distinguish between a genuine plateau (stuck far below peak) and temporary variance (brief dip near the peak).
    - `backend/loop/pivots.py`: added `PIVOT_COMPETITIVE_THRESHOLD = 0.85`. In `needs_pivot()`, after the stall check, if `0 < window_best < all_time_best` and `window_best >= all_time_best * 0.85`, suppress the pivot with a log message. Guard only fires during dips — stuck-at-peak (window_best == all_time_best) still triggers.
    - `tests/unit/test_pivot_engine.py`: updated `test_pivot_triggered_on_plateau` and `test_restore_history_enables_immediate_plateau_detection` — stuck-at-peak cases are no longer suppressed by the guard since `window_best < all_time_best` is False when values are equal.

- [x] **Regression detector persistence across restarts (Step 16.12)**
    - **Problem**: `_pivot_applied` and `_pre_pivot_best` were in-memory only. A service restart reset them to `False`/`None`, silently disabling the post-pivot regression check. After a restart mid-regression-window, `should_revert_pivot()` always returned False.
    - `backend/models/mission.py`: added `pivot_pre_best` column (`VARCHAR(100)`, nullable).
    - `backend/loop/pivots.py`: added `restore_arch_pivot_baseline(pre_pivot_best)` — re-arms `_pivot_applied=True`, `_pre_pivot_best`, and resets post-pivot tracking counters.
    - `backend/loop/state_machine.py`: (1) new `_save_pivot_pre_best()` async helper writes `pivot_pre_best` to DB; (2) called immediately after `record_arch_pivot_baseline()` when an arch/algo pivot fires; (3) cleared to `None` on revert or recovery; (4) startup block restores via `restore_arch_pivot_baseline()` and logs `re-armed regression detector with pre_pivot_best=…`; (5) `_was_pivot_applied` snapshotted before `should_revert_pivot()` to detect recovery (when `_pivot_applied` transitions True→False without a revert).
    - DB migration: `ALTER TABLE missions ADD COLUMN pivot_pre_best VARCHAR(100)`.
    - Total: **498 tests** (489 unit + 9 integration).

- [x] **`mean_reward` inflation fix in `_load_persisted_best` (Step 16.13)**
    - **Problem**: `_load_persisted_best` performed a full telemetry scan (`offset=0`) and took the MAX value. For `mean_reward` missions, the SB3 training callback posts peak training scores (e.g. 164.24) which are higher than the actual eval score. `max(candidates)` always picked the training peak, overwriting the correct DB value on every restart.
    - `backend/loop/state_machine.py`: skip the full telemetry scan when `metric_name == "mean_reward"`. `best_score.txt` and the DB are the authoritative sources for `mean_reward`; the telemetry scan remains active for custom metrics (`food_eaten`, `lines_cleared`) which are only written by the eval path.
    - Total: **498 tests** (489 unit + 9 integration).

---

## Phase 17 — Tetris Obs Refactor + Actor-Critic Trainer

*Goal: Replace the 224-element flat board observation with the proven 4-feature compact representation, and replace the rigid SB3-only code generator with a contract-based Actor-Critic approach that uses `get_next_states()` — matching the reference project that achieved 121 avg lines vs PPO's 45.*

- [x] **Tetris-v0 4-feature observation (Step 17.1)**
    - **Problem**: the 224-element flat board observation made learning extremely difficult. The reference project (`tetris_ppo_cnn`) achieved 45+ lines cleared with a plain PPO MLP using only a 4-feature obs; the flat board approach yielded ~10 lines.
    - `envs/tetris_env.py`: replaced flat 224-element obs (`200 board + 7 current-one-hot + 7 next-one-hot + 10 heights`) with compact 4-feature vector `[lines_cleared_last/4, holes/200, bumpiness/180, sum_height/200]` (all normalized to `[0, 1]`). Simplified reward to `+1 placement + lines²×10 − 2 death` (reference formula); removed hole/bumpiness/height penalty params from reward (board quality is encoded in obs, not reward). Legacy shaping kwargs (`hole_penalty`, `bumpiness_penalty`, `height_penalty`) silently absorbed via `**kwargs` for backwards compatibility. `max_steps` default raised 500→1000. Added `get_board_props()` helper for inspection.
    - `recipes/tetris_ppo_v1.yaml`: renamed from `tetris_mlp_v1.yaml`; updated `name`, `version` → 2.0.0, `target_metric` → `lines_cleared: 20`, `reward_shaping` → reference formula only, `env_kwargs.max_steps` → 1000.
    - `tests/unit/test_tetris_env.py`: rewrote all 33 tests for new obs shape `(4,)`, reward structure, and legacy-kwargs compatibility.

- [x] **Actor-Critic trainer with `get_next_states()` (Step 17.2)**
    - **Problem**: SB3 PPO treats Tetris-v0 as a standard 40-action classification problem — it picks blindly from 40 actions without seeing the resulting board states. The reference project's key insight: call `get_next_states()` to simulate all 40 placements on a board copy, evaluate each resulting 4-feature obs, and pick the best — turning action selection from a 40-way guess into a 40-way lookahead. Reference Actor-Critic achieved 121 avg lines; PPO maxed out at ~45.
    - `envs/tetris_env.py`: added `get_next_states()` returning `{action: 4-feature obs}` for all valid placements of the current piece. Board copy helpers `_drop_on`, `_clear_lines_on`, `_obs_from` operate without mutating live state. 6 new tests (39 total in `test_tetris_env.py`).
    - `backend/agent/code_generator.py`: added `_ACTOR_CRITIC_CONTRACT` — a contract-based prompt (not a rigid template) that specifies the ASTRA integration requirements (telemetry POST URL, `best_model.pth` checkpoint paths, warm-start from `.pth`, `trainer_type.txt` marker) and lets the coder model write the full PyTorch training loop (shared MLP backbone, critic head, epsilon-greedy via `get_next_states()`, experience replay buffer, TD learning). Routes on `plan["trainer_type"] == "actor_critic"`. `train_config.json` now includes `trainer_type` field.
    - `backend/loop/state_machine.py`: automatically injects `trainer_type: "actor_critic"` for `Tetris-v0` missions before code generation. Rolling checkpoint window updated to handle `.pth` alongside `.zip`; revert path (`_pre_pivot_best` restore) also supports both extensions.
    - `backend/evaluator/benchmark.py`: `_is_actor_critic()` detects PyTorch models via `trainer_type.txt` or `.pth` extension. `_rollout_actor_critic()` loads the model with `torch.load`, runs greedy evaluation via `get_next_states()` + critic head value selection. `_rollout()` routes to actor_critic path first.
    - `backend/routers/play.py`: `_run_episode_actor_critic()` uses `get_next_states()` + `torch.no_grad()` for real-time play; `_tetris_viewer_grid()` reconstructs the 224-element viewer layout from live `TetrisEnv` board state (board one-hot + current/next piece one-hots + column heights) so `TetrisPlayer.tsx` renders correctly even though training obs is only 4 floats. `_load()` checks `trainer_type.txt` and prefers `best_model.pth`; returns `(model, env, is_ac)` 3-tuple; `play_ws` uses `episode_fn = _run_episode_actor_critic if is_ac else _run_episode`.
    - `recipes/tetris_ppo_v1.yaml`: added `trainer_type: actor_critic`.
    - New tests: 5 tests for `_is_actor_critic` in `test_benchmark_suite.py`; 5 tests for `_tetris_viewer_grid` in `test_play_router.py`; 7 tests for Actor-Critic prompt routing in `test_code_generator.py`.

- [x] **Actor-Critic infrastructure hardening + visual fixes (Step 17.3)**
    - **Problem**: (1) `play_ws` crashed with `Can't get attribute 'ActorCriticNet'` because `torch.load` in the uvicorn process couldn't resolve the class defined in the training script's `__main__`. (2) Goal metric showed `54.94` (float training mean) instead of an integer max from greedy eval — `_read_telemetry_metrics` was overwriting the post-eval result with sandbox rolling mean. (3) TetrisPlayer board flashed colors on every frame because all settled cells used `PIECE_COLORS[currentPieceIdx]`. (4) Crystallizer wrote PPO-only hyperparams for actor_critic missions.
    - `envs/actor_critic_net.py` (new): canonical `ActorCriticNet` module so `torch.load` can resolve the class in any process (uvicorn, benchmark, evaluator).
    - `backend/routers/play.py` + `backend/evaluator/benchmark.py`: inject `ActorCriticNet` into `sys.modules["__main__"]` before `torch.load`. Play router now captures `_pre_clear_board` / `_last_cleared_rows` from the env and emits a highlight frame (pre-clear board state + exact cleared row indices) before every post-clear frame so the client can flash only those rows.
    - `backend/loop/state_machine.py`: `_run_goal_metric_eval` rewritten to load `.pth` actor_critic checkpoints, run 10 greedy episodes via `get_next_states()`, return `max(values)` as an integer; sandbox `mean_reward` telemetry no longer overwrites the post-eval goal metric.
    - `backend/agent/code_generator.py`: `_ACTOR_CRITIC_CONTRACT` hardened with exact Gymnasium API skeleton (`obs, _ = env.reset()`, 5-tuple `env.step()`), mandatory `from envs.actor_critic_net import ActorCriticNet`, tensor conversion, and correct `mean_reward_50` scoping. `_TETRIS_SETUP` preamble adds `import gymnasium as gym`.
    - `envs/tetris_env.py`: `step()` stores `self._pre_clear_board` (post-placement, pre-clear snapshot); `_clear_lines()` stores `self._last_cleared_rows` (list of Python ints, JSON-serializable).
    - `frontend/src/components/hud/TetrisPlayer.tsx`: per-cell color memory (`cellColorsRef`) — new cells receive the placing piece's color and keep it; color memory shifts down on line clears. Highlight frame from server draws a yellow overlay on only the cleared rows (no full-board flash). Piece sidebar still uses `PIECE_COLORS`.
    - `frontend/src/components/hud/MetricGap.tsx`: sparkline filter excludes live telemetry from the running iteration (`e.iteration >= currentIter`); uses `currentIter + 1` cutoff when mission is not running so the completed iteration's post-eval point is included.
    - `backend/services/crystallizer.py`: added `_VALID_AC_KWARGS`; `_clean_rl_hyperparams` accepts `trainer_type` and uses actor_critic-appropriate keys instead of PPO-only ones; `algorithm` field set to `"actor_critic"` when `trainer_type == "actor_critic"`; `trainer_type` surfaced as top-level recipe field.
    - New tests: 3 tests for `_last_cleared_rows` / `_pre_clear_board` in `test_tetris_env.py`; 3 tests for actor_critic crystallizer behaviour in `test_crystallizer.py`.
    - Total: **529 tests** (520 unit + 9 integration).

    - Total: **529 tests** (520 unit + 9 integration).

## Phase 18 — Hardcode Removal: Drive All Training Knobs from Recipe

*Goal: Eliminate every magic number from the code generator templates and state machine eval loop. All training hyperparameters are read from `plan["hyperparameters"]` with documented defaults, so the optimizer and recipes have full control without touching code.*

- [x] **Remove hardcoded training constants from templates and eval loop**
    - **Problem**: `_RL_TEMPLATE` hardcoded `total_timesteps=2000000` and `n_calls % 2048`; `_ACTOR_CRITIC_CONTRACT` hardcoded replay buffer size (10000), batch size (512), gamma (0.99), epsilon schedule (min 0.01, decay 0.9995), and 50-episode telemetry window; `_run_goal_metric_eval` hardcoded 10 eval episodes for both trainer paths.
    - `backend/agent/code_generator.py`: all template literals replaced with `{…}` format variables; PPO context adds `total_timesteps` and `telemetry_interval`; AC context adds `replay_buffer_size`, `batch_size`, `gamma`, `epsilon_min`, `epsilon_decay`, `ac_telemetry_interval`; all read from `plan["hyperparameters"]` with sensible defaults. AC telemetry condition changed to cleaner `(episode + 1) % {ac_telemetry_interval} == 0`.
    - `backend/loop/state_machine.py`: `_run_goal_metric_eval` reads `eval_episodes` from `plan["hyperparameters"]` (default 10); both actor_critic and SB3 eval paths use it.
    - `recipes/tetris_ppo_v1.yaml`: added `total_timesteps`, `telemetry_interval`, `episodes`, `replay_buffer_size`, `epsilon_min`, `epsilon_decay`, `ac_telemetry_interval`, `eval_episodes` under `hyperparameters:`.
    - `recipes/snake_ppo_v1.yaml`: added `telemetry_interval`, `eval_episodes`.
    - `recipes/sft_llama_lora_v1.yaml`: restructured from `config:` / `training_params:` to flat `hyperparameters:` block matching what `_build_user_prompt` reads via `**hp`.
    - Deleted 9 stale crystallized recipes (`train_rl_v1–6`, `train_ml_v1–3`).
    - New tests: 4 tests verifying `telemetry_interval`, `total_timesteps`, `ac_telemetry_interval` substitution in `test_code_generator.py`.
    - Total: **533 tests** (524 unit + 9 integration).

## Phase 19 — Snake Feature Observation + Recipe v2

*Goal: Replace the flat 256-element grid observation with a compact 25D hand-crafted feature vector so MLP-based PPO policies have the right inductive bias. The flat grid is too sparse and high-dimensional for an MLP to extract spatial structure efficiently.*

- [x] **`envs/snake_env.py`: add `obs_type` parameter**
    - `obs_type="grid"` (default): existing flat grid — no behaviour change.
    - `obs_type="features"`: 25D compact vector encoding immediate danger (3), 5-step path clearance (3), direction one-hot (4), food direction bits (4), food distance (2), spatial scalars — manhattan distance, snake length, space around head (3), wall distances (4), food accessibility + tail distance (2).
    - `observation_space` is set to `Box(-1, 1, (25,))` when `obs_type="features"`.
    - `_FEATURES_DIM = 25` constant exported for tests.
    - Inspired by the `advanced_dqn_snake` reference implementation (28D state).

- [x] **`recipes/snake_ppo_v1.yaml` v2.0.0**
    - `env_kwargs.obs_type: features` — switches to 25D compact obs.
    - `env_kwargs.max_steps: 2000` — long enough for a 20-food episode (~100 steps/food).
    - `env_kwargs.food_reward: 20.0, death_penalty: -10.0, distance_weight: 0.3, survival_bonus: 0.01` — food signal dominates, no distance-chasing incentive.
    - PPO: `n_steps: 2048, total_timesteps: 3000000`, `net_arch: [256, 256]` — smaller net sufficient with compact obs.

- [x] **Tests**: 7 new tests in `test_snake_env.py` — shape, all-finite, danger detection, food direction bits, observation space membership, grid mode unchanged.
    - Total: **540 tests** (531 unit + 9 integration).

- [x] **`backend/agent/code_generator.py`: `_resolve_env_kwargs` helper**
    - Extracted `_resolve_env_kwargs(env_id, plan_env_kwargs)` — applies per-env defaults before any code is generated.
    - Snake-v0: always injects `obs_type="features"` and `max_steps=2000` via `setdefault`, so plans that omit `env_kwargs` still get the compact obs. Explicit plan values are never overridden.
    - Called from both `_build_user_prompt` (generates `gym.make(...)` kwargs string) and `generate_training_script` (writes `train_config.json`), keeping the two in sync.
    - 4 new tests: `_resolve_env_kwargs` unit tests + prompt-level assertion that `obs_type='features'` appears when plan has no env_kwargs.
    - Total: **544 tests** (535 unit + 9 integration).

- [x] **`backend/agent/code_generator.py`: recipe-driven defaults for all task types**
    - `_ENV_RECIPE` maps `env_id` / `task_type` → recipe filename: `Snake-v0 → snake_ppo_v1.yaml`, `Tetris-v0 → tetris_ppo_v1.yaml`, `sft → sft_llama_lora_v1.yaml`.
    - `_load_recipe_for_env(key)` loads and parses the YAML; returns `{}` on any error so missing recipes are safe.
    - `_resolve_hyperparams(key, plan_hp)` applies recipe `hyperparameters` as `setdefault` fallbacks — plan values win, recipe fills gaps. `_build_user_prompt` calls this at the top for all task types (RL uses `env_id`, SFT uses `"sft"`).
    - `_resolve_env_kwargs(env_id, plan_env_kwargs)` similarly applies recipe `env_kwargs` as defaults. No hardcoded values remain in the generator for known envs.
    - SFT context dict removed its 8 hardcoded fields (`base_model`, `lora_r`, `batch_size`, etc.); they now come from `sft_llama_lora_v1.yaml` via `hp`.
    - Tests updated: replaced hardcoded-value assertions with recipe-value assertions; added `test_resolve_hyperparams_snake_uses_recipe_total_timesteps`, `test_resolve_hyperparams_plan_overrides_recipe`, `test_build_user_prompt_snake_uses_recipe_env_kwargs`.

- [x] **`backend/routers/play.py`: `_snake_viewer_grid` helper**
    - `_run_episode` was sending `obs.tolist()` as the canvas grid; with `obs_type=features` this is 25 floats, not the 256-element grid `SnakePlayer.tsx` expects.
    - Added `_snake_viewer_grid(base_env)` — reads `base_env._snake` and `base_env._food` directly and builds the flat 256-element grid (head=1.0, body=0.5, food=−1.0) regardless of `obs_type`.
    - `_run_episode` now dispatches: Tetris → `_tetris_viewer_grid`, Snake → `_snake_viewer_grid`, else → `obs.tolist()`.
    - 5 new tests in `test_play_router.py`: length=256, head=1.0, food=−1.0, body=0.5, works with `obs_type=grid` too.

- [x] **`backend/loop/state_machine.py`: pass `env_kwargs` to `gym.make` in eval**
    - `_run_goal_metric_eval` called `gym.make(env_id)` without `env_kwargs`, so Snake eval used `obs_type=grid` (256D) against a model trained on `obs_type=features` (25D) — shape mismatch exception caught silently, `best_metric_value` stayed `None`, MetricGap stuck at 0.
    - Fix: load `env_kwargs` from `train_config.json` (same pattern as play router) and pass to `gym.make` for both SB3 and actor-critic eval paths.
    - 1 new test in `test_state_machine_helpers.py`: `test_run_goal_metric_eval_passes_env_kwargs_to_gym_make` — asserts `obs_type=features` is forwarded to `gym.make`.
    - Total: **552 tests** (543 unit + 9 integration).

## Phase 20 — MLX LoRA Fine-Tuning Support

*Goal: Add native Apple Silicon MLX LoRA fine-tuning as a first-class task type alongside RL, SFT, and ML. Uses `mlx_lm.lora` CLI (not HuggingFace) for on-device quantized model fine-tuning.*

- [x] **`recipes/mlx_lora_v1.yaml`**
    - Canonical recipe for MLX LoRA: `gemma-3-12b-it-4bit`, rank=8, scale=20, lr=1e-5, iters=600, `mask_prompt=true`, `grad_checkpoint=true`, `max_seq_length=2560`.
    - Dataset section with `train`/`valid` JSONL paths and `format: prompt_completion`.

- [x] **`backend/agent/code_generator.py`: `_MLX_LORA_TEMPLATE` + wiring**
    - `_MLX_LORA_TEMPLATE`: generates a subprocess script that runs `python -m mlx_lm.lora` with all recipe-driven flags, parses `"Val loss:"` lines from stdout, and POSTs `eval_loss` to telemetry.
    - `_ENV_RECIPE["mlx_lora"] → mlx_lora_v1.yaml` so `_resolve_hyperparams` fills defaults.
    - `_build_user_prompt` `mlx_lora` branch: resolves `train_dataset`/`valid_dataset` from plan `dataset` dict or recipe; pre-computes `--mask-prompt` / `--grad-checkpoint` flags to avoid `.format()` key errors.

- [x] **`backend/agent/lead_agent.py`**: `task_type` enum extended to `["rl", "sft", "ml", "mlx_lora"]`; system prompt updated with `mlx_lora` guidance (dataset top-level field, key HP).

- [x] **`backend/loop/state_machine.py`**: reconciliation already persists LLM-inferred `task_type` to DB, so `mlx_lora` missions are correctly typed after planning.

- [x] **Tests**: 6 new tests in `test_code_generator.py` — base model in prompt, dataset paths, recipe defaults (iters=600), `_resolve_hyperparams` fills recipe, plan overrides recipe.
    - Total: **557 tests** (548 unit + 9 integration).

## Phase 21 — Telemetry Integrity & AC Loop Hardening

*Goal: Fix three related bugs that caused MetricGap to show garbage values, Tetris missions to generate broken hybrid scripts, and AC training to run unbounded episodes.*

- [x] **MetricGap only reflects end-of-iteration eval for custom goal metrics**
    - `_load_persisted_best` scanned all telemetry for `lines_cleared`/`food_eaten`, picking up training-time mean-per-N-episodes posts (mean ~1.3 early in training) and writing them to `best_metric_value` — MetricGap showed 1.3 before any real eval ran.
    - Fix: telemetry scan in `_load_persisted_best` restricted to `mean_reward` only. DB value for custom goal metrics only trusted when `best_metric_iteration is not None` (i.e. `_run_goal_metric_eval` has completed at least once).
    - AC template: removed instruction to post goal metric names (`lines_cleared`, `food_eaten`) during training; only `mean_reward` should be posted as training telemetry.
    - 2 tests updated + 2 new tests in `test_state_machine_helpers.py`.

- [x] **AC training loop bounded by `total_timesteps`, not episode count**
    - LLM was free to pick an arbitrary episode count (chose 50 000), making iterations unbounded and delaying the goal metric eval by hours.
    - Template rewritten: `for episode in range({episodes})` → `while total_steps < {total_timesteps}`; `total_steps` incremented each `env.step()`.
    - `episodes` key removed from context dict and `tetris_ppo_v1.yaml` recipe; `total_timesteps` (default 2 M from recipe) now governs iteration length.
    - 1 new test: `test_build_user_prompt_actor_critic_uses_timestep_loop`.

- [x] **Recipe-driven `trainer_type` fallback in `code_generator`**
    - LLM planner never outputs `trainer_type`, so `code_generator` routed Tetris-v0 to the SB3 template despite `tetris_ppo_v1.yaml` declaring `trainer_type: actor_critic`. Healing attempts generated broken hybrid scripts mixing SB3 imports with `ActorCriticNet`.
    - Fix: `_load_recipe_for_env` called after `_resolve_hyperparams` to read `trainer_type` from the recipe as fallback; plan value still takes precedence.
    - `episodes: 50000` removed from `tetris_ppo_v1.yaml` (superseded by `total_timesteps`).
    - 1 new test: `test_build_user_prompt_tetris_uses_ac_template_without_trainer_type`.

- [x] **AC template: `env = gym.make()` added to skeleton**
    - LLM consistently omitted `env = gym.make("{env_id}")`, crashing every generated script with `NameError: name 'env' is not defined`. The contract showed `env.reset()` in the loop body but never defined `env`.
    - Fix: added `env = gym.make("{env_id}")  # MANDATORY` as the first line of the training skeleton.
    - 1 new test: `test_build_user_prompt_actor_critic_includes_gym_make`.

- [x] **`tetris_ppo_v1.yaml`: `total_timesteps` reduced from 2M to 500k**
    - 2M timesteps at ~100 steps/episode = ~20k episodes, far more than needed for AC Tetris. Agent peaks around ep 5k–9k; the remaining budget only adds noise and delays the goal metric eval by 30–60 minutes.
    - 500k steps ≈ 5k episodes — enough to learn, plateau, and eval, enabling faster pivot cycles.

- [x] **`benchmark._rollout_actor_critic`: duplicate `import sys` removed**
    - `_rollout_actor_critic` had `import sys` at line 46 inside the function body, after already using `sys` at line 37 (via the module-level import). Python's scoping rules made `sys` a local variable throughout the function, causing `UnboundLocalError: local variable 'sys' referenced before assignment` on every eval call — mission failed immediately after training completed.

- [x] **Manifest artifact check accepts `.pth` (actor_critic) alongside `.zip` (SB3)**
    - `req_002` used `checkpoints/*.zip` which only matches SB3 models. Actor-critic missions save `best_model.pth`, so `req_002` permanently failed → `2/3 requirements passed` → mission never completed even when `lines_cleared >= 100`.
    - `manifest_generator`: rl pattern changed to `checkpoints/*.{zip,pth}`.
    - `manifest_evaluator._check_file_exists`: expands `{alt1,alt2}` brace syntax before globbing.

- [x] **Screenshots: Snake-v0 and Tetris-v0 live viewer placeholders added to README**

    - Total: **562 tests** (553 unit + 9 integration).

## Phase 22 — Inline Auto-Approve (No Overnight Stalls)

**Problem:** Approval gates for `EXECUTE_CODE` were only auto-approved when the frontend called `POST /approvals/{gate_id}/auto-approve`. With no browser open overnight, iteration 2 of mission `#81a2b6e7` waited 7 hours blocked on a gate that would have been immediately safe.

**Root cause:** `_request_approval` in the state machine created the gate, then polled the DB every 5 seconds waiting for status change. The auto-approve endpoint was frontend-triggered only — never called by the backend itself.

**Fix:**

- [x] **`backend/services/auto_approver.py` — shared `try_auto_approve()` service**
    - Extracted the auto-approve logic (read script, run `CodeSafetyClassifier`, update gate status) into a standalone async function callable from both the router and the state machine.
    - Returns `AutoApproveResult(action="approved"|"blocked"|"skipped")` for all code paths.
    - 6 new tests in `tests/unit/test_auto_approver.py`.

- [x] **`backend/loop/state_machine._request_approval` — inline auto-approve at gate creation**
    - After creating an `EXECUTE_CODE` gate, immediately calls `try_auto_approve()` using the model manager's code provider.
    - If approved, returns `True` before entering the polling loop — no browser needed overnight.
    - Falls back to polling if blocked, skipped, or an exception is raised.

- [x] **`backend/routers/approvals.auto_approve_gate` — delegates to shared service**
    - No duplicated logic; router retains HTTP-specific validation (404/409/422 errors).

    - Total: **568 tests** (559 unit + 9 integration).

## Phase 23 — Curriculum Training & Algorithm-Aware Code Generation

**Problem:** Snake-v0 PPO missions plateau around food_eaten=72 on a 16×16 grid. The agent has no gradient signal for scores above ~72 because it almost never stumbles into a successful 80+ food episode — the environment is too hard from the start. Separately, DQN missions silently dropped `buffer_size`, `learning_starts`, and other DQN-specific constructor kwargs because `_RL_TEMPLATE` hardcoded `_VALID_PPO_KEYS` and `PPO(...)` regardless of the plan's `algorithm` field.

**Fix:**

- [x] **`recipes/snake_ppo_v1.yaml` — added `curriculum.phases` block**
    - 3 phases: `8×8 (300k steps, target food_eaten=15)` → `12×12 (700k steps, target=40)` → `16×16 (2M steps, target=100)`.
    - Total timesteps unchanged at 3M — budget redistributed across phases rather than all spent on the hardest grid.
    - `obs_type=features` keeps obs at 25D regardless of grid size, so policy weights transfer across phases via `model.set_env()` with no architecture change.

- [x] **`recipes/snake_dqn_v1.yaml` — new DQN recipe for Snake-v0**
    - Replay buffer hyperparams: `buffer_size=100000`, `learning_starts=10000`, `target_update_interval=1000`, `exploration_fraction=0.3`, `exploration_final_eps=0.05`.
    - Same 3-phase curriculum as PPO recipe — curriculum is algorithm-agnostic.
    - Same `env_kwargs` as PPO recipe for environment compatibility.

- [x] **`CodeGenerator._inject_curriculum` — deterministic curriculum post-processor**
    - Called after LLM generation so the loop is deterministic, not LLM-generated (LLMs reliably get `reset_num_timesteps`, `set_env`, and callback state tracking wrong).
    - Replaces `model.learn(total_timesteps=..., callback=callback)` with a `_CURRICULUM_PHASES` loop via regex substitution.
    - Patches `CustomCallback.__init__` to add `self._phase_best_food = 0`.
    - Patches `_on_step` to track `food_eaten` from `self.locals["infos"]` per phase.
    - Sets `reset_num_timesteps=(_ph_idx == 0)` so the step counter is continuous across phases.
    - Applied automatically in `generate_training_script` when the resolved recipe contains `curriculum.phases`.

- [x] **`_VALID_ALGO_KEYS` — per-algorithm SB3 constructor key whitelist**
    - Maps PPO / DQN / SAC / A2C / TD3 → their respective valid `__init__` kwargs.
    - `_RL_TEMPLATE` now uses `{valid_keys_var}` / `{valid_keys_set}` / `{algorithm}` placeholders — import, model constructor, and warm-start load are all algorithm-parameterized.
    - DQN missions now correctly pass `buffer_size`, `learning_starts`, `exploration_fraction`, etc. instead of silently filtering them.

- [x] **`_ENV_RECIPE` — algorithm-specific recipe override**
    - Added `Snake-v0/DQN → snake_dqn_v1.yaml` key; `_load_recipe_for_env(env_id, algorithm)` checks the combined key first, falls back to env-only.
    - `_resolve_hyperparams(env_id, plan_hp, algorithm)` passes `algorithm` through so DQN missions pull defaults from `snake_dqn_v1.yaml`.

- [x] **`recipes/train_rl_v12.yaml` deleted**
    - Stale crystallized recipe written to disk as a side effect of crystallization. Nothing reads recipe files for crystallized missions — they live in DB + ChromaDB. File was dead code.

- [x] **Algorithm-aware pivot filtering**
    - `CodeGenerator.valid_algo_keys(algorithm: str) -> set` classmethod exposes `_VALID_ALGO_KEYS` as a public API.
    - `LoopStateMachine` filters pivot `adjustments` against valid keys for the current algorithm after `_clamp_rl_adjustments` — hard guard drops PPO-specific params (`ent_coef`, `vf_coef`) from DQN pivots and logs a warning.
    - `_PIVOT_SYSTEM` Level 0 description updated to warn against cross-algorithm key mixing.
    - `propose_pivot` user message now includes the exact valid key list for the current algorithm so the LLM knows what to propose.

- [x] **16 new tests in `test_code_generator.py`**
    - `_inject_curriculum`: loop present, phases list, `set_env`, grid dims from phase dict, grid dims excluded from base kwargs, `_phase_best_food` in `__init__`, food tracking in `_on_step`, `reset_num_timesteps` flag.
    - Integration: Snake-v0 script gets curriculum injected end-to-end; CartPole script does not.
    - `valid_algo_keys`: PPO contains `ent_coef`/`vf_coef` not `buffer_size`; DQN contains `buffer_size`/`exploration_fraction` not `ent_coef`/`vf_coef`; case-insensitive; unknown algo returns empty set.

    Total: **582 tests** (573 unit + 9 integration).

## Phase 24 — Sandbox Shutdown Fix + Opt-In PPO Learning Rate Schedule

**Problem:** A codebase review (prompted by the Ray multi-node proposal in `docs/IMPROVEMENT.md`) found `SSHSandbox.terminate()` sent a bare `kill -9` with no graceful step, unlike `SubprocessSandbox.terminate()` (`SIGTERM` → wait(10s) → `SIGKILL`), and that the SSH backend had zero direct unit test coverage. Separately, while investigating a plateauing Snake-v0 PPO mission (food_eaten oscillating 38-52 across iterations 9-15 vs a target of 100), found that PPO's learning rate is a static scalar within a single training run — `PivotEngine` only adjusts it *between* iterations, never decays it during one.

**Fix:**

- [x] **`backend/sandbox/ssh_sandbox.py` — graceful `terminate()`**
    - Now runs `kill -TERM` → polls `kill -0` for up to 10s → `kill -9`, all in a single SSH command, matching `SubprocessSandbox`'s shutdown semantics so the remote training process gets a chance to flush/save state before being force-killed.

- [x] **`tests/unit/test_ssh_sandbox.py` — new file, 12 tests**
    - Covers `launch` (remote dir creation, script transfer, PID/status tracking), `is_alive` (including triggering `_sync_back` on death), the new graceful `terminate` sequence (asserts `kill -TERM` precedes `kill -9` in the remote command string), `get_sandbox_id`, and `_sync_back`.

- [x] **`backend/agent/code_generator.py` — opt-in linear LR schedule for PPO**
    - `_RL_TEMPLATE` now always emits a `_linear_schedule(initial_value)` helper, decaying `progress_remaining` (1 → 0) times the initial LR.
    - Guarded by `if _hp.get("lr_schedule") == "linear" and "learning_rate" in _filtered` — opt-in, not always-on, so DQN/SAC/A2C/TD3 recipes are unaffected unless they explicitly set the flag.
    - `lr_schedule` is a recipe-only key, not in `_VALID_ALGO_KEYS`, so it's automatically excluded from `_filtered` before reaching the SB3 constructor.
    - Composes with, rather than replaces, `PivotEngine`'s across-iteration LR adjustments.

- [x] **`recipes/snake_ppo_v1.yaml` — sets `lr_schedule: linear`**
    - All other recipes (Tetris PPO, Snake DQN, SFT, MLX LoRA) untouched, keep constant LR.

- [x] **5 new tests in `test_code_generator.py`**
    - `_linear_schedule` helper always emitted; opt-in guard stays inert when `lr_schedule` absent from `_hp`; activates when set to `"linear"`; `_resolve_hyperparams("Snake-v0", ...)` picks up the recipe's `lr_schedule: linear` default.

    Total: **598 tests** (589 unit + 9 integration).

## Phase 25 — DPO/GRPO Fine-Tune Task Types + Remote Telemetry Tailing

**Problem:** `ensemble/finetune/dpo_train.py` and `grpo_train.py` are existing, battle-tested DPO/GRPO training scripts for the Ensemble routing model, but they only run manually via SSH+nohup on the Mac Mini — no astra mission orchestration (planning, pivoting, manifest checks). Wiring them into astra hit three real constraints: (1) the training scripts must not be modified or reimplemented — astra can only wrap/dispatch them; (2) neither Mac Mini Python environment (`/usr/bin/python3` or `~/finetune-env`) has `requests` installed, so a wrapper script POSTing telemetry over HTTP from the remote host isn't viable without adding a new dependency there; (3) `SandboxManager`/`code_generator.py` had two places that treated a configured `sandbox_host` as "every mission is now remote," which would have silently rerouted unrelated RL missions (including the live Snake-v0 PPO mission) to the Mac Mini the moment `sandbox_host` was set for fine-tuning.

**Fix:**

- [x] **`recipes/ensemble_dpo_v1.yaml` / `recipes/ensemble_grpo_v1.yaml`** — hyperparameters mirroring `dpo_train.py`/`grpo_train.py`'s actual CLI args (LoRA config, collection/training params for DPO, GRPO's reward-shaping/optimizer params), plus `finetune_dir`/`python_bin` pointing at the Mac Mini's actual paths (`~/finetune`, `~/finetune-env/bin/python` — corrected after verifying via SSH; the project does NOT live under a `PyProjects/ensemble/finetune` mirror there).

- [x] **`_DPO_TEMPLATE` / `_GRPO_TEMPLATE` in `code_generator.py`** — thin wrapper templates instructing the LLM to `subprocess.run` the *existing* training scripts with inherited stdout (no capturing, no `requests`, no network calls at all) — explicitly instructed not to reimplement the DPO/GRPO training loop, matching the existing "deterministic post-processor over LLM-authored complex logic" principle from Phase 23's `_inject_curriculum`.

- [x] **`_ENV_RECIPE`** — added `"dpo"`/`"grpo"` → their recipe files, same pattern as `sft`/`mlx_lora`.

- [x] **`manifest_generator.py`** — added `checkpoints/best/` pattern for `dpo`/`grpo` (matches where `dpo_train.py` actually saves its best adapter — `{save_dir}/best/`).

- [x] **Task-type-scoped remote dispatch, not a blanket `sandbox_host` switch**
    - `SandboxManager.launch()` now takes `task_type`; for `dpo`/`grpo` it forces `backend="ssh"` and **hard-raises `RuntimeError`** if `settings.sandbox_host` isn't configured — no silent fallback to a local backend.
    - `_detect_backend()` no longer treats a configured `sandbox_host` as the general default backend for every mission (previously it did) — decoupled so setting `sandbox_host` for fine-tune pinning can't silently reroute RL/ml missions to the Mac Mini.
    - `code_generator.py`'s `checkpoint_dir` resolution (remote vs local path) is now scoped to `task_type in _FINETUNE_REMOTE_TASK_TYPES`, not a blanket `sandbox_host` check — same reasoning, this was a real bug that would have broken the live Snake-v0 mission's checkpoint path the moment `sandbox_host` was set.

- [x] **New telemetry mechanism: astra pulls, the Mac Mini never pushes**
    - `dpo_train.py`/`grpo_train.py` are untouched — they just `print(f"Pass rate: {pct}% (...)")` to stdout as before, captured by `SSHSandbox`'s existing `nohup ... > remote_log 2>&1` redirect.
    - `SSHSandbox.tail_new_output()` — new method, SSHes in and runs `tail -c +{offset}` against the remote log, returning only bytes written since the last call (tracked via an internal byte offset), with no new deps needed on the Mac Mini.
    - `SandboxManager.tail_new_output()` — passthrough, returns `None` for backends that don't support live tailing (e.g. `SubprocessSandbox`).
    - `backend/services/telemetry_emitter.py`'s new `emit_metric()` — records a metric directly into `telemetry.jsonl` + broadcasts over the WebSocket, the same shape `POST /telemetry/.../metrics` produces, but callable in-process with no HTTP round-trip.
    - `LoopStateMachine._wait_for_sandbox()` now accepts `task_type`; for `dpo`/`grpo` it calls the new `_tail_remote_pass_rate()` helper on every poll tick, regex-matching `Pass rate: ([\d.]+)% \((\d+)/(\d+)\)` against new output and recording each match as a `pass_rate` metric with an incrementing step counter.

- [x] **31 new tests** across `test_code_generator.py` (dpo/grpo prompt content, recipe defaults, wrap-not-reimplement assertions, no-network-calls assertions, checkpoint pattern, sandbox_host task-type-scoping regressions), `test_sandbox_manager.py` (hard-fail on missing `sandbox_host`, forced SSH dispatch, non-finetune task types unaffected, `tail_new_output` passthrough), `test_ssh_sandbox.py` (`tail_new_output` offset tracking), and new `test_remote_telemetry_tail.py` (`_PASS_RATE_RE` matching, `_tail_remote_pass_rate` step increments and exception safety).

**Known gaps at time of writing (resolved in Phase 26 below except where noted):**
- `requests` still isn't installed on the Mac Mini, but is no longer needed for dpo/grpo telemetry — only relevant now if a future task type's wrapper script needs to POST directly.
- Post-training goal-metric evaluation is not yet wired up — **resolved in Phase 26** (`_run_bare_eval`).
- Adapters produced by astra fine-tune missions never get promoted into `retrain_best`/similar named adapters the way `ensemble/finetune`'s own manual workflow does — still manual, not addressed.

    Total: **629 tests** (620 unit + 9 integration).

**Follow-up fixes (same day, verified directly on the Mac Mini):**

- [x] **`subprocess.run(..., cwd=finetune_dir)` added to both templates** (superseded by Phase 26's `os.execv` fix below — cwd handling itself stayed, but the subprocess.run() call was replaced) — `dpo_train.py`/`grpo_train.py` resolve `--prompt-template` and their own hardcoded `EVAL_CASES_PATH` relative to the process's working directory, not the script's location (confirmed against `FINETUNE.md`'s actual `cd ~/finetune && ...` usage). Without this, the wrapper would run from `SSHSandbox`'s remote mission directory and both relative-path loads would silently fail. `routing_only_flag` context value changed from a bare `--routing-only` string (shell-style) to `'"--routing-only",'` (Python-list-element style) to match the new list-form call.
- [x] **`ensemble_grpo_v1.yaml`'s adapter default fixed** — was copied verbatim from `grpo_train.py`'s own default constant (`finetune/adapters/retrain_best`), which assumes a different cwd than the one now set. With `cwd=finetune_dir`, that value would have resolved to a nonexistent double-nested path. Corrected to `adapters/retrain_best`, verified against the actual Mac Mini directory listing.
- [x] **Checkpoint/adapter output relocated to `finetune_dir/adapters/astra_<mission_id[:8]>/`** — previously used the generic `sandbox_data_path/missions/<id>/checkpoints/` path. Changed so astra-produced adapters land in the same directory `ensemble/finetune`'s manual workflow already uses (`grpo_v<N>_min/`, `dpo_v<N>_min/`, `retrain_best/`, ...), keeping them part of the normal adapter inventory rather than a separate astra-only location. New shared helper `code_generator.finetune_checkpoint_dir(task_type, plan, mission_id)`, used by both `generate_training_script` (bakes `--save-dir` into the wrapper) and `state_machine.py` (passes the same path to `SandboxManager.launch(remote_checkpoint_dir=...)`).
- [x] **`SandboxConfig.remote_checkpoint_dir` + `SSHSandbox._sync_back()` override** — since the adapter now lives outside the generic `{remote_mission_dir}/checkpoints/` path, sync-back needed to know the real source directory; previously it would have rsynced from the wrong (empty) location and the manifest's checkpoint-exists check would never pass for dpo/grpo missions.
- [x] **`.env.example` updated** — documents that `ASTRA_SANDBOX_HOST` is required (hard-fails, no fallback) for `dpo`/`grpo` but doesn't affect other task types, and that `ASTRA_SANDBOX_PYTHON` must point at a path that exists on the remote host (plain `python3` is enough — the wrapper only needs stdlib).
- [x] **7 new tests**: `finetune_checkpoint_dir` path construction, cwd present in both templates, `routing_only_flag` list-element format, `SSHSandbox._sync_back()` honoring `remote_checkpoint_dir` when set vs. the default.

    Total: **633 tests** (624 unit + 9 integration).

## Phase 26 — DPO/GRPO Hardening: Goal-Achievement Check, Orphan-Proof Dispatch, Recovery Consistency, Training-Signal Metric, Collection Progress, Auto-Approve

**Problem:** Phase 25 shipped dispatch and live telemetry for dpo/grpo, but a full line-by-line comparison against `docs/FINETUNE.md` (every hyperparameter, the actual GRPO/DPO run-history tables, `bare_eval.py`'s role) surfaced several gaps that would have broken or silently degraded a real mission: (1) recipe hyperparameters had been copied from the training scripts' raw CLI defaults rather than the documented current best-practice values — most seriously `grpo.num_layers=16` against 8-layer warm-start adapters, a hard crash on load; (2) `LeadAgent`'s plan schema restricted `task_type` to `rl/sft/ml/mlx_lora`, so the planner could never actually keep a mission as `dpo`/`grpo` — the existing task_type reconciliation step would silently overwrite it; (3) there was no post-training goal-achievement path at all for these task types — `_run_goal_metric_eval` is RL-only (expects a Gym env + SB3/`.pth`/`.zip` checkpoint), so the manifest's `pass_rate` requirement could never be satisfied no matter how well training went; (4) the wrapper script used `subprocess.run()`, forking a child process for the actual training — if the wrapper ever died, its child (the real work) was silently orphaned, invisible to astra; (5) state recovery on backend restart had no equivalent to the local `psutil.pid_exists()` check for SSH-dispatched missions, so it could never detect or clean up a live remote process, unlike local missions which get a real kill-and-clean-restart; (6) dpo/grpo missions only tracked `pass_rate` — updated sparsely (every `--steps-per-eval` steps) — with no equivalent of RL's continuously-updating `mean_reward` training signal, so `MetricChart`'s history showed nothing useful between checkpoints.

**Fix:**

- [x] **Recipe hyperparameters corrected to documented best practice, not raw script defaults** (`ensemble_grpo_v1.yaml`): `num_layers` 16→**8** (critical — matches warm-start adapter, previously would crash on LoRA shape mismatch), `adapter` `retrain_best`→`grpo_v9_min/best` (the documented current warm-start target, not a mutable pointer), `iters` 300→100, `max_tokens` 256→96, `email_weight` 3→0, `num_generations` 4→2 (matches the actual best-performing run, `grpo_v9_min`). Also added `--save-pairs` to the DPO wrapper (previously missing entirely) so a future pivot can reuse collected pairs via `--load-pairs` instead of re-running the slowest phase (~30-60 min) from scratch.

- [x] **`LeadAgent`'s `_PLAN_SCHEMA` and `_PLANNING_SYSTEM`** (`lead_agent.py`) — added `"dpo"`/`"grpo"` to the `task_type` enum, and instructed the LLM to leave `hyperparameters` empty `{}` for these task types (the recipe supplies everything; a guessed adapter path or mismatched `num_layers` crashes the run). Without this, `LoopStateMachine`'s existing task_type reconciliation (mission always starts as the default; corrected to whatever the plan says at iteration 0) would have silently discarded `dpo`/`grpo` in favor of whatever the schema-constrained plan picked instead.

- [x] **`LoopStateMachine._run_bare_eval()`** — new post-training authoritative goal-metric check for dpo/grpo, analogous to `_run_goal_metric_eval` for RL. Runs `ensemble/finetune/bare_eval.py --adapter adapters/astra_<mission_id[:8]>/best --prompt-template ...` over SSH on the Mac Mini (~12 min per docs) — the actual adapter-discriminating eval tool (`run_eval.py` is saturated and doesn't distinguish between adapters, per `FINETUNE.md`). Parses the same `Pass rate: X% (n/total)` format via the existing `_PASS_RATE_RE`, feeds the result into `current_metrics[metric_name]` so `ManifestEvaluator` can finally mark the mission complete. Wired into the main evaluation flow: `task_type in _FINETUNE_REMOTE_TASK_TYPES` branches to `_run_bare_eval` instead of `_run_goal_metric_eval`.

- [x] **Orphan-proof wrapper: `os.execv` instead of `subprocess.run`** (`_DPO_TEMPLATE`/`_GRPO_TEMPLATE`) — the generated wrapper now does `os.chdir(finetune_dir)` then `os.execv(python_bin, [python_bin, script, ...args])`, which **replaces** the wrapper's own process image (no fork) rather than spawning a child. This means the pid `SSHSandbox` tracks is *always* the actual training process, not a parent that can die independently and orphan its child — confirmed against a real incident where exactly this happened (the wrapper process died for an unrelated reason mid-run, the actual `dpo_train.py` child kept running, reparented to pid 1, completely invisible to astra's tracking).

- [x] **`get_sandbox_id()` passthrough on `SandboxManager`** + improved "Sandbox running" log message — SSH-dispatched missions now show `host=mac-mini.local remote_pid=14516` instead of a confusing `pid=None` (the Mission Store's actual `subprocess_pid` field, used by local recovery, is untouched — this only improves the HUD/log display).

- [x] **Consistent state-recovery for SSH-dispatched missions** — new `remote_pid` column on `Mission` (migration `d4e5f6a7b8c9`), persisted after SSH launch. `SandboxManager.recover()` now accepts `remote_pid` and checks it via `ssh {host} "kill -0 {pid}"` — the exact same liveness semantics `psutil.pid_exists()` provides locally. If alive, reconstructs an `SSHSandbox` and reattaches it so `state_recovery.py`'s existing `terminate()` call actually reaches the remote process (graceful `SIGTERM`→`SIGKILL`, matching local behavior) instead of silently doing nothing, which is what happened before this fix (recovery had no pid/container_id to check for SSH missions, fell through to `"dead"` without verifying, and never called `terminate()` — leaving the real process to be discovered only by manual `ssh`+`ps` inspection).

- [x] **New `loss` training-signal metric** — `grpo_train.py` prints `Step {n}/{iters} | loss={avg_loss:.4f} | baseline={reward_ema:.3f} | ...` every `--steps-per-report` step (a real, frequent signal); `dpo_train.py` only prints `=== Epoch {n}/3 done avg_loss={avg_loss:.4f} ===` once per epoch (sparse — 3 points total for the default 3 epochs, but it's what the script provides — same precedent as ml/sft tracking `eval_loss` at whatever cadence the training loop allows). New `_GRPO_LOSS_RE`/`_DPO_LOSS_RE` regexes in `state_machine.py`; `_tail_remote_pass_rate()` renamed to `_tail_remote_metrics()` (now takes `task_type` to pick the right regex) and emits `loss` using the script's own reported step/epoch number (not a local counter, since it's meaningful on its own — unlike `pass_rate`'s counter). `MetricChart.tsx`'s "exclude goal metric, show training signal" logic generalized from `"mean_reward"`-only to `TRAINING_SIGNAL_NAMES = ["mean_reward", "loss"]` — this also surfaced and fixed a latent bug: the y-axis was hardcoded to `[0, 1]` (percentage-style) whenever the *goal* metric's target was `<=1` (true for `pass_rate: 0.85`), which would have clipped `loss` values (0.3–3+ range) off the chart. Now the axis adapts to the actual displayed signal (`excludeGoalMetric || isRaw`) rather than assuming the excluded goal's own scale applies to whatever's shown instead.

- [x] **28 new tests**: recipe hyperparameter regressions (`num_layers=8`, warm-start adapter, `iters`, `max_tokens`, `email_weight`), `LeadAgent` schema/prompt content, `test_bare_eval.py` (6 tests — pass-rate parsing, SSH command construction, missing-config/exception handling), `os.execv`-not-`subprocess.run` assertions for both templates, `get_sandbox_id` passthrough, `remote_pid` recovery (reattach-when-alive, dead-when-gone, dead-when-host-unconfigured, terminate-reaches-the-real-pid) across `test_sandbox_manager.py` and `test_state_recovery.py`, plus 6 more in `test_remote_telemetry_tail.py` for `_GRPO_LOSS_RE`/`_DPO_LOSS_RE` matching, `loss` metric emission with the script's own step/epoch number, task-type-correct regex selection, and `pass_rate`+`loss` both emitted correctly from the same tail call.

**Verified against real live incidents, not just tests:** launched an actual DPO mission end-to-end, four attempts. First run: wrapper died mid-collection (root cause not fully determined — log rotated away, but likely coincided with an unrelated backend restart), orphaning the real `dpo_train.py` process — exactly the failure mode the `os.execv` fix targets; manually killed the orphan, relaunched with the fixed wrapper for a clean second attempt (confirmed: single pid, no fork). A later restart (`make stop`'s `kill -9`, not graceful) again left the remote process dead with `state_recovery.py` reporting "no interrupted missions found" — investigated directly via SQLite: `remote_pid` was confirmed correctly persisted for the live process, and `run()`'s existing `except asyncio.CancelledError` handler already calls `sandbox.terminate()` generically on graceful shutdown (an initial suspicion of a shutdown-handler gap was checked and ruled out) — so the recovery mechanism itself is confirmed working; the specific prior incident's root cause remains unconfirmed (the DB row had already been overwritten by a subsequent relaunch before it could be inspected further). A third attempt crashed with `AttributeError: 'list' object has no attribute 'get'` in `grpo_train.py`'s `score_completion()` (called from `dpo_train.py`) — the model occasionally samples a top-level JSON array instead of an object at `temp=1.2`, and `score_completion` assumed `plan` was always a dict; fixed directly in `ensemble/finetune/grpo_train.py` with an `isinstance(plan, dict)` guard (ensemble's own repo, outside astra — applied and synced to the Mac Mini by the user). The fourth attempt cleared the exact case index where the third had crashed and proceeded cleanly through the rest of pair-collection.

Watching that same live mission end-to-end also surfaced two more friction points not caught by Phase 25/26's testing, fixed in the same pass: (1) `collect_pairs()` is typically the bulk of the mission's wall-clock time — over an hour for 66 cases × K=8 — and printed nothing `_tail_remote_metrics` understood, so the HUD showed "NO METRICS YET" the entire time with no way to tell a healthy run from a stuck one; (2) every dpo/grpo mission's generated dispatch wrapper needed a manual approval click on every single attempt — the static safety pre-filter has no rule for `os.execv`, so a script with zero dangerous patterns and no network calls still fell through to `static_ambiguous` and round-tripped to the LLM classifier, which inconsistently flagged legitimate `os.execv` process-image replacement as "could lead to arbitrary code execution."

- [x] **Pair-collection progress surfaced as a status event, not a metric** (`state_machine.py`) — new `_COLLECT_PROGRESS_RE` matches `dpo_train.py`'s `"  [i/total]  n pairs  (Ts)"` line; `_tail_remote_metrics()` emits the latest match via `emit_status()` (the same INF/OK/WARN event-stream channel already used for "Sandbox running", "Generating training plan…", etc.) rather than `emit_metric()` — its 0..66 case-count scale has nothing to do with `loss`/`pass_rate` and would have distorted `MetricChart`'s y-axis if plotted as a series. 3 new tests in `test_remote_telemetry_tail.py`.

- [x] **Static auto-approve short-circuit for the dpo/grpo dispatch wrapper** (`code_safety_classifier.py`) — added a deterministic rule alongside the existing "all requests target localhost" short-circuit: if the script has no `requests` calls (already established safe/absent by the existing checks) *and* contains `os.execv(...)` targeting `dpo_train.py`/`grpo_train.py` by name, it's auto-approved statically without ever reaching the LLM classifier. This is exactly and only the shape `_DPO_TEMPLATE`/`_GRPO_TEMPLATE` produce — `os.execv` here is legitimate process-image replacement (see the orphan-proofing fix above), not arbitrary code execution.

- [x] **`MetricChart.tsx`'s empty state shows collection progress** — when no `pass_rate`/`loss` data exists yet, the chart now surfaces the latest "Collecting preference pairs: ..." status event instead of a bare "NO METRICS YET" placeholder, so the panel reflects real progress during the (often 1hr+) collection phase.

- [x] **Fixed a real `UnicodeDecodeError` crash in `_wait_for_sandbox`** (`state_machine.py`) — a live DPO mission crashed with `'utf-8' codec can't decode byte 0x96 in position 0` right after training completed successfully (confirmed real checkpoints existed on disk matching the run's own reported 74.2% pass rate). Root cause: the training log contains tqdm's multi-byte block character (`█` = `\xe2\x96\x88` in UTF-8) from a progress bar, and `f.seek(log_offset)` landed exactly mid-character, so decoding started on the `0x96` continuation byte. Fixed with `open(log_path, "r", errors="replace")` — a torn byte at a seek boundary is now tolerated instead of killing an otherwise-successful mission one step before its official `_run_bare_eval` goal-check could run and record the result. 1 new regression test (`test_wait_sandbox_torn_utf8_at_seek_offset_does_not_crash`) reproduces the exact byte sequence and confirmed to fail without the fix.

- [x] **Metric Gap has a crash-safe fallback for dpo/grpo, without updating mid-training** — Metric Gap (`Mission.best_metric_value`/`current_metric_value`) updates exactly once per *completed* iteration, exactly like RL's `food_eaten`: never live during training, only from the official post-training `_run_bare_eval` check. `LoopStateMachine._tail_remote_metrics()` separately tracks the best pass_rate seen so far in memory only (`self._live_pass_rate_best`, keyed by mission, reset at the start of each iteration in `_wait_for_sandbox`) — this is never written to the DB directly. If `_run_bare_eval` fails (missing checkpoint, transient SSH error, or a crash like the `UnicodeDecodeError` above), `run()`'s evaluate step falls back to `self._live_pass_rate_best.pop(mission_id, None)` — consumed once and cleared, so a stale value can't leak into the next iteration. This means a mission whose training completed successfully but whose official eval check failed still records its real, already-observed progress, while a mission still mid-training continues to show "NO METRICS YET" until its current iteration actually finishes — matching RL's behavior exactly. 8 tests in `test_remote_telemetry_tail.py` cover the in-memory tracker (keeps the higher value, resets per iteration) and the pop-based fallback-consumption semantics.

- [x] **Live-tailed pass_rate/loss events are tagged with the real mission iteration, not a local counter** — `MetricGap.tsx`'s sparkline (the small per-iteration "pass_rate history" chart) buckets telemetry events by their `iteration` field and keeps the max value seen per bucket. `_tail_remote_metrics()` now receives `current_iteration` (threaded through from `run()` via `_wait_for_sandbox()`) and tags every `pass_rate`/`loss` event with it, while still using the separate local `pass_rate_step`/script-reported step number for the chart's x-axis (`step`, unchanged) — these are two distinct concerns and must not be conflated: `step` says *where in this run's own progress* a reading happened; `iteration` says *which mission iteration* it belongs to. 3 tests cover this (explicit `current_iteration` vs `step` distinction, and the full `_wait_for_sandbox` → `_tail_remote_metrics` threading path).

- [x] **`MissionUpdate` schema accepts `best_metric_iteration`/`current_metric_value`** (`schemas/mission.py`) — the `PATCH /missions/{id}` endpoint previously only accepted `best_metric_value`, with no way to backfill the other two Metric-Gap-related fields without direct DB access. Used to backfill mission `e16f4f37`'s stale `null` fields (from before the live-tracking fix above existed) with its actual observed values (`best_metric_value=0.742`, `current_metric_value=0.697`) via the API rather than a raw SQL update.

- [x] **`CodeSafetyClassifier` regex robustness fixes** — the eval/exec danger-pattern regexes false-matched `model.eval()`/`mx.eval(...)` (standard PyTorch/MLX idioms) and English words appearing inside print/log strings and comments (e.g. `"--- Baseline eval (step 0) ---"`) — both confirmed as real false positives when reviewing ensemble's actual training scripts. Fixed generally via `_strip_strings_and_comments()` (uses the real `tokenize` module, reconstructing exact source spacing so adjacency-sensitive checks like the `model.eval()` exclusion still work) rather than adding more regex exceptions on top of regex exceptions. Separately, `_strip_leading_docstring()` (uses `ast`, since only the AST reliably identifies "first statement is a bare string" vs. any other string literal) removes a module-level docstring before truncating a script to 2000 chars for the LLM — a long "Usage: nohup python ..." example in a header docstring was dominating that truncated slice, causing the LLM to misread a documented shell-invocation example as a real dangerous action. Both fixes benefit any script classified, not just dpo/grpo's.

- [x] **`CodeSafetyClassifier` prompt clarifies `os.chdir`/`os.execv`/fine-tuning-project paths as safe** — three separate live incidents where the LLM classifier flagged the dpo/grpo dispatch wrapper (`os.chdir` into the finetune dir, then `os.execv` with mission-specific args, then file operations under `~/finetune/adapters/...`) as unsafe, each for a different reason. Rather than a deterministic static bypass (considered, but not adopted — unlike RL's localhost-telemetry rule, this wrapper's own text doesn't show what will actually execute, since it delegates via `os.execv` to a separate file not included in what's classified) or a "remembers prior approvals" precedent mechanism (also considered, also not adopted — not consistent with how any other task type's approval works), the fix is confined to the LLM's own system prompt: explicit clarifications that `os.chdir`/`os.execv` are safe process-control operations, and that operating under a *related* fine-tuning project's own directory (paths containing "finetune"/"ensemble") is expected, not "outside the project." Every dpo/grpo `EXECUTE_CODE` gate still goes through the LLM fresh, every time — a deliberate workflow-consistency choice, not a shortcut.

- [x] **`_request_approval()` reports HOW a gate was approved** — event-stream logs previously showed a generic "Execution approved" for both an inline classifier resolution and a human clicking Approve, indistinguishable from each other. `_request_approval()` now returns `(approved: bool, is_auto: Optional[bool])` — a typed flag, not a formatted string — derived from the existing `reviewer_note` field's `"[auto-approved]"` prefix (no new data, no schema change). The call site alone builds the display text (`"auto-approve"` vs `"manual-approve"`), keeping presentation formatting out of the method that resolves approval state.

- [x] **`CodeSafetyClassifier`'s LLM call token budget was too tight** — `max_tokens=64` for a `{"safe": ..., "reason": "..."}` verdict meant a longer "reason" got cut off mid-string, producing invalid JSON (`Unterminated string...`) and forcing every such script to manual review regardless of what the LLM's actual verdict would have been — a live incident showed a gate stuck on "LLM classifier unavailable — manual review required" (the exception fallback, not a real safety judgment). Bumped to 128.

    Total: **689 tests** (680 unit + 9 integration).
