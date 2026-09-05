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
    - **Level 4** (`count ≥ 15`, `ESCALATION_FORCE_NOVEL`): deep plateau — force a never-before-tried architecture, rejecting any proposal from the full mission history.
  - `_pivot_count` is persisted to the `missions.pivot_escalation_count` DB column after every pivot and restored on server restart, so escalation survives process crashes and restarts.
  - **Convergence stop** (`is_converged()`, Phase 36): once escalation is already maxed (`_pivot_count ≥ ESCALATION_FORCE_NOVEL`) **and** the all-time best has not improved for `STALL_ITERS_WITHOUT_BEST=30` distinct *evaluated* iterations (`iters_since_best()` — loops that produced no goal metric don't count), the search has run out of moves. The state machine saves the best metric, crystallizes the lessons (skipped for `dpo`/`grpo` — see below), and transitions the mission to the terminal `MissionStatus.STALLED` (not resumable without an explicit status reset), keeping the best checkpoint instead of burning compute indefinitely on an unreachable target. Minimum ~75 iterations before it can fire. Real incident: DPO mission `ce2828f4` ran 600+ iterations / 20 days stuck at best 0.833 vs target 0.85, every `dpo` pivot a structural no-op, with nothing to stop it.
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
The worker agents that interface with diverse training paradigms. `task_type` (`rl` / `sft` / `ml` / `mlx_lora` / `dpo` / `grpo` / `distill`) selects which one a mission uses; LoRA is a *mechanism* (efficient low-rank weight updates), not a task type of its own — it underlies `sft`, `mlx_lora`, and the three fine-tune types below.

| Task type | Objective | How it trains |
|---|---|---|
| `rl` | Classical reinforcement learning — an agent learns a policy from trial-and-error reward signal in an environment (Snake-v0, Tetris-v0, or standard Gymnasium envs). | SB3 (PPO/DQN/A2C/SAC/TD3) via `RLTrainer`, or a custom Actor-Critic / lookahead-augmented trainer for Tetris-v0 (Phase 17, Phase 29/31). |
| `sft` | Supervised fine-tuning — adjust a model's outputs toward labeled examples, the standard first fine-tuning step before any preference-based method. | HuggingFace Transformers + PEFT (LoRA/QLoRA) via `SFTTrainer`. |
| `ml` | Classical (non-neural or lightly-neural) machine learning on tabular data. | Scikit-learn / PyTorch Lightning via `MLTrainer`. |
| `mlx_lora` | LoRA fine-tuning on Apple Silicon via MLX, for local/offline workloads. | `mlx_lm.lora` subprocess wrapper (Phase 20). |
| `dpo` | Direct Preference Optimization — given pairs of (chosen, rejected) responses to the same prompt, directly shift probability mass toward the chosen one, without a separate reward model or RL rollout loop. | Wraps `ensemble/finetune/dpo_train.py`: samples `k_collect` candidate completions per prompt at `temp`, ranks them, trains a LoRA adapter on the resulting pairs against a frozen reference policy (Phase 25/26). |
| `grpo` | Group Relative Policy Optimization — on-policy RL without a learned critic: sample a *group* of `num_generations` completions per prompt, score each, and use the group's own mean as the baseline for the policy-gradient update. | Wraps `ensemble/finetune/grpo_train.py`. Used to produce the `grpo_v9_min` checkpoint that `dpo` warm-starts from in this project. |
| `distill` | Knowledge distillation — a strong teacher (`conductor_gemma.md` + `gemma3:12b` via Ollama) generates correct routing completions; the weak student (`conductor_min.md` + `gemma-3-12b-it-4bit`) is SFT'd to imitate them, then scored by the same routing oracle. The lever `dpo`/`grpo` couldn't move: unlike preference/RL fine-tuning, SFT from a teacher isn't bounded by the converged model's overfitting dynamic (Phase 42). | Wraps `ensemble/finetune/distill_train.py` (teacher-gen → `mlx_lm.lora` → routing eval). Warm-starts from `retrain_best` (the SFT lineage), pivot varies only `iters`. The wrapper script is an ensemble-side precondition — a `distill` mission fails at launch until it exists. |

**Why `dpo`/`grpo`/`distill` pivots are more constrained than `rl` pivots:** an RL pivot can safely retune hyperparameters, swap architecture, or reshape rewards — the training script builds everything from scratch each time. A fine-tune-remote mission trains a LoRA adapter warm-started from a *specific* checkpoint (`num_layers`/`lora_rank` must match it exactly, or loading crashes) with hyperparameters (`learning_rate`, `beta`, etc.) already tuned against that checkpoint — a generic pivot-proposed learning rate destroyed a run once (Phase 26). So the recipe stays authoritative for everything except a tiny per-task-type safelist: `temp`/`k_collect` (dpo), `temp`/`num_generations` (grpo) — preference-pair sampling diversity — or `iters` (distill — train longer/shorter). All are clamped to their prompt-declared ranges; anything else the LLM proposes is dropped before it reaches the stored plan or the pivot telemetry (Phase 33, Phase 39, Phase 42).

**Recipe `metric_ceiling` and result flooring (Phase 36):** because a `dpo`/`grpo` mission can only vary those three sampling knobs, once the underlying model is converged there is no lever left to close the gap to an over-ambitious target. Two guardrails: (1) the fine-tune recipes declare a `metric_ceiling` (the empirically-observed best for that recipe + model + eval set — `dpo` 0.85, `grpo` 0.83, `distill` 0.92 as of 2026-09-05, on the blended all-78-case scale; see the eval-oracle-split and "What `pass_rate` measures" notes below), and `POST /missions` rejects (422) any target that exceeds it beyond a small noise margin. (2) Each iteration, `_dpo_run_diagnostics()` parses the training log for the warm-start adapter's own pre-training `Baseline:` score and the `total_steps` count; if the run trained for fewer steps than `steps_per_eval` (so `dpo_train.py`'s in-training best-checkpoint tracker never fired and only the overfit final adapter exists to score), or the scored `pass_rate` lands more than `DPO_BASELINE_FLOOR_MARGIN=0.03` below that baseline, the value fed to `PivotEngine` / the DB is floored to the baseline (the true value still goes to telemetry). The warm-start adapter file is never modified by training, so the mission genuinely still has that score — flooring stops the loop recording false regressions and chasing noise below its own starting point, while plateau-at-peak still drives escalation toward the convergence stop.

**Static-skill vs dynamic-MCP routing in the eval oracle (Phase 39 follow-up, `ensemble` repo `3ab5667`):** the `pass_rate` telemetry astra parses comes from `dpo_train.py`/`grpo_train.py` scoring `ensemble/backend/core/eval_cases.yaml`. That oracle mixes two populations: **static-skill routing** (71 cases — `expected_skill` is a registry skill the training/serve prompt lists) and **dynamic-MCP routing** (7 cases — `expected_skill` is a runtime `mcp:Server:tool` string deliberately absent from the prompt). The model is never shown the MCP strings, so those cases fail every rollout, produce no preference pairs, and previously dragged the reported rate down ~0.10 (7/66) — the direct cause of DPO missions plateauing ~0.02 below a 0.84 target forever. `grpo_train.is_dynamic_mcp_case()` now excludes them from the training pool and pass-rate by default (`--include-dynamic-mcp` overrides); telemetry `Pass rate:` lines are over the 71-case static set (12 teacher-verified cases were also added for previously starved skill classes). On that corrected set the warm-start adapter `grpo_v9_min/best` already scores **0.859 (61/71) before any DPO** — the old 0.833 "ceiling" was largely the MCP artifact — so the `dpo` recipe's `metric_ceiling` was moved to 0.87 — just above the 0.859 warm-start, leaving room for the newly-added starved-class cases (`grpo` held at 0.84). It is not higher because DPO had regressed this converged checkpoint on every evaluated iteration across four missions. The mechanical cause — `dpo_train.py`'s `dpo_loss` using **mean**-token log-prob, which shrank the chosen−rejected margin by completion length and left a near-zero gradient at β≈0.1 — is now fixed and deployed `ensemble`-side (`7b9f45a`, summed sequence log-likelihood; the recipe `learning_rate` was dropped 2e-6 → 3e-7 to match the ~length× larger gradient, pending an offline re-tune). The remaining gap is **no held-out split**: pairs and `best/`-checkpoint selection both run over the same 71 cases, so a DPO `best/` is overfit-to-metric. Making the mission metric a genuine test-set pass-rate is a coordinated `ensemble` + astra change (still open); until it lands, 0.87 is a placeholder and a DPO `best/` checkpoint should not be trusted. This does not change any astra code — `_PASS_RATE_RE` already reads the reported `%` directly, denominator-agnostic. Closing the MCP gap needs the tools surfaced in the routing prompt, a separate project.

**What `pass_rate` measures (Phase 43, 2026-09-05):** the goal metric is the **blended all-78-case** number, and a recipe's `metric_ceiling` is on that blended scale. This reverses the Phase 39 decision recorded above, which made the 71-case static split the selector. Excluding the MCP cases from the *metric* silently excluded them from the *objective*: a run could destroy a capability the base model had and the metric would record a clean win. Measured 2026-09-05 across the three production-relevant configs:

| config | static (71) | MCP (7) | blended (78) |
|---|---|---|---|
| raw 4B + `conductor_gemma.md` (**shipped**) | 59 (83.1%) | 3 | **62 (79.5%)** |
| `grpo_v9_min/best` 12B + `conductor_min.md` | **61 (85.9%)** | 0 | 61 (78.2%) |
| `distill` iter2 12B + `conductor_min.md` | 58 (81.7%) | 3 | 61 (78.2%) |

Static-only ranks `grpo_v9_min/best` top and rejects `distill` iter2 as a 3-case regression; blended shows iter2 traded 3 static cases for 3 MCP against its own warm-start — a wash. Static-only cannot see that trade because it excludes exactly the cases that moved, and it cannot see that the fine-tuned 12B loses to the raw 4B already in production once all 78 count. `_run_bare_eval` therefore reads `"Blended (all cases):"` (`_BARE_EVAL_BLENDED_RE`); static and MCP are recorded as `pass_rate_static` / `pass_rate_mcp` telemetry diagnostics that no decision reads back, so an MCP-for-static trade is visible in the HUD instead of hiding inside a flat blended line. A split report missing its blended line returns `None` rather than falling back to the static line — that fallback would reinstate the old scale against a blended-scale target. Ceilings were rebased, **not re-tuned**: `dpo` 0.87 → 0.85, `grpo` 0.84 → 0.83, `distill` 0.95 → 0.92; the old static-scale values would have become near-unreachable targets. `bare_eval.py`'s own labels still call the static line "fine-tune target metric" and blended "not the selector" — those labels are backwards and are what taught this project to read static as authoritative.

A floored iteration is a *non-result*: it never sets a new all-time best and never chains its checkpoint forward (Phase 39). The floored value belongs to the warm-start, not to that iteration's overfit adapter — chaining it anyway once anchored a mission to a regressed checkpoint while the DB showed a best it could not reproduce (mission `15a1d093` iter 0). When the very first evaluated iteration regresses (no prior best to protect) the *raw* value is recorded rather than the baseline, so no unreproducible best is fabricated; checkpoint chaining is gated on this iteration's genuine (pre-floor) `pass_rate` matching or beating the prior best. The pivot proposer is also hard-restricted for every fine-tune-remote type to its per-task-type safelist (above); any `env_kwargs` / `policy_kwargs` / `algorithm` the LLM proposes is dropped before it can reach the stored plan or the pivot telemetry. `distill` (Phase 42) inherits all of this — the baseline floor, floored-iteration-is-a-non-result, checkpoint chaining, and no crystallization — plus a `metric_ceiling` of 0.92 (generous: SFT from a teacher is not bounded by the converged-model overfitting dynamic that caps `dpo`/`grpo`, and distillation is the one method observed to *recover* MCP cases). One difference: `distill_train.py` trains on a train split and reports `pass_rate` against a **held-out** split, so a `distill` mission's goal metric is read from the training log's `"Best pass rate during training:"` line (`_distill_held_out_metric`) — the same population the floor's `"Baseline:"` line uses — rather than a full-set `bare_eval.py` run, which would leak training cases and report a different number (that mismatch doom-looped mission `b790c69d` for ~9h before the fix).

- **Universal Code Generator**: LLM-driven generation of the actual training script for each task type above, from a recipe + plan/pivot.
- **Framework Wrappers**: Standardized interfaces for common libraries (Transformers, SB3, PyTorch, MLX).
- **Telemetry Producer** (also referred to as the Telemetry Streamer in IMPLEMENT): Streams paradigm-specific metrics via WebSocket (e.g., Reward for RL, Perplexity for SFT, Accuracy/F1 for ML, pass_rate/loss for DPO/GRPO). On recovery, back-fills missed logs from the `data/` volume to the HUD.

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
7. **Crystallization**: If the goal is met (or the mission stalls), the system distills the final, optimized strategy into a new **Recipe** and saves it to the Library — **except for `dpo`/`grpo` missions**, whose training dispatch is hardcoded to a canonical recipe (`_ENV_RECIPE` in `code_generator.py`: `dpo → ensemble_dpo_v1.yaml`, `grpo → ensemble_grpo_v1.yaml`). A crystallized recipe for those task types can never be loaded for dispatch, so `LoopStateMachine._crystallize()` skips them (`_NO_CRYSTALLIZE_TASK_TYPES`) rather than accumulating orphan recipe files, DB rows, and vector-index entries.
8. **Finalization**: Registry update and report generation.

## 4. Security & Autonomy Gates

### 4.1. The Approval Controller
`GateType` (`backend/models/approval.py`) models three gate types (`EXECUTE_CODE`, `RESOURCE_ALLOCATION`, `DEPLOY_MODEL`), but only one is actually wired into the loop today:
- **Gate: `EXECUTE_CODE`**: Pauses the loop and presents the generated script to the user for a "Safety Check." The `CodeSafetyClassifier` runs a two-stage pre-screen: (1) a static regex pass that immediately approves scripts whose only network calls target `localhost`/`127.0.0.1` (telemetry), and immediately blocks known-dangerous patterns (subprocess shell injection, broad file deletion, external pip installs); (2) an LLM classification pass for ambiguous cases. Only genuinely risky scripts reach the human approval queue.
- **Gate: `RESOURCE_ALLOCATION`** / **`DEPLOY_MODEL`**: modeled in the schema, but no code path currently creates either — not implemented yet, despite being defined as gate types.

### 4.2. Autonomy Tiers
`Mission.autonomy_mode` supports three values (`backend/config.py`'s `Literal["guided", "supervised", "full_autonomy"]`), differing only in how the single implemented gate (`EXECUTE_CODE`) is handled — there is no "Silent Mode"/trust-score bypass mechanism implemented anywhere; the description below reflects actual `LoopStateMachine._request_approval()` behavior, not an aspirational design:
1. **Guided**: an `EXECUTE_CODE` gate is created, but the backend's inline classifier auto-approve is deliberately skipped (`allow_inline_auto_approve=False`) — every gate requires an explicit decision: a human resolving it in the UI, or a human deliberately clicking the frontend's own "Auto-Approve" action (a real decision in the moment, not a silent backend shortcut).
2. **Supervised (Default)**: an `EXECUTE_CODE` gate is created, and the backend immediately attempts the `CodeSafetyClassifier` auto-approve inline; only falls back to waiting for an explicit decision if the classifier can't resolve it (`allow_inline_auto_approve=True`).
3. **Full Autonomy**: no gate is created at all — `_request_approval()` is never called, the script runs immediately.

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
| `POST /agent/missions/{id}/cancel` | Cancel a running mission loop; terminates sandbox and resets to pending |
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
2. **Sandbox Handling**: The Mission Store tracks a `ContainerID` (cloud/CPU), a `SubprocessPID` (Apple Silicon GPU, local), or a `remote_pid` (SSH-dispatched, e.g. dpo/grpo on the Mac Mini). `SandboxManager.recover()` checks whether the sandbox is still alive, uniformly across all three backends. **If alive ("reattached")**: the sandbox is left running and reattached in place — a lightweight sandbox object is reconstructed (with `_reattach_pid`/`_remote_pid`/`_container_id` set as appropriate) and registered so subsequent `is_alive()`/`tail_new_output()` polling works, but nothing is killed. The mission keeps its current status and pid/remote_pid untouched. **If gone ("dead")**: the mission is reset to `PENDING` with `container_id`/`subprocess_pid`/`remote_pid` cleared, so it can be relaunched fresh from the last checkpoint. `SandboxManager.launch()` additionally evicts and terminates any sandbox already registered for the same mission before starting a new one, guarding against leaks during mid-loop error retries — this is the one place a live sandbox can still be killed, since launching a genuinely new one for the same mission means the old one is being deliberately superseded.
3. **Loop Auto-Restart / Resume**: `recover_interrupted_missions()` returns `{"restart": [...], "resume": [...]}`. For `restart` IDs (sandbox was gone), the lifespan handler builds a fresh `LoopStateMachine` and creates a task for `loop.run(mission_id)` — resumes from `current_iteration` and the saved pivot plan, replanning nothing, but launching a brand-new sandbox process. For `resume` IDs (sandbox still alive), it instead calls `loop.run(mission_id, resume_existing_sandbox=True)`, which skips planning, the approval gate, and `sandbox.launch()` entirely for that first iteration — it reattaches to the already-running sandbox (registered by `recover()` above) and goes straight to polling it via `_wait_for_sandbox()`. Code generation is also skipped up front and only produced lazily if that resumed sandbox later errors and the healer needs a script to patch — resuming is meant to do essentially nothing besides start polling, not redo per-iteration setup work whose output won't even be used. A still-alive, hours-long remote training run is never interrupted by a backend restart.
   **Shutdown safety**: when `asyncio.CancelledError` is raised (graceful process shutdown, e.g. `make stop`/SIGTERM), the loop calls `SandboxManager.terminate(mission_id)` before resetting the mission to `PENDING`. This prevents the sandbox subprocess from running orphaned after shutdown, which would otherwise cause interleaved telemetry writes if the loop is restarted by a new process before the old one's sandbox is confirmed dead. (Historical note: this was originally motivated by uvicorn's `--reload`/WatchFiles hot-reload triggering frequent `CancelledError`s during development; `--reload` is no longer used by `make run` — see `Makefile` — precisely because hot-reload could interrupt in-flight async work like a mission loop or a DB commit mid-transaction. The `CancelledError` handler itself remains necessary for ordinary shutdown.)
4. **Telemetry Catch-up**: The **Telemetry Producer** back-fills any missed log entries from the `data/` volume to the HUD when the client reconnects, covering the outage window so operators can assess model behaviour during downtime.
