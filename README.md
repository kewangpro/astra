# ASTRA

**A**utonomous **S**trategic **Tr**aining **A**gent

ASTRA is an AI agent system that orchestrates end-to-end ML/RL training autonomously. You set the goal; ASTRA plans, implements, sandboxes, trains, evaluates, and iterates until the target metric is reached.

## Feature Highlights

- **Fully autonomous loop** — Plan → Implement → Sandbox → Train → Evaluate → Refine, driven by an LLM planner with no human intervention required
- **GAN-style self-critique** — a CriticAgent scores every plan on safety, complexity, and overfitting risk before code is written; the LeadAgent revises on low scores
- **Recipe crystallization & evolution** — completed missions are distilled into versioned YAML recipes; recipes can be mutated, selected, and promoted to "Golden" status after consecutive wins
- **Autonomous error learning** — ErrorAnalyzer stores each fix as a ChromaDB lesson so CodeGenerator avoids repeating the same mistake on future missions
- **Auto-approve with LLM classification** — `execute_code` gates are auto-approved via a two-stage classifier (static regex → LLM review); unsafe scripts are flagged with a reason for manual review
- **Multi-sandbox execution** — SubprocessSandbox (Apple Silicon Metal) or ContainerSandbox (Docker/CUDA); SandboxManager auto-selects and handles GPU pool assignment
- **Live mission HUD** — Next.js dashboard with real-time metric charts, log stream, pivot timeline, and critic trace; WebSocket back-fills history on reconnect
- **Custom RL environments** — Snake-v0 and Tetris-v0 Gymnasium-compatible environments; Snake-v0 tracks `food_eaten`; Tetris-v0 uses a placement-based `Discrete(40)` action space and a compact 4-feature observation `[lines_cleared_last, holes, bumpiness, sum_height]` (normalized) matching the reference approach that achieved 45+ lines cleared vs ~10 with a flat 224-element board
- **Live agent viewer** — mission HUD streams the trained agent playing Snake-v0 or Tetris-v0 in real time over WebSocket; auto-detects SB3 (`best_model.zip`) vs PyTorch Actor-Critic (`best_model.pth`) and uses `get_next_states()` lookahead for Tetris playback
- **Persistent escalating pivot strategy** — PivotEngine escalates through 4 levels (HP tuning → architecture change → algorithm switch → reward shaping) across server restarts via DB-persisted `pivot_escalation_count`; pivot event stream shows real old→new diffs
- **Best-architecture memory** — PivotEngine tracks which `net_arch` produced the best goal metric; persisted to DB and restored on restart so the hint survives process restarts; `LeadAgent.propose_pivot` receives this context and is instructed to reuse the proven architecture at Level 1 rather than randomly cycling between `[256, 256]`, `[400, 300]`, and `[256, 256, 128]`, preventing warm-start-breaking architecture thrash
- **Dual metric tracking** — MetricHistory shows the training signal (`mean_reward`); MetricGap tracks the goal metric separately (`food_eaten`, `lines_cleared`) via post-iteration eval rollouts; both update live in the HUD
- **Robust state recovery** — on restart, interrupted missions are automatically detected, stale sandboxes terminated (including reattached processes killed by stored pid, not just by Popen handle), and `LoopStateMachine` relaunched to resume training from the last checkpoint and iteration
- **544 tests** — 535 unit + 9 integration tests covering all core services

### Screenshots

| Command Center | Mission HUD |
|---|---|
| ![Command Center — mission grid with status badges and Run button](docs/screenshots/command_center.png) | ![Mission HUD — metric chart, log stream, pivot timeline, Snake live viewer](docs/screenshots/mission_hud.png) |

| Metric History (current vs. prior run) | Auto-Approve & Approval Panel |
|---|---|
| ![Metric History chart showing bright current run over muted prior runs](docs/screenshots/metric_history.png) | ![Approval panel with Auto-Approve button and safety verdict card](docs/screenshots/approval_panel.png) |

## Documentation

| Doc | Purpose |
|---|---|
| [PRD.md](docs/PRD.md) | Product requirements & feature definitions |
| [DESIGN.md](docs/DESIGN.md) | Technical architecture & component design |
| [IMPLEMENT.md](docs/IMPLEMENT.md) | Phase-by-phase implementation roadmap |
| [UX_SPEC.md](docs/UX_SPEC.md) | Dashboard UX specification |

## Quick Start

```bash
# 1. Create and activate the virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download local MLX models (required for first run)
huggingface-cli download mlx-community/Meta-Llama-3.1-8B-Instruct-4bit
huggingface-cli download mlx-community/Qwen2.5-Coder-7B-Instruct-4bit

# 4. Configure environment
cp .env.example .env   # edit as needed

# 5. Apply database migrations
alembic upgrade head

# 6. Run
make run   # backend + frontend → http://localhost:8200 / http://localhost:3200
```

## Project Structure

```
astra/
├── backend/
│   ├── agent/          # LeadAgent, CriticAgent, CodeGenerator, ErrorAnalyzer, CodeSafetyClassifier, ModelManager, KVCache, inference providers
│   ├── analysis/       # SpatialAnalyzer (Grad-CAM), PolicyAuditor
│   ├── evaluator/      # SpecialistEvaluator, BenchmarkSuite, StressTester, ManifestEvaluator
│   ├── loop/           # LoopStateMachine, PivotEngine
│   ├── models/         # ORM models: Mission, Experiment, ModelRecord, RecipeRecord, ApprovalGate, Manifest
│   ├── routers/        # API route handlers
│   ├── sandbox/        # SubprocessSandbox, ContainerSandbox, SandboxManager
│   ├── schemas/        # Pydantic request/response models
│   ├── services/       # Crystallizer, RecipeLibrary, Evolution, VectorMemory, MissionState, Preflight, StateRecovery
│   └── trainers/       # RLTrainer, SFTTrainer, MLTrainer
├── frontend/           # Next.js 15 mission control dashboard (port 3200)
├── tests/
│   ├── unit/           # 492 unit tests across all core modules
│   └── integration/    # 9 integration tests for the loop state machine
├── alembic/            # Database migrations
├── envs/               # Custom Gymnasium environments (Snake-v0, Tetris-v0)
├── recipes/            # YAML training recipes (hand-crafted + crystallized + evolved)
├── data/               # Runtime data: DB, weights, checkpoints, logs (gitignored)
├── docs/               # Architecture & design documents
├── .env.example
└── requirements.txt
```

## API Overview

Full endpoint reference is in [DESIGN.md § 5.4](docs/DESIGN.md). Interactive docs available at `http://localhost:8200/docs` once the backend is running.

## Make Commands

```bash
make run    # start backend (port 8200) + frontend (port 3200)
make stop   # stop both
make ports  # show port status for all services
```

## Status

| Phase | Description | Status |
|---|---|---|
| 1 | Foundation — backend, DB schema, vector memory, base API | ✅ Complete |
| 2 | Execution — SandboxManager, Trainers, Telemetry | ✅ Complete |
| 3 | Brain — Lead Agent (MLX), Autonomous Loop, Evaluator | ✅ Complete |
| 4 | Mission Control — Next.js dashboard, Live HUD | ✅ Complete |
| 5 | Wisdom — Recipe crystallization, evolution, golden promotion | ✅ Complete |
| 6 | Validation — Test suite, multi-GPU | ✅ Complete |
| 7 | Resilience & Rigor — GAN critique, manifests, preflight, state | ✅ Complete |
| 8 | Autonomous Learning & HUD Polish — error learning, metric display, 223 tests | ✅ Complete |
| 9 | Autonomous Approval & Code Robustness — auto-approve, SB3 patching, checkpoint/warm-start, Snake-v0 viewer | ✅ Complete |
| 10 | Pivot Intelligence & Live Viewer — 4-level escalation (HP/arch/algo/reward), MetricChart windowing, play endpoint | ✅ Complete |
| 11 | Resilience & Dual Metrics — Tetris-v0, dual metric tracking, pivot persistence, algorithm-locked missions | ✅ Complete |
| 12 | Mission Lifecycle & Telemetry — clean deletion, sandbox error detection, goal metric cap, pivot context, resume hardening | ✅ Complete |
| 13 | Training Continuity & Loop Recovery — env_kwargs merge/clamp, distance_weight floor, early-stop threshold fix, 2M timestep floor, arch oscillation detection, MetricChart adaptive x-axis, state recovery auto-restart loop, plan reuse across iterations, 456 tests | ✅ Complete |
| 14 | HUD Polish & Telemetry Performance — WS batch backfill, event stream capped at 100, sidebar height alignment, pivot history scrollable, MetricChart x-axis tickCount, integer iteration labels | ✅ Complete |
| 15 | Sandbox Lifecycle Hardening — orphaned subprocess fix (reattach kill-by-pid), stale sandbox eviction before launch, sandbox terminate on shutdown cancel, 464 tests | ✅ Complete |
| 16 | Post-Pivot Regression Detection, Checkpoint Recovery & Best-Architecture Memory — 20% regression threshold, per-iteration rolling checkpoint window (last 10), revert targets true best-ever iter, de-escalation; best-architecture memory persisted to DB; `_normalize_pivot` policy_kwargs promotion; stop button + MLX shield; auto-approve variable URL fix; competitive-dip guard (15% tolerance) prevents false pivots on variance; `pivot_pre_best` persisted so regression detector survives restarts; `mean_reward` inflation fix in `_load_persisted_best`; 498 tests | ✅ Complete |
| 17 | Tetris Obs Refactor + Actor-Critic Trainer — 4-feature compact obs (Step 17.1); `get_next_states()` env method + Actor-Critic contract prompt replacing SB3 template; benchmark + play router support `.pth` models; `trainer_type` routing in `state_machine`, `benchmark`, `play`; `_tetris_viewer_grid` for HUD compatibility; actor_critic infrastructure hardening (`actor_critic_net.py`, goal metric fix, pre-clear highlight frames, per-cell piece colors in TetrisPlayer, crystallizer actor_critic support); 529 tests | ✅ Complete |
| 18 | Hardcode Removal — all training knobs (`total_timesteps`, `telemetry_interval`, `replay_buffer_size`, `batch_size`, `gamma`, `epsilon_*`, `ac_telemetry_interval`, `eval_episodes`) driven from recipe `hyperparameters:`; recipes restructured (`sft_llama_lora_v1` flat `hyperparameters:` block); 9 stale crystallized recipes deleted; 533 tests | ✅ Complete |
| 19 | Snake Feature Obs — `obs_type=features` adds 25D compact observation (immediate danger, 5-step path clearance, food direction, wall distances, spatial scalars) alongside existing flat grid; `snake_ppo_v1.yaml` v2 uses features obs + `max_steps=2000` + food-dominant rewards; 540 tests | ✅ Complete |

## Hardware Target

Optimized for **Apple Silicon M4, 24 GB unified memory**.

Training sandboxes run locally by default (subprocess using the project `.venv`). To offload training to a remote machine over SSH, set `ASTRA_SANDBOX_HOST` and optionally `ASTRA_SANDBOX_PYTHON` in `.env`.

| Machine | Role | Models / Load |
|---|---|---|
| MacBook M4 24 GB | MLX inference (Lead + Critic agents) + orchestration + local sandbox | Llama-3.1-8B-4bit (~4.5 GB) + Qwen2.5-Coder-7B-4bit (~4 GB) ≈ 8.5 GB |
| mac-mini M4 24 GB (optional) | Remote training execution via SSH | Full 24 GB available for training subprocess |

GPU training runs as a restricted host subprocess (Metal is not accessible inside Docker on Apple Silicon). Docker is used for cloud/CUDA targets only.
