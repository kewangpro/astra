# ASTRA

**A**utonomous **S**trategic **Tr**aining **A**gent

ASTRA is an AI agent system that orchestrates end-to-end ML/RL training autonomously. You set the goal; ASTRA plans, implements, sandboxes, trains, evaluates, and iterates until the target metric is reached.

## Feature Highlights

- **Fully autonomous loop** — Plan → Implement → Sandbox → Train → Evaluate → Refine, driven by an LLM planner with no human intervention required
- **GAN-style self-critique** — a CriticAgent scores every plan on safety, complexity, and overfitting risk before code is written; the LeadAgent revises on low scores
- **Smart KV cache** — three-bucket context window (system / code / history) with sliding-window eviction so long runs never blow the token budget
- **Recipe crystallization & evolution** — completed missions are distilled into versioned YAML recipes; recipes can be mutated, selected, and promoted to "Golden" status after consecutive wins
- **Semantic warm-start** — ChromaDB recipe library retrieves the closest prior strategy before each new plan, reducing wasted iterations
- **Atomic requirement manifests** — structured pass/fail requirements (stability, artifacts, metric thresholds) replace free-text goals; the loop only completes when all are green
- **Autonomous error learning** — ErrorAnalyzer scans the entire script for all instances of an error class per healing pass; each fix is stored as a ChromaDB lesson so the CodeGenerator avoids the same mistake on future missions
- **Deterministic SB3 patching** — post-generation pass in CodeGenerator and ErrorAnalyzer injects missing stable-baselines3 imports, fixes callback class inheritance, and strips invalid constructor kwargs so weak LLM output never blocks training
- **Auto-approve with LLM classification** — `execute_code` gates can be auto-approved via a two-stage classifier: static regex pre-filter (subprocess, eval, external HTTP) followed by LLM review; unsafe scripts are flagged for manual review with a reason
- **Clean handoff protocol** — every iteration writes a `SESSION_SUMMARY.md` capturing the last action, current blocker, and exact next step for reliable warm-restart
- **Multi-sandbox execution** — SubprocessSandbox (Apple Silicon Metal) or ContainerSandbox (Docker/CUDA); SandboxManager auto-selects and handles GPU pool assignment
- **Live mission HUD** — Next.js dashboard with real-time metric charts (current vs. prior run differentiated by color), log stream, pivot timeline, and critic trace; WebSocket back-fills history on reconnect
- **Custom RL environments** — `envs/snake_env.py` provides a Gymnasium-compatible Snake-v0 environment for custom environment missions
- **Live agent viewer** — mission HUD embeds a real-time canvas rendering of the trained Snake-v0 agent playing the game, streamed over WebSocket from `best_model.zip`
- **Persistent escalating pivot strategy** — `PivotEngine` tracks consecutive failed pivots across restarts (DB-persisted `pivot_escalation_count`) and escalates through 4 levels: HP tuning → architecture change → algorithm switch (or reward shaping for algorithm-locked goals) → aggressive reward shaping. Pivot changes display real old→new diffs in the event stream
- **Algorithm-locked missions** — when a goal explicitly names an algorithm (e.g. "Train a Snake-v0 DQN agent"), ASTRA never switches to a different algorithm even at high escalation. Level 2 pivots remap to reward shaping instead
- **366 tests** — unit coverage across all core services (evolution, KV cache, crystallizer, preflight, state recovery, error analyzer, code generator, safety classifier, pivot clamping/escalation, specialist evaluator, missions router, play router, state machine helpers) plus integration tests for the full loop

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
│   ├── unit/           # 358 unit tests across all core modules
│   └── integration/    # 8 integration tests for the loop state machine
├── alembic/            # Database migrations
├── envs/               # Custom Gymnasium environments (snake_env.py → Snake-v0)
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
| 9 | Autonomous Approval & Loop Hardening — auto-approve classifier, SB3 patching, pivot clamping & architecture pivots, best-model preservation, warm-start from peak weights, manifest reconciliation, MLX inference lock, guaranteed Snake-v0 registration, classifier false-positive fixes, absolute checkpoint paths, domain dropdown removed, Snake-v0 live HUD viewer, MetricChart windowing, 4-level escalating pivot strategy (HP/arch/algo/reward), play endpoint algo+reward-config awareness, telemetry peak tracking, pivot change summaries with real old→new diffs, best_model_algo.txt watch fix, MetricGap best-vs-current iter redesign, no-op pivot detection & LLM schema normalization, pivot escalation persisted across restarts, algorithm-locked mission support, telemetry iteration tracking & "best at iter —" fix, callback __init__ peak-weight preservation, pivot plan persisted across restarts, 366 tests | 🔄 In Progress |

## Hardware Target

Optimized for **Apple Silicon M4, 24 GB unified memory**.

Training sandboxes run locally by default (subprocess using the project `.venv`). To offload training to a remote machine over SSH, set `ASTRA_SANDBOX_HOST` and optionally `ASTRA_SANDBOX_PYTHON` in `.env`.

| Machine | Role | Models / Load |
|---|---|---|
| MacBook M4 24 GB | MLX inference (Lead + Critic agents) + orchestration + local sandbox | Llama-3.1-8B-4bit (~4.5 GB) + Qwen2.5-Coder-7B-4bit (~4 GB) ≈ 8.5 GB |
| mac-mini M4 24 GB (optional) | Remote training execution via SSH | Full 24 GB available for training subprocess |

GPU training runs as a restricted host subprocess (Metal is not accessible inside Docker on Apple Silicon). Docker is used for cloud/CUDA targets only.
