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
- **Clean handoff protocol** — every iteration writes a `SESSION_SUMMARY.md` capturing the last action, current blocker, and exact next step for reliable warm-restart
- **Multi-sandbox execution** — SubprocessSandbox (Apple Silicon Metal) or ContainerSandbox (Docker/CUDA); SandboxManager auto-selects and handles GPU pool assignment
- **Live mission HUD** — Next.js dashboard with real-time metric charts, log stream, pivot timeline, and critic trace; WebSocket back-fills history on reconnect
- **223 tests** — unit coverage across all core services (evolution, KV cache, crystallizer, preflight, state recovery, error analyzer, code generator, missions router) plus integration tests for the full loop

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

API docs available at `http://localhost:8200/docs` once the backend is running.

## Project Structure

```
astra/
├── backend/
│   ├── agent/          # LeadAgent, CriticAgent, CodeGenerator, ModelManager, KVCache, inference providers
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
│   ├── unit/           # 215 unit tests across all core modules
│   └── integration/    # 8 integration tests for the loop state machine
├── alembic/            # Database migrations
├── recipes/            # YAML training recipes (hand-crafted + crystallized + evolved)
├── data/               # Runtime data: DB, weights, checkpoints, logs (gitignored)
├── docs/               # Architecture & design documents
├── .env.example
└── requirements.txt
```

## API Overview

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
| `POST /telemetry/missions/{id}/metrics` | Sandbox pushes metrics |
| `WS /ws/missions/{id}/telemetry` | Live telemetry WebSocket (back-fills history on connect) |
| `POST /analysis/missions/{id}/saliency` | Grad-CAM saliency map |
| `POST /analysis/missions/{id}/audit` | Policy audit (action histogram + entropy) |
| `GET /recipes` | List all recipes (disk + DB merged) |
| `GET /recipes/db` | List DB-backed recipes (`domain`, `golden_only` filters) |
| `GET /recipes/search?q=` | Semantic search over recipe library |
| `GET /recipes/{name}` | Fetch a single recipe (DB-first, disk fallback) |
| `POST /recipes/crystallize/{mission_id}` | Distil a completed mission into a recipe |
| `POST /recipes/{id}/evolve` | Spawn a mutated child recipe |
| `GET /recipes/{id}/lineage` | Ancestor chain for an evolved recipe |

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

## Hardware Target

Optimized for **Apple Silicon M4, 24 GB unified memory**.

Training sandboxes run locally by default (subprocess using the project `.venv`). To offload training to a remote machine over SSH, set `ASTRA_SANDBOX_HOST` and optionally `ASTRA_SANDBOX_PYTHON` in `.env`.

| Machine | Role | Models / Load |
|---|---|---|
| MacBook M4 24 GB | MLX inference (Lead + Critic agents) + orchestration + local sandbox | Llama-3.1-8B-4bit (~4.5 GB) + Qwen2.5-Coder-7B-4bit (~4 GB) ≈ 8.5 GB |
| mac-mini M4 24 GB (optional) | Remote training execution via SSH | Full 24 GB available for training subprocess |

GPU training runs as a restricted host subprocess (Metal is not accessible inside Docker on Apple Silicon). Docker is used for cloud/CUDA targets only.
