# astra

**Autonomous Strategic Training Agent**

astra is an AI agent system that orchestrates end-to-end ML/RL training autonomously. You set the goal; astra plans, implements, sandboxes, trains, evaluates, and iterates until the target metric is reached.

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

# 3. Configure environment
cp .env.example .env   # edit as needed

# 4. Apply database migrations
make migrate
# or: alembic upgrade head

# 5. Run
make run   # backend + frontend → http://localhost:8200 / http://localhost:3200
```

API docs available at `http://localhost:8200/docs` once the backend is running.

## Project Structure

```
astra/
├── backend/
│   ├── main.py              # FastAPI app entry point
│   ├── config.py            # Settings (ASTRA_* env vars)
│   ├── database.py          # SQLAlchemy async engine
│   ├── logging_config.py    # Logging setup
│   ├── models/              # ORM: Experiment, ModelRecord, Mission, Metric, ApprovalGate, RecipeRecord
│   ├── schemas/             # Pydantic request/response models
│   ├── routers/             # API route handlers
│   ├── agent/               # LeadAgent, CodeGenerator, ErrorAnalyzer, ModelManager, KVCache, InferenceProviders
│   ├── loop/                # LoopStateMachine, PivotEngine
│   ├── sandbox/             # BaseSandbox, SubprocessSandbox, ContainerSandbox, SandboxManager
│   ├── trainers/            # BaseTrainer, RLTrainer, SFTTrainer, MLTrainer
│   ├── evaluator/           # SpecialistEvaluator, BenchmarkSuite, StressTester
│   ├── analysis/            # SpatialAnalyzer (Grad-CAM), PolicyAuditor
│   └── services/
│       ├── state_recovery.py    # Boot-time mission recovery
│       ├── vector_memory.py     # ChromaDB lessons-learned memory
│       ├── connection_manager.py# WebSocket fan-out for live telemetry
│       ├── crystallizer.py      # Distils completed missions into recipes
│       ├── recipe_library.py    # ChromaDB semantic recipe index + warm-start
│       └── evolution.py         # Mutation, selection, GenePool, GoldenPromoter
├── frontend/                # Next.js 15 mission control dashboard (port 3200)
├── alembic/                 # Database migrations
├── recipes/                 # YAML training recipes (hand-crafted + crystallized)
├── data/                    # Runtime data: DB, weights, checkpoints, logs (gitignored)
├── docs/                    # Architecture & design documents
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
| 5 | Wisdom — Recipe crystallization, evolution | ✅ Complete |
| 6 | Validation — Test suite, multi-GPU | Pending |

## Hardware Target

Optimized for **Apple Silicon (M4, 24GB unified memory)**. Local LLM inference via Native MLX (`mlx-lm`). GPU training runs as a restricted host subprocess (Metal is not accessible inside Docker on Apple Silicon). Docker is used for cloud/CUDA targets only.
