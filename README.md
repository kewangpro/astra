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
alembic upgrade head

# 5. Run the backend
make run
# or: uvicorn backend.main:app --reload
```

API docs available at `http://localhost:8000/docs` once running.

## Project Structure

```
astra/
├── backend/
│   ├── main.py              # FastAPI app entry point
│   ├── config.py            # Settings (ASTRA_* env vars)
│   ├── database.py          # SQLAlchemy async engine
│   ├── logging_config.py    # Logging setup
│   ├── models/              # ORM: Experiment, ModelRecord, Mission, Metric
│   ├── schemas/             # Pydantic request/response models
│   ├── routers/             # API route handlers
│   └── services/
│       ├── state_recovery.py   # Boot-time mission recovery
│       └── vector_memory.py    # ChromaDB semantic memory
├── alembic/                 # Database migrations
├── recipes/                 # Predefined YAML training recipes
├── data/                    # Runtime data: DB, weights, checkpoints, logs (gitignored)
├── docs/                    # Architecture & design documents
├── .env.example
└── requirements.txt
```

## API Overview

| Endpoint | Description |
|---|---|
| `GET /health` | System status + memory stats |
| `GET/POST/PATCH/DELETE /registry/experiments` | Experiment CRUD |
| `GET/POST/PATCH/DELETE /registry/models` | Model record CRUD |
| `GET/POST/PATCH/DELETE /missions` | Mission CRUD |
| `GET /recipes` | List predefined YAML recipes |
| `GET /recipes/{name}` | Fetch a single recipe |

## Status

| Phase | Description | Status |
|---|---|---|
| 1 | Foundation — backend, DB schema, vector memory, base API | ✅ Complete |
| 2 | Execution — SandboxManager, Trainers, Telemetry | Pending |
| 3 | Brain — Lead Agent (MLX), Autonomous Loop, Evaluator | Pending |
| 4 | Mission Control — Next.js dashboard, Live HUD | Pending |
| 5 | Wisdom — Recipe crystallization, evolution | Pending |
| 6 | Validation — Test suite, multi-GPU | Pending |

## Hardware Target

Optimized for **Apple Silicon (M4, 24GB unified memory)**. Local LLM inference via Native MLX (`mlx-lm`). GPU training runs as a restricted host subprocess (Metal is not accessible inside Docker on Apple Silicon). Docker is used for cloud/CUDA targets only.
