# ASTRA

**A**utonomous **S**trategic **Tr**aining **A**gent

ASTRA is an AI agent system that orchestrates end-to-end ML/RL training autonomously. You set the goal; ASTRA plans, implements, sandboxes, trains, evaluates, and iterates until the target metric is reached.

## Feature Highlights

- **End-to-End Autonomous Loop** — Plan → Critique → Implement → Sandbox → Train → Evaluate → Refine. Integrates GAN-style plan critique, automatic error recovery, and convergence guards that prevent runaway compute.
- **Adaptive Escalating Pivots** — Stalled missions systematically escalate across hyperparameter tuning, network architecture mutations, algorithm switches, and environment reward shaping, with progress persisted across restarts.
- **Model Registry & Tournament Arena** — Benchmark multiple models head-to-head on identical deterministic seeds (`2000 + ep`), tracking score distributions and tie-split win rates to automatically crown champion policies.
- **Recipe Library & Lineage Evolution** — Reusable YAML training blueprints with genetic mutation tracking (Lineage DAG), "Golden" recipe distillation from successful runs, and one-click mission dispatch.
- **Live Mission HUD & Explainability** — Real-time telemetry, memory gauges, and interactive canvas players streaming frame-by-frame Q-values, action probabilities, and Shannon policy entropy across all environments.
- **High-Throughput Custom Envs** — Pure Python/NumPy environments (>50k steps/sec) for Snake, Tetris, 2048 (with 1-step lookahead evaluation), and the MinAtar arcade suite (Breakout, Space Invaders, Asteroids).
- **Multi-Paradigm Post-Training & Hybrid Compute** — Supports RL, SFT (with strict held-out validation and reasoning `<think>...</think>` preservation), DPO, GRPO, Distillation, and ML across local Apple Silicon (Metal/MLX) and remote SSH compute nodes with real-time cluster memory visibility.


### Screenshots

| Command Center | Mission HUD |
|---|---|
| ![Command Center — mission grid with status badges and Run button](docs/screenshots/command_center.png) | ![Mission HUD — metric chart, log stream, pivot timeline, Snake live viewer](docs/screenshots/mission_hud.png) |

| Metric History (current vs. prior run) | Auto-Approve & Approval Panel |
|---|---|
| ![Metric History chart showing bright current run over muted prior runs](docs/screenshots/metric_history.png) | ![Approval panel with Auto-Approve button and safety verdict card](docs/screenshots/approval_panel.png) |

| Snake-v0 Live Viewer | Tetris-v0 Live Viewer |
|---|---|
| ![Snake-v0 agent playing live in the mission HUD — grid canvas with head, body, and food rendered in real time](docs/screenshots/snake_viewer.png) | ![Tetris-v0 agent playing live in the mission HUD — board canvas with piece colors and line-clear highlights](docs/screenshots/tetris_viewer.png) |

| Game2048-v0 Live Viewer | MinAtar-Breakout-v0 Live Viewer |
|---|---|
| ![Game2048-v0 agent playing live in the mission HUD — 4x4 tile canvas with score and max tile tracking](docs/screenshots/game2048_viewer.png) | ![MinAtar-Breakout-v0 agent playing live in the mission HUD — 10x10 symbolic arcade canvas with paddle, ball, and bricks](docs/screenshots/minatar_viewer.png) |

| Model Registry & Tournament Leaderboard | Recipe Library & Lineage Visualizer |
|---|---|
| ![Model Registry & Tournament Arena — fixed-seed head-to-head simulations and champion podium](docs/screenshots/model_registry.png) | ![Recipe Library & Lineage — canonical recipe gallery and evolutionary lineage DAG](docs/screenshots/recipe_library.png) |



## Documentation

| Doc | Purpose |
|---|---|
| [PRD.md](docs/PRD.md) | Product requirements & feature definitions |
| [DESIGN.md](docs/DESIGN.md) | Technical architecture & component design |
| [IMPLEMENT.md](docs/IMPLEMENT.md) | Phase-by-phase implementation roadmap |
| [UX_SPEC.md](docs/UX_SPEC.md) | Dashboard UX specification |

## Task Types

ASTRA supports nine training paradigms — `rl`, `sft`, `ml`, `mlx_lora`, `dpo`, `grpo`, `distill`, `rft`, `prompt` — each driving a different trainer/code-gen path. See [DESIGN.md § 2.4](docs/DESIGN.md) for what each one optimizes and how it trains.

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
│   ├── sandbox/        # SubprocessSandbox, ContainerSandbox, SSHSandbox, SandboxManager
│   ├── schemas/        # Pydantic request/response models
│   ├── services/       # Crystallizer, RecipeLibrary, Evolution, VectorMemory, MissionState, Preflight, StateRecovery
│   └── trainers/       # RLTrainer, SFTTrainer, MLTrainer
├── frontend/           # Next.js 15 mission control dashboard (port 3200)
├── tests/
│   ├── unit/           # 1063 unit tests across all core modules
│   └── integration/    # 15 integration tests for the loop state machine
├── alembic/            # Database migrations
├── envs/               # Custom Gymnasium environments (Snake-v0, Tetris-v0, Game2048-v0, MinAtar Suite)

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

## Development Milestones

| Milestone | Scope & Key Capabilities | Status |
|---|---|---|
| **Core Architecture (Phases 1–7)** | Autonomous loop, MLX Lead Agent, SandboxManager, GAN critique, manifest state machine, vector memory | ✅ Complete |
| **Autonomous Learning (Phases 8–16)** | Self-healing code generation, auto-approvals, 4-stage escalating pivots, regression rollback, best-arch memory | ✅ Complete |
| **RL Environments & Lookahead (Phases 17–24, 31–32)** | Pure Gym envs (Snake, Tetris), 1-step successor lookahead trainers, flood-fill reachable space features, curriculum learning | ✅ Complete |
| **Fine-Tuning & Distillation (Phases 25–28, 35–48)** | Remote SSH sandboxes, DPO, GRPO, Distillation, RFT, and Prompt optimization with checkpoint chaining & convergence guards | ✅ Complete |
| **Mission Control & Kanban (Phases 34, 38, 49)** | Cluster visibility (Nodes panel), status-grouped Kanban board, live dual-metric history, and approval gates | ✅ Complete |
| **Arcade Suite & Explainability (Phases 51–54)** | 2048 & MinAtar arcade suite (Breakout, Space Invaders, Asteroids), lookahead DQN, live Policy Audit Inspector (Q-values/entropy) | ✅ Complete |
| **Registry, Tournaments & Recipes (Phases 55–56)** | Model Registry, fixed-seed Tournament Arena, champion crowning, Recipe Library with Lineage DAG, and 1-click dispatch | ✅ Complete |
| **SFT & Reasoning Post-Training (Phase 57)** | Full SFTTrainer, strict held-out train/val splitting (val_split), reasoning CoT (<think>...</think>) preservation, AST-guarded codegen & self-healing | ✅ Complete |
| **Remote SFT via MLX (Phase 58)** | Apple Silicon Mac Mini offload following DPO pattern, zero-orphan `os.execv` wrapper, dynamic validation batch-clamping, live SSH loss tailing | ✅ Complete |

> Full phase-by-phase implementation logs and technical changelogs are maintained in [IMPLEMENT.md](docs/IMPLEMENT.md).




## Hardware Target

Optimized for **Apple Silicon M4, 24 GB unified memory**.

Training sandboxes run locally by default (subprocess using the project `.venv`). To offload training to a remote machine over SSH (supporting `sft`, `dpo`, `grpo`, `distill`, `rft`), set `ASTRA_SANDBOX_HOST` and optionally `ASTRA_SANDBOX_PYTHON` in `.env`.

| Machine | Role | Models / Load |
|---|---|---|
| MacBook M4 24 GB | MLX inference (Lead + Critic agents) + orchestration + local sandbox | Llama-3.1-8B-4bit (~4.5 GB) + Qwen2.5-Coder-7B-4bit (~4 GB) ≈ 8.5 GB |
| mac-mini M4 24 GB (optional) | Remote training execution via SSH (`sft`, `dpo`, `grpo`, `distill`, `rft`) | Full 24 GB available for training subprocess (Gemma-3-12B-it-4bit, etc.) |

GPU training runs as a restricted host subprocess (Metal is not accessible inside Docker on Apple Silicon). Docker is used for cloud/CUDA targets only.
