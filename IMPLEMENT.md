# astra: Implementation Roadmap

This document outlines the phased implementation strategy for `astra`.

---

## Phase 1: The Foundation (Backend & Memory)
*Goal: Establish the core API, database schema, and project structure.*

- [ ] **Step 1.1: Project Scaffolding**
    - Setup FastAPI backend structure.
    - Configure environment variables and logging.
- [ ] **Step 1.2: Model Registry & Mission Store (SQL Storage)**
    - Implement SQLAlchemy models for `Experiments`, `Models`, `Metrics`, and `Missions`.
    - Create migration scripts.
    - Implement the "State Recovery Manager" to handle boot-time resumption.
- [ ] **Step 1.3: Vector Memory (Semantic Storage)**
    - Integrate ChromaDB or FAISS for storing "Lessons Learned."
    - Implement embedding utility for log analysis.
- [ ] **Step 1.4: Base API Endpoints**
    - CRUD for Recipes and Model Registry.
    - Health checks and system status.

## Phase 2: The Execution (Sandbox & Trainers)
*Goal: Enable safe, containerized code execution and specialized training logic.*

- [ ] **Step 2.1: Sandbox Manager**
    - Implement Docker/Podman orchestration logic.
    - Define resource limiting policies (CPU/GPU/RAM).
- [ ] **Step 2.2: Universal Specialist Trainer**
    - Build the base `Trainer` class.
    - Implement `RLTrainer` (wrapping SB3/PyTorch).
    - Implement `SFTTrainer` (wrapping HuggingFace/LoRA).
    - Implement `MLTrainer` (wrapping Scikit-learn/Lightning).
- [ ] **Step 2.3: Telemetry Streamer**
    - Implement a WebSocket-based metrics exporter from the sandbox to the backend.

## Phase 3: The Brain (LLM & Autonomous Loop)
*Goal: Implement the planning, self-healing, and iteration logic.*

- [ ] **Step 3.1: Lead Agent (The Orchestrator)**
    - Integrate LLM (OpenAI/Gemini/Ollama) for goal decomposition.
    - Implement the "Planner" that converts goals to DAGs.
- [ ] **Step 3.2: Code Generator & Self-Healer**
    - Create prompt templates for generating training scripts.
    - Implement the "Error Analyzer" that fixes code based on stack traces.
- [ ] **Step 3.3: The Autonomous Loop State Machine**
    - Implement the logic: Plan -> Implement -> Sandbox -> Execute -> Eval -> Refine.
    - Add support for strategic pivots (e.g., hyperparameter adjustments).

## Phase 4: Mission Control (Web Dashboard)
*Goal: Build the professional Next.js interface for monitoring and control.*

- [ ] **Step 4.1: Dashboard Scaffolding**
    - Setup Next.js 15 with Tailwind CSS and shadcn/ui.
- [ ] **Step 4.2: The Command Center (Home)**
    - Implement the "Goal Input" bar.
    - Create the "Active Missions" grid.
- [ ] **Step 4.3: Live Training HUD**
    - Implement real-time charts using `recharts`.
    - Build the "Metric Gap" gauge and strategic pivot timeline.
- [ ] **Step 4.4: Approval Controller UI**
    - Build the side-by-side code diff viewer for security gates.

## Phase 5: The Wisdom (Recipes & Sharing)
*Goal: Finalize crystallization logic and the strategy sharing library.*

- [ ] **Step 5.1: Recipe Crystallization Logic**
    - Automate the distillation of a successful run into a YAML recipe.
- [ ] **Step 5.2: Recipe Library & Retrieval**
    - Implement semantic search for recipes based on new goals.
    - Build the "Warm-Start" planning logic.
- [ ] **Step 5.3: Strategy Evolution**
    - Implement cross-mission learning where "Golden Recipes" are refined over time.

## Phase 6: Validation & Scaling
*Goal: Ensure robustness and prepare for multi-GPU/distributed use.*

- [ ] **Step 6.1: Comprehensive Test Suite**
    - Integration tests for the full loop (using a "Mock" environment).
- [ ] **Step 6.2: Multi-GPU Orchestration**
    - Add support for assigning specific sandboxes to specific GPUs.
- [ ] **Step 6.3: "Golden Set" Benchmarking**
    - Finalize the automated Evaluator's stress-testing suite.
