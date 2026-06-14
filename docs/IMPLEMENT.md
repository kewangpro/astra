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
    - Implement the "State Recovery Manager" for boot-time resumption: query `RUNNING`/`PAUSED` missions and reset their state using atomic DB transactions (no sandbox operations yet — sandbox re-attachment is extended in Step 2.1).
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
    - Extend the State Recovery Manager (Step 1.2) with sandbox re-attachment: detect live containers by stored `ContainerID`; restart from last checkpoint if the container is gone.
- [ ] **Step 2.2: Universal Specialist Trainer**
    - Build the base `Trainer` class.
    - Enforce checkpoint cadence: write weights + optimizer state to `storage/` volume every 2–5 minutes of wall-clock time; register checkpoint path in the Model Registry as metadata.
    - Implement `RLTrainer` (wrapping SB3/PyTorch).
    - Implement `SFTTrainer` (wrapping HuggingFace/LoRA). Override default `save_strategy` to `steps` with `save_steps` tuned to the 2–5 minute target.
    - Implement `MLTrainer` (wrapping Scikit-learn/Lightning).
- [ ] **Step 2.3: Telemetry Producer**
    - Implement a WebSocket-based metrics exporter from the sandbox to the backend (FastAPI → HUD).
    - Implement back-fill logic: on recovery, replay missed log entries from `storage/` volume to the HUD.

## Phase 3: The Brain (LLM & Autonomous Loop)
*Goal: Implement the planning, self-healing, and iteration logic.*

- [ ] **Step 3.1: Lead Agent (The Orchestrator)**
    - Integrate **Native MLX Inference** (via `mlx-lm`) as the primary local provider for 24GB hardware.
    - Implement a `ModelManager` to dynamically adjust LLM memory footprint and trigger garbage collection when training sandboxes require more VRAM.
    - Setup prefix caching for efficient real-time log analysis.
    - (Optional) Build an abstraction layer to support **vLLM (Metal)** for high-memory environments.
    - Setup system prompts for strategic goal decomposition.
- [ ] **Step 3.2: Code Generator & Self-Healer**
    - Create prompt templates for generating training scripts.
    - Implement the "Error Analyzer" that fixes code based on stack traces.
- [ ] **Step 3.3: The Autonomous Loop State Machine**
    - Implement the logic: Plan -> Implement -> Sandbox -> Execute -> Eval -> Refine.
    - Add support for strategic pivots (e.g., hyperparameter adjustments).
- [ ] **Step 3.4: Specialist Evaluator**
    - Build the independent `Evaluator` agent (DESIGN §2.6).
    - Implement `BenchmarkSuite`: runs the model against a fixed "Golden Set" of challenges.
    - Implement `StressTester`: introduces noise and edge cases to verify robustness.
    - Wire the Evaluator as the mandatory actor in the `Eval` phase of the loop state machine.
- [ ] **Step 3.5: Analysis & Introspection Suite**
    - Implement `SpatialAnalyzer`: generates CNN saliency/activation maps and exposes them via API for the Model Registry deep-dive view (DESIGN §2.7).
    - Implement `PolicyAuditor`: computes and logs action-distribution histograms to detect mode collapse or bias.

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
    - Embed the Approval Queue panel (built in Step 4.4) as an inline overlay within the HUD view — the user must never need to navigate away from the HUD to act on a pending gate.
- [ ] **Step 4.4: Approval Controller UI**
    - Build the side-by-side code diff viewer for security gates.
    - Expose as an embeddable panel (not a standalone page) so Step 4.3 can mount it inside the HUD.

## Phase 5: The Wisdom (Recipes & Sharing)
*Goal: Finalize crystallization logic and the strategy sharing library.*

- [ ] **Step 5.1: Recipe Crystallization Logic**
    - Automate the distillation of a successful run into a YAML recipe.
- [ ] **Step 5.2: Recipe Library & Retrieval**
    - Implement semantic search for recipes based on new goals.
    - Build the "Warm-Start" planning logic.
- [ ] **Step 5.3: Strategy Evolution**
    - Implement a mutation operator: produce candidate child recipes by perturbing hyperparameter values within configurable bounds.
    - Implement a selection policy: promote a child recipe to replace its parent only if it achieves a higher score on the Specialist Evaluator's benchmark suite.
    - Build a cross-mission candidate pool: aggregate top-performing recipes across different runs of the same domain into a gene pool.
    - Define Golden Recipe promotion criteria: a recipe earns "Golden" status after N consecutive successful runs above the target threshold.
    - Add a regression test harness: ensure a newly promoted Golden Recipe does not regress on any prior benchmark the domain has solved.

## Phase 6: Validation & Scaling
*Goal: Ensure robustness and prepare for multi-GPU/distributed use.*

- [ ] **Step 6.1: Comprehensive Test Suite**
    - Integration tests for the full loop (using a "Mock" environment).
- [ ] **Step 6.2: Multi-GPU Orchestration**
    - Add support for assigning specific sandboxes to specific GPUs.
- [ ] **Step 6.3: "Golden Set" Benchmarking**
    - Harden and expand the Specialist Evaluator (built in Step 3.4): add domain-specific Golden Set scenarios for Snake, Tetris, and NLP tasks.
    - Stress-test the StressTester itself: verify edge-case coverage, noise injection calibration, and benchmark reproducibility.
