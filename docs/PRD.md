# ASTRA: Product Requirements Document (PRD)

**Project Name:** ASTRA (**A**utonomous **S**trategic **Tr**aining **A**gent)  
**Status:** Complete (all 6 phases shipped)  
**Target:** Autonomous Machine Learning Orchestration

---

## 1. Executive Summary
ASTRA is an autonomous agent designed to manage the end-to-end lifecycle of Reinforcement Learning (RL) and Machine Learning (ML) training. It leverages lessons from high-performance Snake and Tetris AI implementations to automate curriculum shifts, reward shaping, and competitive benchmarking.

## 2. Problem Statement
Manual ML training is repetitive and error-prone. Engineers often spend hours:
- Monitoring training logs for convergence or plateaus.
- Manually adjusting grid sizes or difficulty levels (Curriculum Learning).
- Comparing different algorithms (PPO vs DQN) across multiple seeds.
- Writing boilerplate code for model saving and performance plotting.

## 3. Goals & Objectives
- **Autonomy**: Reduce human intervention in the training loop by 80%.
- **Optimization**: Discover better hyperparameter and reward configurations through automated experimentation.
- **Observability**: Provide high-fidelity insights into *why* a model is performing via feature map analysis.
- **Portability**: Create a system that can be easily plugged into different environments (Snake, Tetris, Finance, etc.).

## 4. Key Features

### 4.1. Autonomous Curriculum Manager
- Automatically shifts training phases based on success metrics.
- Scales environment complexity (e.g., grid size) without manual restarts.

### 4.2. Multi-Agent Orchestrator
- Dispatches parallel training jobs for different algorithms (PPO, DQN, A2C).
- Implements "Survival of the Fittest" where underperforming experiments are pruned.

### 4.3. Dynamic Reward Evolver
- Tests variations of reward shaping.
- Uses a "critic" agent to propose reward adjustments based on agent behavior.

### 4.4. The Registry & Benchmark Suite
- Persistent storage for models, weights, and training metadata.
- Automated "Tournament Mode" to compare new models against the current "Champion."

### 4.5. Smart Visualizer
- Automatically captures video of "Breakthrough Moments."
- Generates CNN activation maps and feature plots.

### 4.6. Autonomous Iteration Loop
- Continuous "Plan-Train-Evaluate-Refine" cycle.
- System autonomously restarts or pivots training strategies if goals are not met within predicted timelines.
- Learns from failed iterations to adjust future hyperparameters or curriculum steps.

### 4.7. Autonomous Code Implementation
- ASTRA can generate necessary training scripts across various paradigms:
  - **RL**: Environment wrappers, reward functions, and policy networks.
  - **SFT**: Tokenization logic, prompt templates, and supervised loss functions.
  - **ML**: Feature engineering, data loaders, and standard classification/regression architectures.
- Autonomously fixes bugs in training code by analyzing error logs and stack traces.

### 4.8. Secure Sandboxed Execution
- All generated code runs in an isolated sandbox to protect the host system. The sandbox type depends on the hardware target: **Docker/Podman** (cloud or CPU-only workloads) or a **restricted host subprocess** (Apple Silicon, where Metal GPU is not accessible inside Docker).
- Resource constraints (CPU/GPU/RAM) are enforced at the sandbox level to prevent runaway processes.

### 4.9. Training Recipes & Crystallization
- **Recipe Generation**: Upon achieving a goal, ASTRA "crystallizes" the successful strategy into a **Training Recipe** (a package containing the optimized hyperparameters, reward/loss logic, curriculum phases, and model architecture).
- **Strategy Sharing**: Recipes are stored in a global library, allowing ASTRA to "warm-start" new, similar goals by retrieving and adapting existing recipes.
- **Recipe Evolution**: ASTRA can treat a recipe as a "DNA" strand, mutating and improving it across different training runs to discover universal "Golden Recipes" for specific domains (e.g., "The Golden Snake Recipe").

### 4.10. Predefined "Golden" Recipes
- ASTRA ships with a set of **Predefined Base Recipes** derived from proven, high-performance training runs (e.g., the workspace's existing Snake and Tetris models).
- These recipes serve as the "Initial Knowledge" of the system, allowing users to achieve expert-level results on day one for common tasks.

### 4.11. Crash-Safe Mission Persistence
- **Stateful Resumption**: If the system is interrupted (crash, power loss, restart), ASTRA automatically recovers the state of all "In-Progress" missions.
- **Checkpoint-Aware Training**: All trainers are required to save frequent checkpoints (weights + optimizer state) to the **File Store** (`data/` volume) at regular intervals (target: every 2–5 minutes of wall-clock training time), ensuring no more than a few minutes of progress is lost. Checkpoint paths are registered as metadata in the Model Registry for discovery.
- **Atomic State Transitions**: Loop transitions (e.g., from Training to Eval) are logged as atomic events in the Mission Store to prevent duplicate or inconsistent execution upon resume.

## 5. User Experience & Autonomy Model

### 5.1. The "Goal-First" Interface
The user provides a high-level goal and a success threshold.
- **Example**: `{"task": "SFT", "base_model": "Llama-3.1-8B", "dataset": "customer_logs.jsonl", "target_metric": {"eval_loss": 0.05}}`
- **Example**: `{"task": "RL", "env": "Snake-v0", "target_metric": {"mean_reward": 150}}`

### 5.2. Fully Autonomous Execution
Once the goal is set, ASTRA enters a recursive loop:
1. **Plan**: LLM designs the training trajectory.
2. **Implement**: Generates scripts and configs.
3. **Execute**: Runs in sandbox.
4. **Evaluate**: Compares current metrics against the `target_metric`.
5. **Iterate**: If the goal isn't met, ASTRA self-corrects and repeats from Step 1.

### 5.3. Security & Approval Gates
To balance autonomy with safety, ASTRA implements a **Graduated Autonomy** system:
- **Mandatory Approval**: Irreversible or high-risk actions (e.g., accessing external APIs, deploying to production) require user confirmation.
- **Configurable Gates**: Users can set "Approval Flags" for:
  - **Code Generation**: Reviewing scripts before execution.
  - **Resource Usage**: Approving training runs that exceed a specific cost/time budget.
- **Silent Mode**: In **Supervised** mode only, once a specific sub-task strategy is "trusted" (high success rate in previous iterations), ASTRA can bypass approval gates for that sub-task automatically. Silent Mode does not apply in Guided mode (all gates remain active) and is redundant in Full Autonomy mode (all gates are already suppressed by user choice). See DESIGN §4.2 for the full gate-priority model.
