# astra: Product Requirements Document (PRD)

**Project Name:** astra (**A**utonomous **S**trategic **Tr**aining **A**gent)  
**Status:** Draft  
**Target:** Autonomous Machine Learning Orchestration

---

## 1. Executive Summary
astra is an autonomous agent designed to manage the end-to-end lifecycle of Reinforcement Learning (RL) and Machine Learning (ML) training. It leverages lessons from high-performance Snake and Tetris AI implementations to automate curriculum shifts, reward shaping, and competitive benchmarking.

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
- astra can generate necessary training scripts across various paradigms:
  - **RL**: Environment wrappers, reward functions, and policy networks.
  - **SFT**: Tokenization logic, prompt templates, and supervised loss functions.
  - **ML**: Feature engineering, data loaders, and standard classification/regression architectures.
- Autonomously fixes bugs in training code by analyzing error logs and stack traces.

### 4.8. Secure Sandboxed Execution
- All generated code runs in an isolated sandbox (Docker or restricted subprocess) to protect the host system.
- Resource constraints (CPU/GPU/RAM) are enforced at the sandbox level to prevent runaway processes.

### 4.9. Training Recipes & Crystallization
- **Recipe Generation**: Upon achieving a goal, astra "crystallizes" the successful strategy into a **Training Recipe** (a package containing the optimized hyperparameters, reward/loss logic, curriculum phases, and model architecture).
- **Strategy Sharing**: Recipes are stored in a global library, allowing astra to "warm-start" new, similar goals by retrieving and adapting existing recipes.
- **Recipe Evolution**: astra can treat a recipe as a "DNA" strand, mutating and improving it across different training runs to discover universal "Golden Recipes" for specific domains (e.g., "The Golden Snake Recipe").

### 4.10. Predefined "Golden" Recipes
- astra ships with a set of **Predefined Base Recipes** derived from proven, high-performance training runs (e.g., the workspace's existing Snake and Tetris models).
- These recipes serve as the "Initial Knowledge" of the system, allowing users to achieve expert-level results on day one for common tasks.

## 5. User Experience & Autonomy Model

### 5.1. The "Goal-First" Interface
The user provides a high-level goal and a success threshold.
- **Example**: `{"task": "SFT", "base_model": "Llama-3-8B", "dataset": "customer_logs.jsonl", "target_metric": {"eval_loss": 0.05}}`
- **Example**: `{"task": "RL", "env": "Snake-v0", "target_metric": {"mean_reward": 150}}`

### 5.2. Fully Autonomous Execution
Once the goal is set, astra enters a recursive loop:
1. **Plan**: LLM designs the training trajectory.
2. **Implement**: Generates scripts and configs.
3. **Execute**: Runs in sandbox.
4. **Evaluate**: Compares current metrics against the `target_metric`.
5. **Iterate**: If the goal isn't met, astra self-corrects and repeats from Step 1.

### 5.3. Security & Approval Gates
To balance autonomy with safety, astra implements a **Graduated Autonomy** system:
- **Mandatory Approval**: Irreversible or high-risk actions (e.g., accessing external APIs, deploying to production) require user confirmation.
- **Configurable Gates**: Users can set "Approval Flags" for:
  - **Code Generation**: Reviewing scripts before execution.
  - **Resource Usage**: Approving training runs that exceed a specific cost/time budget.
- **Silent Mode**: Once a strategy is "trusted" (high success rate in previous iterations), astra can bypass gates for that specific sub-task.
