# ASTRA: Design Document

**Architecture Version:** 1.0.0  
**Core Stack:** Python, PyTorch, SQLAlchemy, FastAPI, Next.js 15

Phase-level incidents and code contracts live in [IMPLEMENT.md](IMPLEMENT.md).

---

## 1. System Overview

ASTRA is a modular system: a **Lead Agent** plans, a **loop** executes, **specialist trainers** run in a **sandbox**, and a **Next.js HUD** observes. Missions train; models play.

```
                    +-----------------------+
                    |    Next.js Web UI     |
                    +-----------+-----------+
                                |
                                | HTTP / WebSocket
                                v
+-------------------+       +-----------+-----------+       +-----------------------+
| Live Training HUD |<----->|  FastAPI Orchestrator | <---> |    Memory / Registry  |
+-------------------+       +-----------+-----------+       +-----------------------+
                                |           |
                                v           v
                    +-----------------------+     +-----------------------+
                    |  Lead Agent / Planner |     |  Specialist Trainer   |
                    +-----------------------+     +-----------+-----------+
                                                                    |
                                                                    v
                                                          +-----------------------+
                                                          |    Secure Sandbox     |
                                                          +-----------+-----------+
                                                                    |
                                                                    v
                                                          +-----------------------+
                                                          |      Environment      |
                                                          +-----------------------+
```

---

## 2. Components

### 2.1. Lead Agent

The planner. Cloud APIs are supported; production is **local MLX** on Apple Silicon (plan model + coder model). Inference shares unified memory with training, so the orchestrator must be able to evict models before a sandbox launch.

Worth optimizing: context/KV discipline, structured (schema-constrained) generation, and not loading Metal at import time on hosts without a GPU. Core matmul and quantization are left to MLX/MPS.

**Deployed split (both machines 24 GB unified memory):**
- **MacBook** — planning and codegen inference.
- **mac-mini.local** — training host (scripts over SSH, checkpoints and logs back on completion).

### 2.2. Autonomous Training Loop

Plan → generate script → sandbox → evaluate → pivot or stop.

- **Curriculum** advances when metrics justify it.
- **Pivots** escalate when a plateau holds: hyperparameters, then architecture, then algorithm (unless the goal named one), then reward shape. Deep plateau plus a long stretch without a new best is **converged below target** (terminal `stalled`). Reverting a bad architecture restores the checkpoint but still counts as a failed search.
- **Unnamed RL goals** take the env recipe’s algorithm; a named trainer stays locked.
- **Warm-start** loads a prior checkpoint only when algorithm and policy tensor shapes match. A smaller net than the best-known architecture is refused until deep plateau.
- **Fine-tune-remote** missions (`dpo` / `grpo` / `distill` / `rft` / `sft` / `prompt`) keep the recipe authoritative. Pivots may only touch a small per-type sampling or duration safelist so a LoRA warm-start cannot be broken by a hallucinated learning rate or layer count.

### 2.3. Memory

- **SQL registry** — experiments, models, mission state, recipes.
- **Vector memory** — lessons with enough metadata to retrieve by regime (env, HP, run).
- **Recipe library** — versioned YAML strategies, indexed for warm-start. Fine-tune dispatch uses canonical recipes only (no per-mission crystallization).
- **Working memory** — live logs and telemetry in the planner context.

Targets that exceed a recipe’s declared empirical ceiling are rejected at create time.

### 2.4. Specialist trainers

`task_type` selects the worker. LoRA is a mechanism, not a task type. `prompt` changes no weights.

| Type | Intent |
|---|---|
| `rl` | Policy from reward in a Gym env (Snake, Tetris, 2048, MinAtar suite, Grid Pac-Man, AgentGym, or a standard Gymnasium id). SB3, or a custom lookahead / actor-critic trainer where the env needs it. |
| `sft` | Supervised fine-tune on labeled completions (held-out split, optional CoT traces). Local HF/PEFT or remote MLX. |
| `ml` | Tabular / classical ML. |
| `mlx_lora` | Local Apple-Silicon LoRA. |
| `dpo` | Preference pairs; no separate reward model. |
| `grpo` | On-policy group-relative policy gradient. |
| `distill` | Teacher completions → student SFT. Not bounded by the student’s own plateau the way DPO/GRPO are. |
| `rft` | Rejection-sample the student itself; SFT on survivors. No teacher. |
| `prompt` | Append routing rules to a **copy** of the conductor prompt and score. Production prompt is never edited. |
| `star` | Self-taught reasoner: rollouts, then hint-guided rationalization, then LoRA. |

**Environments.** Arcade envs are 10×10 NumPy Gym wrappers. Seaquest, Asterix, and Grid Pac-Man follow their source rules closely enough that play and checkpoints stay honest (oxygen/surface, spawn/ramp, leftover ghosts chase, mouth faces movement). Env classes load lazily so importing the package does not load every game.

**Code generation.** The coder writes `train.py`. Mechanical post-patches exist because a 7B coder will omit imports, `register()`, or `gym.make`, and a healer that treats every `NameError` as a missing import can swap in the wrong env. SFT falls back to a canonical runner when the AST is bad. RL scripts pin `gym.make` to the planned env id.

**Remote fine-tune.** The training host has no git checkout of ASTRA. A thin `os.execv` wrapper replaces itself with the standalone trainer. Batch size is clamped to the split. The loop tails the remote log for HUD metrics. Loss-like metrics are lower-is-better.

**Chaining.** SFT → DPO (and similar) is two missions. The second warm-starts the first’s adapter; trainers read LoRA shape from the adapter config rather than trusting the planner.

### 2.5. Sandbox

Where training runs.

- **Apple Silicon** — no Metal passthrough in Docker. GPU work is a restricted host subprocess (or SSH to the Mini). Docker is for CPU/isolation only.
- **CUDA** — container with GPU toolkit.
- **Isolation** — memory/compute caps; writes limited to mission data and the registry.

### 2.6. Evaluator and introspection

A **benchmark suite** (golden challenges) and **stress** cases sit outside the training loop. **Saliency** and a **policy auditor** (action distribution, entropy) explain play on `/models/{id}`.

### 2.7. Resilience

- **Safety critic** — GAN-style review of the plan before execute.
- **Mission manifest** — structured source of truth for the next iteration, not a growing chat log.
- **Validation contract** — primary metric plus health signals (entropy, loss stability).

---

## 3. Data flow

1. User (or recipe dispatch) states a goal and target.
2. Lead Agent warm-starts from the recipe library when a close match exists.
3. Critic approves or sends the plan back.
4. Codegen writes a sandbox script; autonomy gates may pause for execute-code approval.
5. Sandbox trains; telemetry streams to the HUD.
6. Evaluator scores the checkpoint against the goal metric.
7. Loop pivots, completes, or stalls. Successful **RL** (and similar) runs may crystallize a recipe; fine-tune-remote types do not.
8. Registry and model page pick up the checkpoint for play and tournaments.

---

## 4. Security and autonomy

One gate is live: **execute code**. A static pass auto-approves localhost telemetry and blocks obvious danger; ambiguous scripts go to an LLM classifier, then a human if needed. Resource and deploy gates are modeled, not wired.

| Mode | Execute-code gate |
|---|---|
| **Guided** | Always a human decision (UI approve or explicit auto-approve click). |
| **Supervised** (default) | Classifier may auto-approve; otherwise wait. |
| **Full autonomy** | No gate; script runs. |

The HUD shows loop status, metric gap, pivot history, and the approval queue.

---

## 5. Runtime

**Persistent.** FastAPI process, asyncio mission loops, SQLite (or Postgres), Chroma sidecar, `data/` volume for weights and logs.

**Transient.** One sandbox per training iteration (subprocess, SSH, or container). The model manager keeps LLM and trainer from fighting over unified memory.

**Recovery.** On boot, running/planning/evaluating missions are inspected. A live sandbox is reattached and only polled. A dead sandbox resets the mission to pending so the loop can relaunch from the last checkpoint. Shutdown terminates the sandbox so the next process does not inherit an orphan. Telemetry back-fills the HUD after a reconnect.

Interactive API: `http://localhost:8200/docs`.
