# ASTRA: UX Specification

**Interface Strategy:** A "Mission Control" center for autonomous ML. High-fidelity, data-dense, and real-time.

---

## 1. Design Language
- **Aesthetic**: Dark Mode by default ("Obsidian & Teal").
- **Components**: `shadcn/ui` (Radix UI) for accessibility and polish.
- **Data Viz**: `recharts` for training curves and `react-flow` for the Orchestration DAG.

## 2. Core Views

### 2.1. The "Command Center" (Home)
- **Goal Input**: A plain-text input bar for the training goal (e.g. "Train a Snake-v0 PPO agent to achieve mean_reward of 200") paired with an algorithm selector grouped by task: Environment RL (PPO, DQN, A2C, SAC, TD3), reinforcement fine-tuning (GRPO, DPO), Agentic RL (Agent GRPO), supervised fine-tuning (SFT, local LoRA, distillation, rejection sampling, STaR), on-policy distillation (OPD), calibrated decision (RLCD), prompt edit (rule append), classical ML (random forest), retrieval (embedding, reranker), and generative recommendation (HSTU), plus `auto (detect)`. In `auto` mode, the worker is inferred from keywords in the goal text (`rlcd` or "calibrated decision" / "decision foundation" selects RLCD; `hstu` / MovieLens / generative recommend selects the from-scratch HSTU trainer), and backend reconciliation ensures that submitted defaults never misdirect fine-tuning or distillation missions into RL. **Product fine-tunes:** name the product in the prompt or include `use_case: <module>` (e.g. Ensemble → `use_case: ensemble`); that binds the `UseCasePlugin` at mission create — recipes supply training HPs only. Full create ↔ recipe ↔ product wiring: [DESIGN.md §3](DESIGN.md). RLCD uses product recipes (routing-only or foundation → specialize → focus); Mini trains dispatch only via Astra recipe missions. Held-out pass_rate is not the promote gate.
- **Operational Board**: A row-based layout of active training-loop cards organized into full-width horizontal sections in workflow priority order:
  - **Running / Active** (teal indicator with live pulse animation when runs are active; displays a subtle compact indicator when 0 runs are in-flight).
  - **Stalled / Paused** (orange indicator; rendered dynamically when stalled runs exist, hidden when 0). Stalled cards show `error_log` and **do not** offer Resume — the search is finished; start a new mission. Paused cards still offer Resume.
  - **Failed** (red indicator; rendered dynamically when failed runs exist, hidden when 0).
  Within each active row, missions are arranged in a responsive grid (`grid-cols-1 md:grid-cols-2 lg:grid-cols-3`), the same breakpoints as `/completed` and `/recipes`. Historical completed runs are cleanly moved to the dedicated Completed Missions Archive (`/completed`) to preserve operational clarity.
- **Global Metrics**: A top-right stat row showing active operational counts: `Total`, `Running`, `Stalled`, and `Failed`.

### 2.2. Live Training HUD (The "Loop" View)
- **The Metric Gap**: An arc gauge showing the all-time best metric value. Gap (`−X to close`) and percentage of target sit directly below the arc. For lower-is-better metrics (`eval_loss`, `train_loss`, `perplexity`), the component computes progress inversely (tracking loss reduction toward the target threshold), showing positive convergence progress rather than inverted percentages. If `target_metric` was not explicitly extracted at mission creation, the component extracts the target percentage directly from the mission goal string rather than falling back to an arbitrary constant. Right column shows two lines: "best at iter N" (which iteration achieved the peak) and the current iteration's score when it differs from the best. This makes it unambiguous whether the displayed score is the historical peak or the latest result. While a remote LM fine-tune is `running`, a **live eval** subtitle shows the latest tailed `pass_rate` (and step when present) without moving the arc until iteration bookkeeping updates the DB.
- **Metric History**: Training-signal chart (`loss` / `mean_reward`). For DPO/GRPO-style fine-tunes, mid-run `pass_rate` evals are dual-charted with `loss` (SSH-tailed per-step loss — including GRPO `Step N/M | loss=…` and negative sentinel losses — not only DPO epoch `avg_loss`). The Y-axis stays **raw/adaptive** on dual charts so loss `0.70` is not labeled `70%`; tooltips still format `pass_rate` as percent. Legend lists series names (`loss`, `pass_rate`) plus current/prior run keys. Epoch `avg_loss` and synthetic end-of-iter `pass_rate` points at `step ≥ 500000` are omitted from this chart (Metric Gap still buckets by iteration). Empty state (no points yet) may surface the latest mirrored `sandbox.log` line or lifecycle status; live train narrative belongs in the **Event Stream**, not the chart header.
- **MetricChart**: Training curve capped to the last 3 iteration runs (current + 2 prior). Run-reset boundaries prefer the telemetry `iteration` field (fallback: step decrease). Cross-run `runOffset` advances by the **max step in the completed run**, not the last event’s step (avoids overlapping loss curves when a run ends on a low-step `pass_rate`).
- **Resource Monitor**: A real-time gauge showing **Unified Memory** allocation between the Lead Agent, Specialist Trainer, and System. Total capacity is read dynamically from system info (e.g., 24GB on M4, 64GB+ on higher-tier hardware) and displayed alongside the gauge.
- **Event Stream**: Real-time telemetry from the mission WebSocket (JSONL back-fill on connect). For remote LM fine-tunes the primary narrative is **mirrored Mini `sandbox.log`** (`type=log`, noise-filtered; consecutive `[i/n]` / `Step N/M` progress lines collapse with “+N similar”). Orchestrator lifecycle stays `info` / `success` / `warn` / `error`. Sparse fine-tune metrics (`pass_rate`, `loss`, …) still appear; high-frequency RL metrics (`mean_reward`, …) stay on Metric History only. Visibility does **not** depend on a status-regex whitelist — unknown trainer banners still show. Pivot events include a `| changes:` suffix with real old→new values. The HUD Pivot History lists `type=pivot` only (subtitle falls back to "plateau detected" if `reason` is missing); revert is a separate `warn`. Command Center "pivots" is the persisted consecutive-failed-pivot count, not the HUD event count. No-op pivots are filtered as "Pivot skipped". Algorithm-locked missions drop algo-switch proposals; unnamed "RL agent" missions may show a real `algo: PPO→DQN` at level 2+.
- **Live Game Viewers**: Not on this page. Missions are training (metrics, logs, pivots). Play / inference is `/models/{id}`.

### 2.3. The Recipe Library (`/recipes`)
- **Gallery View**: Grid of training recipes (disk YAML from platform `recipes/` **plus** configured product packs via `ASTRA_PRODUCT_RECIPES`, merged with DB records) featuring target metric pills, an algorithm tag (embedding, HSTU, GRPO, DQN, PPO, …), a source badge (**Astra built-in** or the product `use_case` key), and quick YAML inspection modals. Tabs are `All Recipes` plus each task group. A recipe stays on a tab when its YAML `task_type` is one of that task’s workers, so Snake, MinAtar, 2048, and AgentGym share Environment RL. Product cards appear only when the Orchestrator can see that pack path.
- **Source filters**: Checkboxes for **Astra built-in** and each configured product pack. Product labels come from `GET /recipes/packs` (`ASTRA_PRODUCT_RECIPES` use_case keys — not a hardcoded brand list); each label shows its recipe count, and the row shows how many cards match the current filters. Unchecking a source hides those cards without changing task tabs or search.
- **Lineage DAG Visualizer**: Interactive drawer tracing the evolutionary chain (Gen 0 → Gen 1 → Gen 2) for crystallized and mutated recipes, displaying score progressions and hyperparameter deltas.
- **One-Click Dispatch**: "Dispatch" on a recipe card/modal calls the same `MissionCreate` pipeline as the goal bar (`POST /recipes/{name}/dispatch`). Goal text is canonicalized from the recipe description + target when omitted; recipe YAML seeds `task_type` and training HPs. For product fine-tune recipes the UI prompts for the **use-case plugin module**, defaulting to the recipe’s `product` field when set, and sends it as `use_case` on create — not baked into recipe HP; create/goal binding wins over Lead Agent (never treat `mlx_dpo` / other `learner_id`s as the product). See [DESIGN.md §3](DESIGN.md).
- **Delete**: Auto-crystallized recipes (not hand-crafted YAML) have a Delete control on the card and in the YAML modal.

### 2.4. Models (`/models`)
- Missions train; models play / infer.
- **Checkpoint cards**: Click to open `/models/{id}` for live play (`WS /ws/models/{id}/play`).
- **Model page**: Canvas + policy inspector + checkpoint dossier (metric, training mission link, path). MinAtarPlayer covers Breakout, Space Invaders, Asteroids, Asterix, Freeway, Seaquest, and Grid Pac-Man (mouth rotates with movement).
- **Policy inspector**: confidence bars, Q-values, entropy, selected action.
- **Tournament Arena**:
  - Environment and seed selector configuring fixed deterministic evaluation runs across 3 to 20 episodes.
  - Side-by-side simulation evaluating SB3 and Lookahead PyTorch models on identical seeds.
  - Dynamic Leaderboard displaying Gold 🥇, Silver 🥈, and Bronze 🥉 ranks, Mean Score $\pm$ Std, Min/Max range, Win Rate %, and per-episode score breakdowns.
  - "Crown as Champion" action promoting top performers into production status.

### 2.5. The Completed Missions Archive (`/completed`)
- **Gallery Layout**: Modeled after the Recipe Library (`/recipes`) with dark glass UI, KPI overview cards, and responsive 3-column card grid (`grid-cols-1 md:grid-cols-2 lg:grid-cols-3`).
- **Task Filter Tabs**: `All Completed`, then each catalog task (Environment RL, reinforcement fine-tuning, Agentic RL, supervised fine-tuning, on-policy distillation, calibrated decision, prompt edit, classical ML, retrieval, generative recommendation). A tab keeps missions whose `task_type` is one of that task’s workers. Snake, MinAtar, and AgentGym MLP PPO sit under Environment RL (`rl`); Agent GRPO missions sit under Agentic RL (`agrpo`).
- **Interactive Search & Sorting**: Real-time filtering across goals, IDs, environments, hosts, and task types; sortable by newest, highest metric score, or fastest duration.
- **KPI Summary Cards**: Displays Total Completed Runs, Target Pass Rate (% meeting or exceeding target), and Total Iterations Executed across historical runs.
- **Convergence Cards**: Shows short ID, task type tag, remote/local node host, relative/exact timestamp, goal text, target metric vs achieved best score, progress bar, iterations, and duration. Clicking a card opens the Mission HUD.

### 2.6. Docs (`/docs`)

Nav item after Models. A task index on the left, one section per task. Each section states the objective, then one row per algorithm with its worker key (`task_type`) and what the update does. Copy distinguishes **online** (fresh policy rollouts during training) vs **offline** (fixed logged / preference / JSONL data). Task sections match the goal-bar groups, including **Agentic RL** (Agent GRPO / `agrpo` — multi-turn tool trajectories).

Complete flywheel: gym-coupled RL is **AstraGym** (`gym_id`) × **AstraRL** (`learner_id`); offline LM is **AstraTune** (`mlx_sft` / `mlx_dpo` / …); fuel/truth are **AstraData** (`dataset_id@rev`) + **AstraEval** (`suite_id@rev` + FailSlice). Astra is Orchestrator. AstraRL owns the platform GRPO runtime (rollout / DAPO-lite / loss); AstraTune owns SFT/DPO/distill/RLCD math and **shared** `lora_io` (also imported by AstraRL MLX trainers). Products plug in via `astra_plugin` (routing reward, preference pairs, decision cases). Named in the relevant `/docs` task summaries.



## 3. Technology Stack Recommendation

| Layer | Choice | Rationale |
|---|---|---|
| **Frontend** | **Next.js 15 (App Router)** | Performance, SEO (for public models), and excellent SSE/WebSocket support. |
| **Styling** | **Tailwind CSS** | Rapid, consistent design system implementation. |
| **UI Components** | **shadcn/ui** | Highly customizable, professional accessible primitives. |
| **State** | **React Query** | For caching telemetry and registry data with automatic background refresh. |
| **Real-time** | **FastAPI + WebSockets** | Lightweight, high-speed bidirectional communication for the HUD. |

## 4. User Interaction Flow

**Goal-first (Command Center)**  
1. **User**: Input goal (and optional algorithm). For product LM runs, name the product or `use_case: <module>`.
2. **Dashboard**: Lead Agent plans from the goal (recipe library may warm-start knobs).
3. **Dashboard**: `EXECUTE_CODE` approval with script preview (autonomy mode dependent).
4. **User**: Approves when required → HUD streams metrics until target or stall/pivot.

**Recipe-first (`/recipes`)**  
1. **User**: Dispatch a recipe; for fine-tunes, enter the product plugin module when prompted.
2. Same create → plan/seed → codegen → sandbox loop as goal-first ([DESIGN.md §3](DESIGN.md)).

**Example (RL)**  
1. Goal: "Master Tetris with 200 lines." → plan → approve → HUD 0/200 → goal achieved → optional crystallize.
