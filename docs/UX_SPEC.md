# ASTRA: UX Specification

**Interface Strategy:** A "Mission Control" center for autonomous ML. High-fidelity, data-dense, and real-time.

---

## 1. Design Language
- **Aesthetic**: Dark Mode by default ("Obsidian & Teal").
- **Components**: `shadcn/ui` (Radix UI) for accessibility and polish.
- **Data Viz**: `recharts` for training curves and `react-flow` for the Orchestration DAG.

## 2. Core Views

### 2.1. The "Command Center" (Home)
- **Goal Input**: A plain-text input bar for the training goal (e.g. "Train a Snake-v0 PPO agent to achieve mean_reward of 200") paired with a task type selector (`auto (detect)`, `rft`, `distill`, `dpo`, `grpo`, `prompt`, `rl`, `sft`, `ml`, `mlx_lora`). In `auto` mode, the task type is semantically inferred from keywords in the goal text, and backend reconciliation ensures that submitted defaults never misdirect fine-tuning or distillation missions into RL.
- **Operational Board**: A row-based layout of active training-loop cards organized into full-width horizontal sections in workflow priority order:
  - **Running / Active** (teal indicator with live pulse animation when runs are active; displays a subtle compact indicator when 0 runs are in-flight).
  - **Stalled / Paused** (orange indicator; rendered dynamically when stalled runs exist, hidden when 0). Stalled cards show `error_log` and **do not** offer Resume — the search is finished; start a new mission. Paused cards still offer Resume.
  - **Failed** (red indicator; rendered dynamically when failed runs exist, hidden when 0).
  Within each active row, missions are arranged in a responsive grid (`grid-cols-1 md:grid-cols-2 lg:grid-cols-3`), the same breakpoints as `/completed` and `/recipes`. Historical completed runs are cleanly moved to the dedicated Completed Missions Archive (`/completed`) to preserve operational clarity.
- **Global Metrics**: A top-right stat row showing active operational counts: `Total`, `Running`, `Stalled`, and `Failed`.

### 2.2. Live Training HUD (The "Loop" View)
- **The Metric Gap**: An arc gauge showing the all-time best metric value. Gap (`−X to close`) and percentage of target sit directly below the arc. For lower-is-better metrics (`eval_loss`, `train_loss`, `perplexity`), the component computes progress inversely (tracking loss reduction toward the target threshold), showing positive convergence progress rather than inverted percentages. If `target_metric` was not explicitly extracted at mission creation, the component extracts the target percentage directly from the mission goal string rather than falling back to an arbitrary constant. Right column shows two lines: "best at iter N" (which iteration achieved the peak) and the current iteration's score when it differs from the best. This makes it unambiguous whether the displayed score is the historical peak or the latest result.
- **MetricChart**: Training curve capped to the last 3 iteration runs (current + 2 prior). Run-reset boundaries detected from step counter drops. Earlier runs are excluded to prevent chart compression on long-running missions (50+ iterations).
- **Resource Monitor**: A real-time gauge showing **Unified Memory** allocation between the Lead Agent, Specialist Trainer, and System. Total capacity is read dynamically from system info (e.g., 24GB on M4, 64GB+ on higher-tier hardware) and displayed alongside the gauge.
- **Event Stream**: Real-time telemetry events from the sandbox. Pivot events include a `| changes:` suffix showing exactly what changed with real old→new values (e.g. `learning_rate: 0.001→0.0005 | net_arch: [256, 256] | env_kwargs: {food_reward=20.0, distance_weight=0.0}`). The HUD Pivot History lists `type=pivot` only (the subtitle falls back to "plateau detected" if `reason` is missing); revert is a separate `warn` (`Pivot reverted — restored checkpoint from iter N`). Command Center "pivots" is DB `pivot_escalation_count`, which reverts decrement — it is not the HUD event count. No-op pivots (proposed values identical to current) are filtered and shown as a "Pivot skipped" warning instead. For algorithm-locked missions (goal names a specific algorithm, including `SB3 PPO` as PPO), algo-switch proposals are silently dropped and the pivot escalates to reward shaping instead. Unnamed "RL agent" missions may show a real `algo: PPO→DQN` at level 2+, and a later `algo: PPO→A2C` at level 3+, when the loop forces a different trainer rather than an alias rename.
- **Live Game Viewers**: Not on this page. Missions are training (metrics, logs, pivots). Play / inference is `/models/{id}`.

### 2.3. The Recipe Library (`/recipes`)
- **Gallery View**: Grid of training recipes (both disk-based YAML and DB records) featuring target metric pills, domain badges, and quick YAML inspection modals.
- **Lineage DAG Visualizer**: Interactive drawer tracing the evolutionary chain (Gen 0 → Gen 1 → Gen 2) for crystallized and mutated recipes, displaying score progressions and hyperparameter deltas.
- **One-Click Dispatch**: Single-click "Dispatch" button on any recipe card or inspection modal to instantly spin up an autonomous mission loop without manual configuration.
- **Delete**: Auto-crystallized recipes (not hand-crafted YAML) have a Delete control on the card and in the YAML modal.

### 2.4. Models (`/models`)
- Missions train; models play / infer.
- **Checkpoint cards**: Click to open `/models/{id}` for live play (`WS /ws/models/{id}/play`).
- **Model page**: Canvas + policy inspector + checkpoint dossier (metric, training mission link, path). MinAtarPlayer covers Breakout, Space Invaders, Asteroids, Asterix, Freeway, Seaquest, and Grid Pac-Man.
- **Policy inspector**: confidence bars, Q-values, entropy, selected action.
- **Tournament Arena**:
  - Environment and seed selector configuring fixed deterministic evaluation runs across 3 to 20 episodes.
  - Side-by-side simulation evaluating SB3 and Lookahead PyTorch models on identical seeds.
  - Dynamic Leaderboard displaying Gold 🥇, Silver 🥈, and Bronze 🥉 ranks, Mean Score $\pm$ Std, Min/Max range, Win Rate %, and per-episode score breakdowns.
  - "Crown as Champion" action promoting top performers into production status.

### 2.5. The Completed Missions Archive (`/completed`)
- **Gallery Layout**: Modeled after the Recipe Library (`/recipes`) with dark glass UI, KPI overview cards, and responsive 3-column card grid (`grid-cols-1 md:grid-cols-2 lg:grid-cols-3`).
- **Domain Filter Tabs**: 8 category tabs (`All Completed`, `MinAtar`, `Snake`, `Tetris`, `2048`, `LLM & Reasoning`, `AgentGym`, `Classic Control & ML`).
- **Interactive Search & Sorting**: Real-time filtering across goals, IDs, environments, hosts, and task types; sortable by newest, highest metric score, or fastest duration.
- **KPI Summary Cards**: Displays Total Completed Runs, Target Pass Rate (% meeting or exceeding target), and Total Iterations Executed across historical runs.
- **Convergence Cards**: Shows short ID, task type tag, remote/local node host, relative/exact timestamp, goal text, target metric vs achieved best score, progress bar, iterations, and duration. Clicking a card opens the Mission HUD.



## 3. Technology Stack Recommendation

| Layer | Choice | Rationale |
|---|---|---|
| **Frontend** | **Next.js 15 (App Router)** | Performance, SEO (for public models), and excellent SSE/WebSocket support. |
| **Styling** | **Tailwind CSS** | Rapid, consistent design system implementation. |
| **UI Components** | **shadcn/ui** | Highly customizable, professional accessible primitives. |
| **State** | **React Query** | For caching telemetry and registry data with automatic background refresh. |
| **Real-time** | **FastAPI + WebSockets** | Lightweight, high-speed bidirectional communication for the HUD. |

## 4. User Interaction Flow
1. **User**: Input goal: "Master Tetris with 200 lines."
2. **Dashboard**: Shows the Lead Agent "Thinking..." (Planning phase).
3. **Dashboard**: Prompts for `EXECUTE_CODE` approval with a preview of the generated PyTorch script.
4. **User**: Clicks "Approve & Start."
5. **Dashboard**: Switches to HUD. Gauge shows "0/200." Curves start plotting.
6. **Dashboard**: Gauge hits "205/200." Screen flashes "Goal Achieved."
7. **Dashboard**: Prompts to "Crystallize as Tetris-Expert Recipe."
