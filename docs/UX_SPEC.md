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
- **Active Missions**: A grid of training-loop cards with status badges (Planning, Running, Evaluating, Completed, Failed, Stalled — converged below target, escalation exhausted). Each card shows the best metric value, current iteration, and its compute node. Cards are grouped into labelled sections in order — **Running** (any non-terminal status), **Failed**, **Stalled**, **Completed** — each with a count; empty sections are hidden; newest-first within each group.
- **Global Metrics**: A stat row — Total, then Running / Failed / Stalled / Completed mission counts (same order as the card sections).

### 2.2. Live Training HUD (The "Loop" View)
- **The Metric Gap**: An arc gauge showing the all-time best metric value. Gap (`−X to close`) and percentage of target sit directly below the arc. If `target_metric` was not explicitly extracted at mission creation, the component extracts the target percentage directly from the mission goal string rather than falling back to an arbitrary constant. Right column shows two lines: "best at iter N" (which iteration achieved the peak) and the current iteration's score when it differs from the best. This makes it unambiguous whether the displayed score is the historical peak or the latest result.
- **MetricChart**: Training curve capped to the last 3 iteration runs (current + 2 prior). Run-reset boundaries detected from step counter drops. Earlier runs are excluded to prevent chart compression on long-running missions (50+ iterations).
- **Resource Monitor**: A real-time gauge showing **Unified Memory** allocation between the Lead Agent, Specialist Trainer, and System. Total capacity is read dynamically from system info (e.g., 24GB on M4, 64GB+ on higher-tier hardware) and displayed alongside the gauge.
- **Event Stream**: Real-time telemetry events from the sandbox. Pivot events include a `| changes:` suffix showing exactly what changed with real old→new values (e.g. `learning_rate: 0.001→0.0005 | net_arch: [256, 256] | env_kwargs: {food_reward=20.0, distance_weight=0.0}`). No-op pivots (proposed values identical to current) are filtered and shown as a "Pivot skipped" warning instead. For algorithm-locked missions (goal names a specific algorithm), algo-switch proposals are silently dropped and the pivot escalates to reward shaping instead.
- **Live Game Viewers**: For RL missions targeting supported environments (`Snake-v0`, `Tetris-v0`, `Game2048-v0`, `MinAtar-Breakout-v0`, `MinAtar-SpaceInvaders-v0`, `MinAtar-Asteroids-v0`), the Mission HUD embeds an interactive live canvas player (`SnakePlayer`, `TetrisPlayer`, `Game2048Player`, `MinAtarPlayer`). Each connects to `WS /ws/missions/{id}/play?env_id=&fps=`, streaming live environment frames in real time with dedicated palettes and game-tailored FPS sliders (up to 30 FPS).
- **Policy Audit & Explainability Inspector**: Directly embedded in each game player, the `PolicyInspector` provides a live audit drawer showing:
  - Horizontal confidence bars rendering the softmax probability distribution over legal actions.
  - Estimated Q-values for each candidate move.
  - Shannon Policy Entropy ($H$) with dynamic certainty badge: **High Certainty** ($H < 0.3$), **Balanced** ($0.3 \le H \le 0.8$), or **High Exploration** ($H > 0.8$).
  - Real-time indicator highlighting the exact action chosen by the policy.

### 2.3. The Recipe Library (`/recipes`)
- **Gallery View**: Grid of training recipes (both disk-based YAML and DB records) featuring target metric pills, domain badges, and quick YAML inspection modals.
- **Lineage DAG Visualizer**: Interactive drawer tracing the evolutionary chain (Gen 0 → Gen 1 → Gen 2) for crystallized and mutated recipes, displaying score progressions and hyperparameter deltas.
- **One-Click Dispatch**: Single-click "Dispatch" button on any recipe card or inspection modal to instantly spin up an autonomous mission loop without manual configuration.

### 2.4. Model Registry & Tournament Arena (`/models`)
- **Model Registry Table**: Production checkpoint catalog showing model name, domain, architecture, framework, best metric, and active **Champion 👑** status with promotion and deletion controls.
- **Tournament Arena**:
  - Environment and seed selector configuring fixed deterministic evaluation runs across 3 to 20 episodes.
  - Side-by-side simulation evaluating SB3 and Lookahead PyTorch models on identical seeds.
  - Dynamic Leaderboard displaying Gold 🥇, Silver 🥈, and Bronze 🥉 ranks, Mean Score $\pm$ Std, Min/Max range, Win Rate %, and per-episode score breakdowns.
  - "Crown as Champion" action promoting top performers into production status.


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
