# ASTRA: UX Specification

**Interface Strategy:** A "Mission Control" center for autonomous ML. High-fidelity, data-dense, and real-time.

---

## 1. Design Language
- **Aesthetic**: Dark Mode by default ("Obsidian & Teal").
- **Components**: `shadcn/ui` (Radix UI) for accessibility and polish.
- **Data Viz**: `recharts` for training curves and `react-flow` for the Orchestration DAG.

## 2. Core Views

### 2.1. The "Command Center" (Home)
- **Goal Input**: A plain-text input bar for the training goal (e.g. "Train a Snake-v0 PPO agent to achieve mean_reward of 200"). Domain field removed — task type is resolved automatically from the goal text (`taskType: "rl"` hardcoded for current RL-only missions).
- **Active Missions**: A grid of running training loops with status badges (Planning, Running, Evaluating, Completed, Failed). Each card shows the best metric value and current iteration.
- **Global Metrics**: Total compute hours, models discovered, and successful recipes crystallized.

### 2.2. Live Training HUD (The "Loop" View)
- **The Metric Gap**: An arc gauge showing the all-time best metric value. Gap (`−X to close`) and percentage of target sit directly below the arc. Right column shows two lines: "best at iter N" (which iteration achieved the peak) and the current iteration's score when it differs from the best. This makes it unambiguous whether the displayed score is the historical peak or the latest result.
- **MetricChart**: Training curve capped to the last 3 iteration runs (current + 2 prior). Run-reset boundaries detected from step counter drops. Earlier runs are excluded to prevent chart compression on long-running missions (50+ iterations).
- **Resource Monitor**: A real-time gauge showing **Unified Memory** allocation between the Lead Agent, Specialist Trainer, and System. Total capacity is read dynamically from system info (e.g., 24GB on M4, 64GB+ on higher-tier hardware) and displayed alongside the gauge.
- **Event Stream**: Real-time telemetry events from the sandbox. Pivot events include a `| changes:` suffix showing exactly what changed with real old→new values (e.g. `learning_rate: 0.001→0.0005 | net_arch: [256, 256] | env_kwargs: {food_reward=20.0, distance_weight=0.0}`). No-op pivots (proposed values identical to current) are filtered and shown as a "Pivot skipped" warning instead. For algorithm-locked missions (goal names a specific algorithm), algo-switch proposals are silently dropped and the pivot escalates to reward shaping instead.
- **Approval Queue**: Real-time "Pause" state with side-by-side code diffs for user sign-off. Scripts targeting only `localhost`/`127.0.0.1` are auto-approved without LLM classification.
- **Snake-v0 Live Viewer**: For Snake missions, a ▶ Watch button launches a `SnakePlayer` canvas overlay. Connects to `WS /ws/missions/{id}/play`, streams 16×16 grid frames from `best_model.zip` in real time. Displays episode number and cumulative reward. The endpoint reads `best_model_algo.txt` to load the correct SB3 class and passes `env_kwargs` to `gym.make()` so reward configuration matches training.

### 2.3. The Recipe Library
- **Gallery View**: Cards for each Golden Recipe (Snake, Tetris, Llama-SFT).
- **Lineage Map**: A graph showing how recipes evolved from each other.
- **Recipe Editor**: JSON/YAML editor with "Dry Run" validation.

### 2.4. Model Registry & Analysis
- **Leaderboard**: Ranking all models ever trained.
- **Deep-Dive**:
  - **Spatial View**: For CNNs, an interactive board showing saliency maps.
  - **Audit Logs**: Full history of the LLM's reasoning for every pivot in that model's lifecycle.

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
