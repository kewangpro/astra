# astra: UX Specification

**Interface Strategy:** A "Mission Control" center for autonomous ML. High-fidelity, data-dense, and real-time.

---

## 1. Design Language
- **Aesthetic**: Dark Mode by default ("Obsidian & Teal").
- **Components**: `shadcn/ui` (Radix UI) for accessibility and polish.
- **Data Viz**: `recharts` for training curves and `react-flow` for the Orchestration DAG.

## 2. Core Views

### 2.1. The "Command Center" (Home)
- **Goal Input**: A command-line style bar (or structured form) to set training targets.
- **Active Missions**: A grid of running training loops with status badges (Planning, Implementation, Sandboxed, Training, Evaluating).
- **Global Metrics**: Total compute hours, models discovered, and successful recipes crystallized.

### 2.2. Live Training HUD (The "Loop" View)
- **The Metric Gap**: A massive visual indicator showing the distance between `current_best` and `user_target`.
- **Streamed Logs**: Real-time telemetry from the sandbox, filtered by LLM-identified "critical events."
- **Strategic Pivot Timeline**: A vertical timeline showing the Lead Agent's decisions (e.g., "09:41 - Plateau detected; Retrying with Phase 2 curriculum").
- **Approval Queue**: Real-time "Pause" state with side-by-side code diffs for user sign-off.

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
5. **Dashboard**: Switches to HUD. Gauge shows "0/200." Curbs start plotting.
6. **Dashboard**: Gauge hits "205/200." Screen flashes "Goal Achieved."
7. **Dashboard**: Prompts to "Crystallize as Tetris-Expert Recipe."
