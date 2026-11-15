# ASTRA: Design Document

**Architecture Version:** 1.0.0  
**Core Stack:** Python, PyTorch, SQLAlchemy, FastAPI, Next.js 15

Phase-level incidents and code contracts live in [IMPLEMENT.md](IMPLEMENT.md).

---

## 1. System Overview

ASTRA is a modular system: a **Lead Agent** plans, a **loop** executes as **Orchestrator**, **specialist trainers** run in a **sandbox**, and a **Next.js HUD** observes. Missions train; models play.

**Interactive / gym-coupled RL** uses **AstraGym** (`gym_id` worlds) and **AstraRL** (`learner_id` learners) under that Orchestrator (§ 2.4–2.5). That includes off-policy SB3 (DQN/SAC/TD3 replay). **Offline / dataset LM post-training** (SFT, DPO, distill, RFT, STAR, RLCD) uses the sibling **AstraTune** package — not AstraRL. **AstraData** + **AstraEval** close the complete flywheel (fuel → train → truth → FailSlice → fuel).

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
                    platform pkgs (local .venv or Mini ~/astra-pkg)      v
          +--------+--------+----------+----------+----------+
          | AstraGym | AstraRL | AstraTune | AstraData | AstraEval |
          |  gym_id  | learner |  learner  | dataset@rev| suite@rev |
          +----+-----+---+----+----+-----+-----+----+----+--+
               |         |     ^     |           |
               v         v     |lora |           v
        worlds/rewards SB3/GRPO| I/O  SFT/DPO/…  suite gate +
                       (gym)   |     distill/…   FailSlice bank
```

---

## 2. Components

### 2.1. Lead Agent

The planner. Cloud APIs are supported; production is **local MLX** on Apple Silicon (plan model + coder model). Inference shares unified memory with training, so the orchestrator must be able to evict models before a sandbox launch.

Inference practice: keep context/KV disciplined, prefer structured (schema-constrained) generation, and avoid loading Metal at import time on hosts without a GPU. Core matmul and quantization stay with MLX/MPS.

**Deployed split (both machines 24 GB unified memory):**
- **MacBook** — planning and codegen inference.
- **mac-mini.local** — training host (scripts over SSH, checkpoints and logs back on completion).

### 2.2. Autonomous Training Loop

Plan → generate script → sandbox → evaluate → pivot or stop.

- **Curriculum** advances when metrics justify it.
- **Pivots** escalate when a plateau holds: hyperparameters, then architecture, then algorithm (unless the goal named one), then reward shape. Deep plateau plus a long stretch without a new best is **converged below target** (terminal `stalled`). Reverting a bad architecture restores the checkpoint but still counts as a failed search.
- **Unnamed RL goals** take the env recipe’s algorithm; a named trainer stays locked.
- **Warm-start** loads a prior checkpoint only when algorithm and policy tensor shapes match. A smaller net than the best-known architecture is refused until deep plateau.
- **Fine-tune-remote** missions keep the recipe authoritative. Gym-coupled LM workers (`grpo` / `agrpo` / `opd`) dispatch via `astrarl.make(..., use_execv=True)`; offline LM workers (`sft` / `dpo` / `distill` / `rft` / `star` / `rlcd`) via `astratune.make(..., use_execv=True)`. Pivots may only touch a small per-type sampling or duration safelist so a LoRA warm-start cannot be broken by a hallucinated learning rate or layer count. Recipes one-click dispatch fills empty `target_metric` from recipe `metric_ceiling` / `target_metric`.

### 2.3. Memory

- **SQL registry** — experiments, models, mission state, recipes.
- **Vector memory** — lessons with enough metadata to retrieve by regime (env, HP, run).
- **Recipe library** — versioned YAML strategies, indexed for warm-start. Fine-tune dispatch uses canonical recipes only (no per-mission crystallization).
- **Working memory** — live logs and telemetry in the planner context.

Targets that exceed the **suite** ceiling (when `eval_suite` is set) or else the recipe’s `metric_ceiling` are rejected at create time.

### 2.4. Complete flywheel

Fuel → train → truth → FailSlice → fuel. **AstraData** versions fuel (`dataset_id@rev`); train packages produce checkpoints; **AstraEval** scores them (`suite_id@rev`) and emits FailSlices back into Data; the Orchestrator suite-gates completion and may inject `focus_cases` on the next turn.

**Wiring** (solid = forward path; labeled = Eval feedback):

```mermaid
flowchart LR
  Orch[Orchestrator]
  Plugin[astra_plugin]
  Packs[Product_packs]
  Data[AstraData]
  Gym[AstraGym]
  RL[AstraRL]
  Tune[AstraTune]
  Eval[AstraEval]
  Neither[Neither_hstu_embed]
  Orch --> Gym
  Orch --> RL
  Orch --> Tune
  Orch --> Neither
  Orch --> Plugin
  Packs --> Plugin
  Data --> RL
  Data --> Tune
  Data --> Neither
  Gym --> RL
  RL --> Eval
  Tune --> Eval
  Neither --> Eval
  Eval -->|"FailSlice"| Data
  Eval -->|"suite_gate"| Orch
```

| Layer | Role on the flywheel |
|---|---|
| **AstraData** | Fuel + FailSlice bank (`dataset_id@rev`) |
| **Train** | AstraGym × AstraRL, AstraTune, or Neither (`hstu` / `embed` / …) |
| **AstraEval** | Truth: `suite.run(ckpt) → {primary, case_ids}`, ceilings, `tournament_seed` |
| **Orchestrator** | Stamps `dataset_*` / `suite_*`, suite gate, FailSlice → optional `focus_cases` |

**Bind axes (mission create):** `task_type` · `gym_id`×`learner_id` · `use_case` · `dataset_id@rev` · `suite_id@rev`. Orchestrator stamps Data/Eval from the recipe. Shipped proving verticals: HSTU (`HstuHr10Suite`) and Ensemble routing (`RoutingBareEvalSuite` + FailSlice → `focus_cases`).

Platform packages, product plugins, and recipe packs are § 2.5 — not part of the flywheel loop itself.

### 2.5. Platform and product

**Astra = platform and training center** (contracts + math + train CLIs/loops + facades + Data/Eval). **Products = use cases** (plugins, prompts, eval data, live app) — not a second training host. Inspired by Gym×RL product splits; **not** NVIDIA NeMo / HTTP env servers. Every **interactive / gym-coupled** RL path is AstraGym × AstraRL (including off-policy SB3). Every **offline / dataset LM** path is AstraTune — do not fold DPO/SFT into AstraRL.

Train path / plugin boundary:

```
+----------------------------------+          +----------------------------------+
|  Astra platform + training center |          |         Product use case         |
+----------------------------------+          +----------------------------------+
| Mission loop / recipes / sandbox |          | Use-case plugin (--use-case)     |
| AstraData / AstraEval            |  facade  |   score / case_pool / oracle     |
|   +--------+--------+            |  execv   |   prompts / eval cases           |
|   v                 v            |  -m cli  |   adapters / bare_eval (product) |
| AstraRL             AstraTune    |--------->|                                  |
| cli + train_*       cli + train_*|          | Reward / Preference / Dataset    |
| rollout/DAPO/math   losses/CE    |          | hooks via UseCasePlugin          |
|            ^                     |          |                                  |
|            |  lora_io (shared)   |          | (no product train CLIs)          |
| AstraGym worlds (gym_id)         |          |                                  |
+----------------------------------+          +----------------------------------+
```

| Layer | Package / role | Owns |
|---|---|---|
| **AstraGym** | `astragym/` | Worlds: `reset` / `step`, rewards, eval (`gym_id`) |
| **AstraRL** | `astrarl/` | Online learners + **`python -m astrarl.cli`** train entrypoints; rollout, DAPO-lite, math |
| **AstraTune** | `astratune/` | Offline LM + **`python -m astratune.cli`** train entrypoints; losses, `ce_loop`, shared `lora_io` |
| **AstraData** | `astradata/` | Fuel: `dataset_id@rev` registry, hash/freeze, splits, FailSlice bank, HSTU builders |
| **AstraEval** | `astraeval/` | Truth: `suite.run(ckpt) → {primary, case_ids}`, ceilings, FailSlice emit, tournament seeds |
| **astra_plugin** | `astra_plugin/` | Use-case contract: `UseCasePlugin`, `load_use_case`, per-learner hook attrs, `validate_hooks` |
| **Orchestrator** | Astra loop / platform `recipes/` + product packs / sandbox / HUD | Missions, pivots, `task_type` worker keys, stamp `dataset_*`/`suite_*`, suite gate, telemetry, canned facade codegen |
| **Product (use case)** | importable plugin module | Implements `UseCasePlugin`; oracle / cases / prompts / adapters / `bare_eval` — **no** train CLIs; does **not** own suite IDs or dataset revs |

| Concern | Astra owns | Product owns |
|---|---|---|
| **Contracts** | `Learner`, gym ids, **`astra_plugin`**, recipe schema | Plugin module implements `UseCasePlugin` + hook bags |
| **Runtime** | Train CLIs, `run_*` loops, math, `lora_io` | Hooks via mission-create / plan `--use-case <module>` (env `ASTRA_USE_CASE` last resort) → `astra_plugin.load_use_case` |
| **Orchestration** | Missions, SSH, telemetry, pivots, facade dispatch | Consume platform (no parallel mission loop) |
| **Product data** | — | Prompts, eval cases, adapters, `bare_eval`, decision JSONL |

**Use-case plugin contract (`astra_plugin`).** Products do not own train entrypoints. They export a module satisfying `UseCasePlugin` (`make_hooks`, `parse_args`; optional `get_runner` for CE distill/rft/star). Missions bind the product at **create** time: `MissionCreate.use_case`, an explicit marker in the goal (`use_case: …` / `--use-case …`), or the Lead Agent reading the product named in the goal — not recipe HP. Create/goal binding **always wins** over a Lead Agent value; `learner_id`-shaped strings (`mlx_dpo`, …) are rejected as `use_case` (they are workers, not product modules). Train CLIs load that importable module via `--use-case`; `ASTRA_USE_CASE` is only a bare-CLI fallback when the flag is omitted (no product aliases). If the top-level package lacks hooks, `load_use_case` tries `{name}.usecase`. Before `run_*`, CLIs call `validate_hooks(command, hooks)` against the attribute set each loop requires (`GRPO_HOOK_ATTRS`, `DpoHooks`, …). Legacy `astrarl.plugins.RewardPlugin` / `astratune.plugins.DatasetPlugin` are step-loop / data-only helpers — not the CLI contract.

**Recipe packs.** Astra `recipes/` holds **platform** YAML only (Snake/Tetris/MinAtar/2048/Pac-Man/AgentGym/hstu/`mlx_lora`/…). Product training knobs (e.g. Ensemble `ensemble_*.yaml`) live in the product repo (`ensemble/astra_recipes/`) and are discovered by the Orchestrator via `ASTRA_PRODUCT_RECIPES=use_case:/abs/path` (comma-separated for multiple products). Search order: platform dir, then configured packs (`backend/services/recipe_paths.py`). List/detail APIs annotate each recipe with `source` (`platform` | `product`) and `product` (the config use_case key when from a pack). `GET /recipes/packs` returns the platform path plus configured product keys for the Recipes UI source checkboxes (dynamic — not a fixed brand list). When `plan.recipe` is omitted but `use_case` is bound, the default stem is `{use_case}_{task}_v1` (`default_recipe_for_task`); `distill` has no silent default. Generic LM `task_type` keys are **not** mapped to Ensemble filenames in `_ENV_RECIPE`.

**Packaging (Mini):** `pip install -e` an Astra tree into Mini `finetune-env` covers `astragym` + `astrarl` + `astratune` + `astra_plugin` + `astradata` + `astraeval`. Typical layout is `~/astra-pkg` (rsync or clone of the Astra repo) — **not** a full Orchestrator HUD/DB checkout. A **workspace** directory (historically `finetune_dir`, e.g. `~/finetune`) is cwd for adapters/data — **not** the train-script home. Products may ship branded plugin packages under `<workspace>/.astra_plugins/` (prepended by `ensure_workspace_on_path`) so `--use-case <brand>` imports cleanly. The product plugin module must be importable. Preflight verifies `import astrarl.cli, astratune.cli, astra_plugin` (and the configured use-case module when set) plus local `astradata` / `astraeval`; mission wrappers may also `sys.path.insert` `astra_pkg_root` before the facade import. Recipes may declare `gym_id` + `learner_id` + `dataset_id`/`dataset_rev` + `eval_suite`/`suite_rev`; when omitted gym/learner are inferred from `task_type` / `env_id` / algorithm / `reward_mode`. `task_type` remains the sandbox and preflight worker key. AstraGym is required (or inferred) for AstraRL; AstraTune workers use recipe `task_type` / `learner_id` only. Create rejects targets above the **suite** ceiling when `eval_suite` is set (else recipe `metric_ceiling`). Recipes one-click dispatch defaults empty `target_metric` from recipe `metric_ceiling` / `target_metric`.

| Gym-coupled `task_type` | AstraGym `gym_id` | AstraRL `learner_id` |
|---|---|---|
| `rl` | `gymnasium:<env_id>`, `agent_gym_vector` | `sb3_{ppo,dqn,a2c,sac,td3}` |
| `grpo` | `routing_prompt` | `mlx_grpo_routing` |
| `agrpo` | `agent_gym_text`, `e2e_stubs` | `mlx_grpo_traj` |
| `opd` | `routing_prompt` | `mlx_opd` |

| Offline `task_type` | AstraTune `learner_id` |
|---|---|
| `sft` / `mlx_lora` | `mlx_sft` |
| `dpo` | `mlx_dpo` |
| `distill` | `mlx_distill` |
| `rft` | `mlx_rft` |
| `star` | `mlx_star` |
| `rlcd` | `mlx_rlcd` |

**AstraGym ids:** `gymnasium:<env_id>`, `agent_gym_vector`, `agent_gym_text`, `e2e_stubs`, `routing_prompt`.  
**AstraRL ids:** `sb3_{ppo,dqn,a2c,sac,td3}`, `mlx_grpo_routing`, `mlx_grpo_traj`, `mlx_opd`.  
**AstraTune ids:** `mlx_sft`, `mlx_dpo`, `mlx_distill`, `mlx_rft`, `mlx_star`, `mlx_rlcd`.

**MLX facades.** Gym-coupled missions use `astrarl.make(..., use_execv=True).learn()`; offline LM missions use `astratune.make(..., use_execv=True).learn()`. Both `os.execv` into `python -m astrarl.cli <cmd>` / `python -m astratune.cli <cmd>` (same pid). CLIs use `astra_plugin.load_use_case` + `validate_hooks` then platform `run_*`. Healer must not rewrite these facades into SB3/gym scripts on `ModuleNotFoundError`.

**Platform GRPO runtime** (Astra `astrarl/`): CLI + train loops (`train_grpo_routing`, `train_grpo_traj`, `train_opd`), rollout backends, DAPO-lite, `grpo_math` / `opd_math`, `mlx_grpo_step`. **Platform AstraTune** (`astratune/`): CLI + train loops (`train_dpo`, `train_sft`, `train_rlcd`, shared `ce_loop`), offline losses, and **`lora_io` shared with AstraRL MLX trainers**. Products own rewards/cases/prompts/oracles behind their `UseCasePlugin` module — not train entrypoints.
**Rollout / DAPO-lite flags** (AstraRL only): `--rollout-backend mlx|openai`, `--rollout-base-url`, `--clip-epsilon-low` / `--clip-epsilon-high`, `--dynamic-sample-retries` (default 0 = off).

| Package? | Scope |
|---|---|
| **AstraRL In** | Interactive / gym-coupled: `sb3_*`, `mlx_grpo_*`, `mlx_opd` |
| **AstraTune In** | Offline LM: `dpo`, `sft`, `mlx_lora`, `distill`, `rft`, `star`, `rlcd` |
| **Neither** | Different stacks: `prompt`, `embed`, `rerank`, `hstu`, `ml` |

### 2.6. Specialist trainers

A task is the objective. An algorithm is the update rule under that task. `task_type` is the worker key for the algorithm. LoRA is a mechanism, not a task. Only Environment RL switches algorithms during a mission. `prompt` changes no weights.

| Task | Algorithms |
|---|---|
| Environment RL | PPO, DQN, A2C, SAC, TD3 (`rl`) |
| Reinforcement fine-tuning | GRPO (`grpo`), DPO (`dpo`) |
| Agentic RL | Agent GRPO (`agrpo`) |
| Supervised fine-tuning | SFT (`sft`, `mlx_lora`), distillation (`distill`), rejection sampling (`rft`), STaR (`star`) |
| On-policy distillation | OPD (`opd`) |
| Calibrated decision | RLCD (`rlcd`) |
| Prompt edit | rule append (`prompt`) |
| Classical ML | random forest (`ml`) |
| Retrieval | embedding (`embed`), reranker (`rerank`) |
| Generative recommendation | HSTU (`hstu`) |

**Online vs offline (data collection).** *Online* means training uses data the current policy collects during the run (act → reward → update → act again). *Offline* means training uses a fixed logged dataset only — no new env interaction or fresh rollouts in the optimizer loop. Hybrids exist (collect a batch online, then train offline on it). Package ownership is independent of that label (§ 2.5): gym-coupled → AstraRL; offline LM → AstraTune.

| Mode | In ASTRA | Package | Notes |
|---|---|---|---|
| **Online** | Env: PPO, A2C; LM: GRPO, Agent GRPO (`agrpo`); OPD (student rollouts) | AstraRL | On-policy samples from the live policy each step |
| **Offline** | Env: DQN/SAC/TD3 (replay) | AstraRL | Replay buffer, but still `astragym` × `sb3_*` |
| **Offline** | LM: DPO, SFT, distill, RFT survivors, STaR JSONL, RLCD `--cases` | AstraTune | Fixed preference pairs / labeled JSONL; no AstraGym |
| **Mixed** | DPO can *harvest* pairs online then optimize offline; RFT samples then SFT | AstraTune | Collection ≠ the loss; workers stay dataset-family |

Language-model PPO is not a reinforcement fine-tuning algorithm. PPO runs only as environment RL (including the AgentGym MLP policy). Rejection sampling (`rft`) keeps oracle-correct student completions and runs SFT on them. GRPO is **online** RL: scores on-policy **single-turn** rollouts with the routing oracle and a group-relative advantage. Agent GRPO (`agrpo`) is **online multi-turn** tool RL: K trajectories in AgentGym text (or e2e stubs), trajectory-level reward, GRPO over assistant turns — not routing `score_completion`. DPO is **offline** preference learning via AstraTune (`mlx_dpo`): the same oracle labels chosen/rejected pairs, then a preference loss trains on that set (no policy-gradient step on live rollouts). RLCD (AstraTune `mlx_rlcd`) trains a calibrated distribution over declared option strings after `Answer:` with a Brier (or log) score; Choice-facing runs use fixed `--cases` JSONL (offline data) with empty-rationale stage 1 to match live readout. Live Choice endpoints that post an empty rationale should use recipes with `empty_rationale: true`.

**Decision foundation approach** (Astra recipes dispatch only): generic `decision_prompt` JSONL → foundation LoRA → product specialize/focus → live readout. Product recipes may stage foundation → continue → specialize → focus, or one-shot from-scratch / routing-only RLCD. Diagram + promote rules live in the product’s decision data docs.

```
  Public_Noul_Choice_Score ──┐
                             v
  Product_exporters ──> decision_cases_jsonl
                             |
                             v
                       Foundation_LoRA
                             |
                             v
                       Specialize_LoRA
                             |
                             v
                      live_rlcd_readout
```

The remote wrapper passes `--cases` when the recipe sets `cases`, omits `--routing-only` when `routing_only: false`, and passes `--eval-max-cases` when `eval_max_cases` > 0. Dispatch Mini RLCD **only** through Astra missions (`POST /recipes/{name}/dispatch`) — never bare product train scripts outside a mission. Empty-rationale pivots on `temp`/`num_generations` are no-ops. HSTU (`hstu`) trains a from-scratch PyTorch Hierarchical Sequential Transduction Unit on MovieLens-1M next-item prediction (same loop shape as a small LSTM recommender); local like `embed`/`rerank`, not Mini MLX, and scores Hit Rate@10.
**Environments.** Arcade envs are 10×10 NumPy Gym wrappers. Seaquest, Asterix, and Grid Pac-Man follow their source rules closely enough that play and checkpoints stay honest (oxygen/surface, spawn/ramp, leftover ghosts chase, mouth faces movement). Env classes load lazily so importing the package does not load every game.

**Code generation.** The coder writes `train.py`. Mechanical post-patches exist because a 7B coder will omit imports, `register()`, or env construction, and a healer that treats every `NameError` as a missing import can swap in the wrong env. SFT falls back to a canonical runner when the AST is bad. Gym-coupled SB3 scripts use `astragym.make(gym_id)` + `astrarl.make(learner_id, …)` (no bare `gym.make`); legacy `gym.make` is rewritten to AstraGym.

**Remote fine-tune.** Mini holds a product **workspace** (e.g. `~/finetune`) plus an editable Astra tree (`~/astra-pkg` by default) and runs trainers under `~/finetune-env`. Canned wrappers bake a **path triad** (absolute paths at codegen — same style as each other):

```
  MacBook (Orchestrator)                    Mini (training host)
  +----------------------+                  +----------------------------------+
  | recipes / .env       |   SSH train.py   | ~/finetune-env/bin/python        |
  |  finetune_dir        | ---------------->|   (ASTRA_SANDBOX_PYTHON)         |
  |  python_bin          |                  |                                  |
  |  astra_pkg_root /    |                  | 1. chdir  finetune_dir           |
  |  ASTRA_PKG_ROOT      |                  | 2. path+= finetune_dir           |
  +----------------------+                  | 3. path+= astra_pkg_root         |
                                            | 4. astratune|astrarl.make(…)     |
                                            | 5. os.execv → -m astrarl/astratune.cli |
                                            |                                  |
                                            | ~/finetune     workspace+plugin  |
                                            | ~/astra-pkg    pip -e packages   |
                                            | ~/finetune-env MLX + site-pkgs   |
                                            +----------------------------------+
```

| Knob | Typical Mini value | Role |
|---|---|---|
| `finetune_dir` | `/Users/kewang/finetune` | Workspace cwd + product data; parent often on `PYTHONPATH` for the plugin |
| `python_bin` | `/Users/kewang/finetune-env/bin/python` | MLX + pip-installed platform packages (`os.execv` target) |
| `astra_pkg_root` | `/Users/kewang/astra-pkg` | `sys.path.insert` before `import astrarl` / `astratune` |
| `use_case` | importable module | Product `UseCasePlugin` — bound at **mission create** (`MissionCreate.use_case`, goal marker `use_case: …` / `--use-case …`, or Lead Agent from the product named in the goal). Not recipe HP. |
| `ASTRA_USE_CASE` | env (optional) | Last-resort bare-CLI fallback when `--use-case` was omitted; **not** the mission product binding. |

Resolution for `astra_pkg_root`: recipe HP `astra_pkg_root` → env `ASTRA_PKG_ROOT` (`settings.pkg_root`) → `/Users/kewang/astra-pkg`. Facade call is `astratune.make` / `astrarl.make(..., use_execv=True).learn()` — the wrapper process image is replaced by `python -m astrarl.cli` / `astratune.cli` (same pid), which loads the product plugin via plan-baked `--use-case` (or `ASTRA_USE_CASE` only if unset). Batch size is clamped to the split. The loop tails the remote log for HUD metrics. Loss-like metrics are lower-is-better.

**Chaining.** SFT → DPO (and similar) is two missions. The second warm-starts the first’s adapter; trainers read LoRA shape from the adapter config rather than trusting the planner.

### 2.7. Sandbox

Where training runs.

- **Apple Silicon** — no Metal passthrough in Docker. GPU work is a restricted host subprocess (or SSH to the Mini). Docker is for CPU/isolation only.
- **CUDA** — container with GPU toolkit.
- **Isolation** — memory/compute caps; writes limited to mission data and the registry.

### 2.8. Evaluator and introspection

**AstraEval** owns authoritative mission gates when `suite_id@rev` is stamped: `suite.run(ckpt) → {primary, case_ids}`, suite ceilings at create, FailSlice emit into AstraData, and tournament seeds (`2000 + ep`). Play on `/models/{id}` and the tournament both load that checkpoint's `train_config` env so the leaderboard matches the player. Arcade **benchmark / stress** helpers remain for introspection outside suite-gated missions. **Saliency** and a **policy auditor** (action distribution, entropy) explain the same play page.

### 2.9. Resilience

- **Safety critic** — GAN-style review of the plan before execute.
- **Mission manifest** — structured source of truth for the next iteration, not a growing chat log.
- **Validation contract** — primary metric plus health signals (entropy, loss stability).

---

## 3. Starting a mission (goal ↔ recipe ↔ product ↔ train)

Both entry points share one pipeline: **`POST /missions`** (Command Center goal bar) and **`POST /recipes/{name}/dispatch`** (Recipes one-click → same `MissionCreate`). `prepare_mission_params` builds the stored mission fields; the loop then plans, codegens, and trains.

### 3.1. What each piece is

| Piece | Where it lives | Role at mission start |
|---|---|---|
| **Goal (mission prompt)** | `Mission.goal` (from create body or canonicalized from recipe description + target) | Human intent: env/product, algorithm, target. Lead Agent reads this. Explicit `use_case: <module>` / `--use-case <module>` markers bind the product without a separate field. |
| **Recipe** | Disk YAML / DB record | Known-good **training knobs** for a worker: `task_type`, adapter paths, LoRA, `finetune_dir` / `python_bin` / `astra_pkg_root`, epochs, ceilings. Optional `dataset_id`/`dataset_rev`, `eval_suite`/`suite_rev`, multi-`stages`. Does **not** own the product plugin name. Astra `recipes/` is **platform-only** (Snake/Tetris/MinAtar/…); product packs (e.g. Ensemble `astra_recipes/`) are discovered via `ASTRA_PRODUCT_RECIPES=use_case:/path` after the platform dir. |
| **`task_type`** | Create body, recipe, or inferred from goal keywords | Worker key (`rl`, `dpo`, `grpo`, …) → sandbox / preflight / which facade CLI. |
| **`use_case` (product)** | Bound at **create** → `plan.hyperparameters.use_case` → baked `--use-case` | Importable `UseCasePlugin` module (e.g. Ensemble → `ensemble`). See resolution order below. |
| **`dataset_id@rev`** | Recipe → plan stamp (AstraData) | Versioned fuel; create/preflight bind paths (e.g. HSTU ML-1M). |
| **`suite_id@rev`** | Recipe `eval_suite`/`suite_rev` → plan stamp (AstraEval) | Authoritative score + ceiling; FailSlice → optional `focus_cases`. |
| **`current_plan`** | Seeded at create when a recipe is present; refreshed by Lead Agent / stages | What the loop executes: recipe name, algorithm, hyperparameters, stages, Data/Eval axes. |
| **Target metric** | Create override → **suite ceiling** when suite stamped, else recipe `target_metric` / `metric_ceiling` → parsed from goal | Completion gate for the HUD / evaluator. |

```
  Command Center goal bar          Recipes "Dispatch"
           |                                |
           |  goal [+ task_type]            |  recipe [+ optional goal / use_case]
           v                                v
                    MissionCreate
                         |
                         v
              prepare_mission_params
                 |  resolve recipe (if any)
                 |  format_canonical_goal
                 |  seed current_plan (recipe / stages)
                 |  bind use_case → plan.hyperparameters
                         |
                         v
                    Mission row (DB)
                         |
                         v
              LoopStateMachine (PLANNING)
                 |  stages / saved plan → reuse
                 |  else Lead Agent.plan(goal)  (+ use_case from goal/product name)
                 |  _ensure_plan_use_case (create/goal wins; strip mlx_* hallucinations)
                 |  Critic review
                         |
                         v
              Codegen + sandbox
                 |  resolve recipe YAML via platform + ASTRA_PRODUCT_RECIPES
                 |  _resolve_hyperparams: recipe HP + plan pivot safelist
                 |                      + plan use_case (wins; not learner_id)
                 |  facade → python -m astrarl|astratune.cli --use-case …
                         v
                    train host (local / Mini)
```

### 3.2. Two ways to start

**A. Goal-first (Command Center)**  
User types a prompt (and optional algorithm dropdown). Backend may infer `task_type` from keywords. If no recipe is attached, the Lead Agent chooses algorithm / `task_type` and may warm-start knobs from a close recipe-library match. When recipe is still omitted but `use_case` is bound, the default recipe stem is `{use_case}_{task}_v1` (e.g. `ensemble` + `dpo` → `ensemble_dpo_v1`); `distill` has no silent default (multiple product variants). Stem resolution walks `settings.recipes_path` then `ASTRA_PRODUCT_RECIPES` packs. For product fine-tunes, name the product in the prompt or write `use_case: <module>` so the plugin is bound before codegen.

**B. Recipe-first (one-click dispatch)**  
`dispatch` builds `MissionCreate(recipe=…, goal?=…, use_case?=…)`. Goal text is canonicalized from the recipe description + target when omitted. `current_plan` is seeded with the recipe name (and stage-0 HPs for multi-stage). Finetune recipes still need a product: pass `use_case` on the dispatch body, put a marker in `goal`, or let the Lead Agent set it when the goal names the product. Recipe YAML must not hardcode `use_case` as a training HP.

### 3.3. Product (`use_case`) resolution

Order used when baking the train wrapper:

1. **`MissionCreate.use_case`** (API / Recipes prompt / dispatch JSON) — **authoritative** when set
2. **Explicit marker in the goal** — `use_case: ensemble` or `--use-case ensemble`
3. **Lead Agent** — may set `hyperparameters.use_case` from the product named in the goal (e.g. “Ensemble routing” → `ensemble`); create/goal binding overwrites this before codegen; never accept `learner_id` values (`mlx_dpo`, …) as `use_case`
4. **`ASTRA_USE_CASE` env** — last-resort bare-CLI fallback only when `--use-case` was omitted; **not** the mission product binding

Recipe hyperparameters supply paths and LoRA settings; they are not the place to stamp the product module. Product recipe **files** live in the product pack (`ASTRA_PRODUCT_RECIPES`), not in Astra `recipes/`.

### 3.4. After create (same for A and B)

1. Critic approves or revises the plan.
2. Codegen writes the sandbox script (canned facade for remote LM workers).
3. Autonomy gates may pause for execute-code approval.
4. Sandbox trains; telemetry streams to the HUD.
5. When `suite_id@rev` is stamped, AstraEval suite-gates the checkpoint (primary metric + FailSlice into AstraData); otherwise the loop scores against the goal / recipe metric.
6. Loop pivots, completes, or stalls. Ensemble follow-ons may inject `focus_cases` from the latest FailSlice (same suite). Successful **RL** (and similar) runs may crystallize a recipe into the platform `recipes/` dir, or into the product pack when `plan.hyperparameters.use_case` maps to `ASTRA_PRODUCT_RECIPES`; fine-tune-remote types do not auto-crystallize.
7. Registry and model page pick up the checkpoint for play and tournaments (same env table; seeds via `astraeval.tournament_seed`).

---

## 4. Security and autonomy

One gate is live: **execute code**. A static pass auto-approves localhost telemetry and blocks obvious danger; ambiguous scripts go to an LLM classifier, then a human if needed. Resource and deploy approval gates are schema-only today (not enforced in the loop).

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
