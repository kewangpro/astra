# Astra Improvement Ideas

Proposed enhancements, architectural upgrades, and future directions for the astra system.

Living doc for future work — not a ship log ([IMPLEMENT.md](IMPLEMENT.md) is
the phase history). Capability matrix for learner runtimes: Cursor canvas
`canvases/astrarl-vs-peers.canvas.tsx` (revisited 2026-11-14). Platform
contracts: [DESIGN.md](DESIGN.md) §2.4–§2.5, §3.

---

## AstraRL vs verl / NeMo-RL / peers

### Category mismatch (keep this framing)

AstraRL is a **gym-coupled learner runtime** inside ASTRA’s mission loop (Apple
MLX + SB3), not a cluster LLM-RL engine. **verl**, **NeMo-RL**, and **OpenRLHF**
are full distributed post-training systems (Ray / HybridFlow, vLLM/SGLang,
Megatron/FSDP/DeepSpeed). **TRL** is a trainer library. Comparing feature counts
without that split misleads; improvements below target the learner runtime and
its seam into the Orchestrator — not a Ray rewrite of `astrarl/`.

Offline LM (DPO/SFT/distill/RFT/STaR/RLCD) belongs in **AstraTune**, not AstraRL
([DESIGN.md](DESIGN.md) §2.5). Products bind via **`astra_plugin.UseCasePlugin`**
at mission create — not recipe HP.

### Closed since the first peer review (Epoch 9–11)

| Former gap | Resolution |
|---|---|
| Learner facade honesty (`mlx_*` was a dict/CLI shim) | `MLXLearnerFacade.learn()` → `os.execv` into `python -m astrarl.cli` |
| No generation backend abstraction | `astrarl.rollout` — `mlx` \| `openai` (+ recipe flags) |
| Thin algo surface (no DAPO) | DAPO-lite: `clip_epsilon_low`/`high`, `dynamic_sample_retries` on routing GRPO |
| Training logic lived in Ensemble CLIs | Platform `train_grpo_routing` / `train_grpo_traj` / `train_opd` in `astrarl/` |
| DPO/SFT mixed into the RL story | Sibling **AstraTune**; DESIGN boundary enforced |
| Product hardcoding in train entrypoints | `astra_plugin` + mission-create `use_case` |
| Ensemble YAML / defaults baked into Astra | Platform `recipes/` only; product packs via `ASTRA_PRODUCT_RECIPES`; `{uc}_{task}_v1` defaults |
| Missing Data / Eval flywheel layers | **Epoch 11:** `astradata/` + `astraeval/`; suite gate + FailSlice → `focus_cases` ([DESIGN.md](DESIGN.md) §2.4) |

### Remaining gaps (re-ranked)

**P0 — async & latency.** Peers overlap generation with training (NeMo-RL’s async
subsystem; verl experimental; OpenRLHF second path). Agent GRPO on Mini is still
sync collect-then-train and latency-bound. **Next win:** overlap next-group
generation with the update, or packed multi-trajectory batches — single-machine
first, no Ray required.

**P0 — agentic / multi-turn fidelity.** Peers race on TITO, R3, prefix reuse, and
server-owned tokenization. `agrpo` scores trajectories but lacks a peer-grade
episode schema and token-accurate credit for long tool traces. **Next win:**
document and export AgentGym / `routing_prompt` trajectories
(messages / tool calls / turn rewards) before wrapping an external verl/NeMo
worker.

**P1 — rollout depth.** `mlx|openai` is enough to A/B a remote vLLM URL; peers
colocate generation (weight sync, prefix cache, continuous batching). **Gap:**
adapter/weight hot-reload into the OpenAI-compat server; optional local
SGLang/vLLM only when on CUDA — not on Mini MLX.

**P1 — algorithm surface.** DAPO-lite is in; GSPO / RLOO / ReMax / CISPO / GDPO
multi-reward are not. Add only variants that move routing or `agrpo` metrics —
do not mirror verl’s full catalog.

**P2 — scale path.** No FSDP / Megatron / multi-GPU story (intentional for
Apple-first). If cloud NVIDIA is adopted: wrap **verl** or **NeMo-RL** as a
remote `learner_id` (e.g. `cuda_grpo_verl`) with checkpoint sync — same SSH
pattern as Mini — do **not** reimplement HybridFlow inside `astrarl/`.

**P2 — experiment parity.** Mission HUD is strong for ops; peers publish
Hydra-like configs + step curves for paper baselines. Emit a recipe dump + GRPO
step-metrics artifact so Mini runs compare cleanly to published DAPO/GRPO
numbers.

### What not to copy

- Don’t turn `astrarl/` into a Ray cluster for LLM RL — that fights the
  Orchestrator and the Apple path. (Ray for **RL mission concurrency** across
  LAN nodes is a separate idea below.)
- Don’t pull offline DPO/SFT back into AstraRL.
- Don’t chase 100B+ Megatron until a `cuda_*` remote learner exists.
- Do keep Gym×RL + `astra_plugin` — clearer than most peers’ env story.

### Suggested roadmap

| Phase | Status | Work | Borrow from |
|---|---|---|---|
| A | Done | Real Learner facade + platform CLI | TRL Trainer simplicity |
| B | Done | Pluggable rollout (MLX \| OpenAI-compat) | verl / NeMo backends |
| C | Done | DAPO-lite on routing GRPO | verl / NeMo DAPO |
| D | **Next** | Trajectory schema + async group overlap for `agrpo` | NeMo-Gym / async GRPO / TITO ideas |
| E | Later | Optional `cuda_grpo_verl` remote worker | verl as engine, ASTRA as orchestrator |

---

## Ray Multi-Node Training Cluster

Scoped to **LAN scheduling of classic env-RL sandboxes** across MacBook + Mini.
This is **not** “adopt verl/NeMo-RL’s Ray stack for LLM GRPO” — that belongs under
[AstraRL vs peers](#astrarl-vs-verl--nemo-rl--peers) phase E (`cuda_grpo_verl`).
Do not conflate the two.

### What is Ray

Ray is an open-source distributed computing framework for Python-first ML
workloads: a shared task queue, object store, and resource scheduler over a LAN
(no Kubernetes required).

Relevant pieces:

- **Ray Core** — `@ray.remote` tasks scheduled onto free nodes. Candidate API for
  a future `RaySandbox` that runs RL `train.py` where capacity exists.
- **Ray Dashboard** — cluster UI (port 8265): nodes, CPU/memory, tasks, logs.
- **Ray Tune** — optional later: multi-trial HP search across nodes (today’s
  `PivotEngine` is sequential, one adjustment per mission iteration).
- **Ray Serve** — out of scope (inference serving).

### Motivation (accurate as of 2026-11)

**What already works without Ray**

| Path | Today |
|---|---|
| Orchestrator | MacBook FastAPI + mission loops (`_running_tasks` is multi-mission; loops can overlap) |
| Classic env RL (`rl`, local `embed`/`rerank`/`hstu`/`ml`, …) | Default backend `_detect_backend()` → `subprocess` on Apple Silicon, else `container`. **Does not** follow `sandbox_host`. |
| Remote LM / gym-coupled LM workers | Forced `SSHSandbox` to `settings.sandbox_host` (Mini) for `_FINETUNE_REMOTE_TASK_TYPES`: `dpo`, `grpo`, `agrpo`, `distill`, `rft`, `opd`, `rlcd`, `prompt`, `sft`, `post-training`, `star`. No local fallback if host unset. |
| Mini layout | Triad: `~/astra-pkg` (editable astragym/astrarl/astratune/astra_plugin/astradata/astraeval), `~/finetune` (workspace + product plugin), `~/finetune-env` (MLX Python). Not a full Orchestrator/HUD/DB checkout. |
| Telemetry | Sandboxes POST to `settings.telemetry_host` (default `macbook.local:8200`). |

**Real gap Ray would address**

There is still **no least-loaded node scheduler for classic RL**. All remote LM
work goes to one static `sandbox_host`. A second LAN machine can sit idle while
the MacBook runs a heavy SB3 mission, because nothing picks “run this `rl`
sandbox over there.” Ray (or a lighter round-robin across N SSH hosts) would
fill that hole **without** changing planning / eval / manifest.

Weigh Ray against: extend `SSHSandbox` to round-robin N configured RL hosts —
most of the concurrency benefit, no new dependency.

### Environment prerequisites (re-checked 2026-11-14 — still not done)

- **`ray` is not installed** in Astra’s MacBook `.venv` or Mini `~/finetune-env`.
- **Mini does not need the Orchestrator repo** for today’s SSH LM path (triad
  above). Free-scheduled **RL** via Ray is different: the worker must run
  SB3/gym scripts that assume `settings.data_path` / mission dirs — either NFS /
  rsync of `data/missions/`, a fuller tree on the worker, or a self-contained
  `ray.remote` payload.
- Mission checkpoints for RL still live under Orchestrator
  `data/missions/<id>/`; SSH LM adapters live under Mini `finetune_dir/adapters/`
  and sync back via `SSHSandbox` (not a live mount).

### Architecture (proposed)

```
MacBook Air (head)                    Mac Mini (worker)
┌──────────────────────────┐          ┌──────────────────────────┐
│ ray start --head         │◄────────►│ ray start --address=…    │
│ Astra Orchestrator       │   LAN    │ RL train.py sandboxes    │
│ lead + code MLX models   │          │ (optional)               │
│ local RL sandboxes       │          │                          │
│ SSH → Mini for LM types  │──────────│ astra-pkg + finetune-env │
└──────────────────────────┘          │ (unchanged LM path)      │
                                      └──────────────────────────┘
```

- **MacBook** — head: Orchestrator, lead/code MLX, local RL, and SSH dispatch of
  LM workers to Mini (existing).
- **Mini** — already the LM train host; under this proposal, also a Ray worker
  for **classic RL** when the MacBook is saturated.
- Ray dashboard: `http://macbook.local:8265` (if head runs there).

### How it would work

```bash
# MacBook (head)
ray start --head --port=6379

# Mac Mini (worker) — only needed for Ray-scheduled RL, not for today's SSH LM
ray start --address=macbook.local:6379
```

```python
@ray.remote(num_cpus=4)
def run_training(mission_id, script_path):
    subprocess.run(["python", script_path])
```

Scheduler places the task on a free node. LM task types keep today’s forced SSH
path; Ray must not steal `dpo`/`grpo`/… away from the Mini triad.

### What would change

| Component | Current (2026-11) | With Ray (proposed) |
|---|---|---|
| Default backend | Darwin arm64 → `subprocess`; else `container`. `_detect_backend()` **ignores** `sandbox_host`. | Add opt-in `ray` backend for **RL only** |
| Remote LM types | Forced `backend="ssh"` + `settings.sandbox_host` | **Unchanged** — still Mini SSH / facades |
| `sandbox_host` | Mini hostname for remote LM only | Still LM-only; do not reuse as Ray default |
| Lead/code MLX | MacBook | MacBook |
| Telemetry | POST `telemetry_host` | Same |
| RL placement | Always local default (or explicit backend) | Schedule onto least-loaded Ray node |

There is **no** `ASTRA_SANDBOX_BACKEND` setting today. Backends are selected in
`SandboxManager.launch()` (`ssh` forced for remote types, else
`backend or default_backend` ∈ {`subprocess`, `container`}). A Ray path needs a
**new** opt-in (env or settings flag), not overload of `sandbox_host`.

### Implementation plan

1. Install `ray` in MacBook `.venv` and, if Mini is an RL worker, `~/finetune-env`
   (or a separate RL venv with gym/SB3 — Mini’s finetune-env is MLX-centric).
2. Add `backend/sandbox/ray_sandbox.py` implementing the same contract as
   `SubprocessSandbox` / `SSHSandbox` (`launch`, `terminate`, log tail, exit).
3. Add an explicit opt-in (e.g. `ASTRA_SANDBOX_BACKEND=ray` or
   `settings.sandbox_backend`) — default stays `subprocess` on Apple Silicon.
4. Wire `SandboxManager`: only offer Ray for non-remote task types; keep
   `_FINETUNE_REMOTE_TASK_TYPES` → SSH. Never treat non-empty `sandbox_host` as
   “default everything to Ray/SSH” (Phase 25 invariant).
5. Share `data/missions/` for RL (NFS or rsync-before-launch). Today’s
   `SSHSandbox` rsyncs logs/checkpoints **on exit**; live pivot files
   (`best_score.txt`, etc.) need a consistent view if the state machine stays
   on the MacBook while the sandbox writes on Mini.
6. Map cancel: local `SIGTERM`→`SIGKILL` vs `ray.cancel` — implement
   `RaySandbox.terminate()` explicitly (see Phase 24 SSH parity).
7. Telemetry already targets `telemetry_host` — no change if workers can reach
   the MacBook.
8. Integration test: plateau / best-score detection with sandbox on Mini and
   loop on MacBook before trusting cross-node pivots.

### Practical result (if shipped)

- Classic RL can use idle LAN capacity; MacBook stays Orchestrator + planner.
- Remote LM missions stay on Mini via existing SSH + `astrarl`/`astratune`
  facades — Ray does not replace that path.
- Planning / eval / manifest loop unchanged; only sandbox placement changes.
- Fault tolerance: Ray can requeue RL tasks if a worker drops (design goal;
  must be verified).
