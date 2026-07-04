# Astra Improvement Ideas

Proposed enhancements, architectural upgrades, and future directions for the astra system.

---

## Ray Multi-Node Training Cluster

### What is Ray

Ray is an open-source distributed computing framework built for Python-first ML workloads. It turns a group of machines into a single compute cluster with a shared task queue, fault tolerance, and a unified resource scheduler — without requiring Kubernetes or cloud infrastructure.

Key components relevant to astra:

- **Ray Core** — the foundation. Lets you annotate any Python function with `@ray.remote` and Ray will schedule it on any available node in the cluster. Results are stored in a distributed object store and retrieved asynchronously. This is what astra would use to dispatch training sandboxes.
- **Ray Dashboard** — a built-in web UI (port 8265) showing live cluster state: which nodes are connected, CPU/memory usage per node, running tasks, and logs. No extra setup needed.
- **Ray Tune** — distributed hyperparameter search. Runs many trials in parallel across the cluster. Relevant to astra's pivot engine — instead of trying one HP configuration per iteration, Tune could run 4–8 trials simultaneously across both machines and report the best result back.
- **Ray Serve** — model serving framework for production APIs. Not relevant to astra's current use case (local training orchestration), but useful if astra's trained models ever need to be deployed as inference endpoints.

Ray is designed to run on commodity hardware over a LAN — no special networking or GPUs required. The head node manages the scheduler and dashboard; worker nodes just need Python, Ray, and network access to the head.

### Motivation

Astra currently runs one mission at a time — the training sandbox is a local subprocess on the MacBook Air, or dispatched to a single statically-configured remote host via `SSHSandbox`. A second machine can sit idle while another mission runs because there's no concurrent dispatch: `SandboxManager` runs one sandbox at a time, and there's no scheduling logic to pick a least-loaded node (`sandbox_host` is a static config value). Ray would enable parallel RL mission dispatch across both machines without changing astra's planning/eval/manifest loop.

Before building this, weigh it against the lighter alternative: extending `SSHSandbox` to round-robin across N configured hosts, which may get most of the concurrency benefit without a new dependency.

Note: this is scoped to **RL mission concurrency only**. Fine-tune task types (`dpo`, `grpo`) are already pinned to the Mac Mini via a task-type check in `SandboxManager.launch()` that forces the existing `SSHSandbox` backend — that's implemented (see `docs/IMPLEMENT.md` Phase 25) and doesn't need Ray. Ray's value-add here is purely "let a second machine pick up RL missions when the first is busy," not remote dispatch itself (which `SSHSandbox` already does).

### Environment prerequisites (checked 2026-07-03 — not yet satisfied)

Before implementing, note the current state of both machines:

- **`ray` is not installed** in astra's venv, nor in the Mac Mini's `~/finetune-env`. Needs installing in both places before any of this works.
- **Astra's codebase is not checked out on the Mac Mini.** Free-scheduled RL dispatch via Ray means the remote worker needs to actually run `SubprocessSandbox`-equivalent training scripts — either astra needs to be present there, or the `ray.remote` task needs to be self-contained enough not to need it (plausible for simple `subprocess.run(["python", script])` dispatch, since Ray ships the function closure to the worker, but the training script itself may `sys.path.insert` against `settings.data_path`/project root in ways that assume an astra checkout exists locally).
- Mission data (`data/missions/`) is not shared/mounted between machines — needed for RL missions' checkpoint/telemetry file access if they run on the Mac Mini (see Implementation Plan step 5 below).

### Architecture

```
MacBook Air (head node)          Mac Mini (worker node)
┌─────────────────────┐          ┌─────────────────────┐
│  ray start --head   │◄────────►│  ray start --address │
│  astra backend      │  LAN     │  training sandbox    │
│  lead + code model  │          │  Mission B           │
│  Mission A          │          │                      │
└─────────────────────┘          └─────────────────────┘
```

- **MacBook Air** — head node, runs astra backend, MLX inference (lead + code models), and local training sandboxes
- **Mac Mini** — worker node, runs training sandboxes dispatched from the head node; no LLM inference needed
- All telemetry from both nodes posts back to astra's backend on the MacBook (port 8200) — already works today via `settings.telemetry_host` (defaults `macbook.local`), no code change needed
- Ray dashboard available at `http://macbook.local:8265`

### How It Works

**Cluster startup:**
```bash
# MacBook (head)
ray start --head --port=6379

# Mac Mini (worker)
ray start --address=macbook.local:6379
```

**Ray dispatches training jobs as remote tasks:**
```python
@ray.remote(num_cpus=4)
def run_training(mission_id, script_path):
    subprocess.run(["python", script_path])
```

Ray's scheduler routes each task to whichever node has free resources. If the MacBook is busy with Mission A, Mission B automatically lands on the Mac Mini.

### What Changes in Astra

| Component | Current | With Ray |
|---|---|---|
| `SubprocessSandbox` / `SSHSandbox` | local subprocess, or one static remote host | `RaySandbox` submits task to cluster scheduler |
| LLM inference | MacBook only | MacBook only (MLX is single-machine) |
| Telemetry | already POSTs to `settings.telemetry_host` (defaults `macbook.local`) — no change needed | same |
| RL concurrency | 1 mission at a time | 2+ missions simultaneously |

### Implementation Plan

1. Install `ray` in astra's venv and the Mac Mini's environment (see Environment prerequisites above — not done yet)
2. Add `backend/sandbox/ray_sandbox.py` — implements `SandboxBase`, submits training script as a `ray.remote` task
3. Add `ASTRA_SANDBOX_BACKEND=ray` env var (default: `subprocess`, joining existing `subprocess`/`ssh` options)
4. `SandboxManager` selects backend based on config (same pattern `_detect_backend()` already uses for `ssh`) — note `_detect_backend()` was deliberately changed to NOT check `settings.sandbox_host` (see Phase 25) since that setting is now scoped to fine-tune task-type pinning only; a `ray` default would need its own opt-in signal, not reuse `sandbox_host`
5. Mission data (`data/missions/`) must be accessible on both nodes — either via NFS mount or rsync before sandbox launch. `SSHSandbox` currently gets away with rsync-back-on-exit instead of a live mount — checkpoints under `checkpoints/` are only synced post-hoc when the sandbox exits, never live during a run. Same limitation would apply to `RaySandbox` unless NFS is used.
6. Confirm `SandboxBase` contract (launch, cancel, log streaming, exit status) can be satisfied by a Ray remote task — cancel/kill semantics differ between a local subprocess (`SIGTERM`/`SIGKILL`, matching `SubprocessSandbox` and now `SSHSandbox` — see Phase 24) and a Ray task (`ray.cancel`), so `RaySandbox.cancel()` needs its own implementation, not a passthrough.
7. Telemetry already POSTs to `settings.telemetry_host` (defaults `macbook.local`) — no code change needed here.
8. Verify `LoopStateMachine`/`PivotEngine` plateau and best-score tracking (currently reads/writes local files like `best_score.txt` under `data/missions/<id>/`) still works correctly when the sandbox executing the write is on the Mac Mini and the state machine reading it is on the MacBook — this requires the NFS mount from step 5 to be consistent, and should be tested explicitly before relying on pivot detection across nodes

### Practical Result

- Two RL missions train in parallel — one per machine
- If MacBook is busy, new missions queue to Mac Mini automatically
- No changes to the planning/eval/manifest loop — only the sandbox layer changes
- Ray handles fault tolerance: if Mac Mini goes offline, jobs requeue to MacBook
