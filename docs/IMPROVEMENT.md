# Astra Improvement Ideas

Proposed enhancements, architectural upgrades, and future directions for the astra system.

---

## Ray Multi-Node Cluster + Mac-Mini-Pinned Fine-Tuning

### What is Ray

Ray is an open-source distributed computing framework built for Python-first ML workloads. It turns a group of machines into a single compute cluster with a shared task queue, fault tolerance, and a unified resource scheduler — without requiring Kubernetes or cloud infrastructure.

Key components relevant to astra:

- **Ray Core** — the foundation. Lets you annotate any Python function with `@ray.remote` and Ray will schedule it on any available node in the cluster. Results are stored in a distributed object store and retrieved asynchronously. This is what astra would use to dispatch training sandboxes.
- **Ray Dashboard** — a built-in web UI (port 8265) showing live cluster state: which nodes are connected, CPU/memory usage per node, running tasks, and logs. No extra setup needed.
- **Ray Tune** — distributed hyperparameter search. Runs many trials in parallel across the cluster. Relevant to astra's pivot engine — instead of trying one HP configuration per iteration, Tune could run 4–8 trials simultaneously across both machines and report the best result back.
- **Ray Serve** — model serving framework for production APIs. Not relevant to astra's current use case (local training orchestration), but useful if astra's trained models ever need to be deployed as inference endpoints.
- **Custom resources** — Ray lets a worker node advertise arbitrary named resources (e.g. `resources={"mac_mini": 1}` at `ray start`), and a task can require that resource to force scheduling onto a specific node rather than "any free node." This is the mechanism astra needs for fine-tune pinning below — not just free-for-all load balancing.

Ray is designed to run on commodity hardware over a LAN — no special networking or GPUs required. The head node manages the scheduler and dashboard; worker nodes just need Python, Ray, and network access to the head.

### Motivation

This proposal now covers two related but distinct needs, both solved by adding a `RaySandbox`:

1. **Concurrency for RL missions.** Astra currently runs one mission at a time — the training sandbox is a local subprocess on the MacBook Air, or dispatched to a single statically-configured remote host via `SSHSandbox`. A second machine can sit idle while another mission runs because there's no concurrent dispatch: `SandboxManager` runs one sandbox at a time, and there's no scheduling logic to pick a least-loaded node (`sandbox_host` is a static config value). This part is genuinely a free-scheduling problem — either machine could run an RL mission, so Ray's default "any available node" behavior is a good fit. (Before building this, weigh it against the lighter alternative: extending `SSHSandbox` to round-robin across N configured hosts, which may get most of the concurrency benefit without a new dependency.)

2. **Mandatory node pinning for fine-tune task types.** Separately, astra has (or will have) four fine-tuning task types — `mlx_lora`, `sft`, and the planned `dpo`/`grpo` (see below) — that should **always** run on the Mac Mini, never the MacBook, regardless of load. This isn't a load-balancing decision: `dpo`/`grpo` depend on the `ensemble/finetune/` project directory (custom DPO/GRPO training scripts, adapters, eval oracle) which lives only on the Mac Mini, and pinning all fine-tune work there keeps the MacBook free for LLM inference (lead/code models) and RL mission orchestration. If the Mac Mini is unavailable, these missions should **fail immediately, with no fallback** to the MacBook — silently running a `dpo`/`grpo` mission somewhere `ensemble/finetune` doesn't exist isn't a valid degraded mode, it's a broken one.

Both needs are served by the same `RaySandbox` implementation; they differ only in whether a task requests a specific resource tag.

### Architecture

```
MacBook Air (head node)                    Mac Mini (worker node, resources={"mac_mini": 1})
┌─────────────────────────┐                ┌──────────────────────────────────┐
│  ray start --head       │◄──────────────►│  ray start --resources=           │
│  astra backend          │      LAN        │    '{"mac_mini": 1}'             │
│  lead + code model      │                 │  ensemble/finetune/ (scripts,    │
│  RL missions (any node) │                 │    adapters, ~/finetune-env)     │
│                         │                 │  fine-tune missions (pinned)     │
│                         │                 │  RL missions (if MacBook busy)   │
└─────────────────────────┘                └──────────────────────────────────┘
```

- **MacBook Air** — head node, runs astra backend, MLX inference (lead + code models). RL missions launch here by default, or on the Mac Mini if the MacBook is busy (free scheduling).
- **Mac Mini** — worker node, advertises the `mac_mini` custom resource. All `mlx_lora`/`sft`/`dpo`/`grpo` missions are pinned here via `ray.remote(resources={"mac_mini": 1})`; RL missions may also land here under load.
- All telemetry from both nodes posts back to astra's backend on the MacBook (port 8200) — already works today via `settings.telemetry_host` (defaults `macbook.local`), no code change needed.
- Ray dashboard available at `http://macbook.local:8265`.

### How It Works

**Cluster startup:**
```bash
# MacBook (head)
ray start --head --port=6379

# Mac Mini (worker) — advertises the mac_mini resource tag
ray start --address=macbook.local:6379 --resources='{"mac_mini": 1}'
```

**Free-scheduled RL dispatch** (either node):
```python
@ray.remote(num_cpus=4)
def run_training(mission_id, script_path):
    subprocess.run(["python", script_path])
```

**Pinned fine-tune dispatch** (Mac Mini only, hard-fails if unavailable):
```python
@ray.remote(num_cpus=4, resources={"mac_mini": 1})
def run_finetune(mission_id, script_path):
    subprocess.run(["python", script_path], cwd=ENSEMBLE_FINETUNE_DIR)
```

Before submitting a pinned task, `SandboxManager` checks `ray.cluster_resources().get("mac_mini", 0) < 1` and raises a launch failure immediately (not a healing/retry — this is infra unavailability, not a script bug, so it should not go through `ErrorAnalyzer`'s sandbox-error healing path). RL missions have no such check — Ray's scheduler just routes them to whichever node is free.

### What Changes in Astra

| Component | Current | With Ray |
|---|---|---|
| `SubprocessSandbox` / `SSHSandbox` | local subprocess, or one static remote host | `RaySandbox` submits task to cluster scheduler; free-scheduled for RL, pinned to `mac_mini` for fine-tune task types |
| LLM inference | MacBook only | MacBook only (MLX is single-machine) |
| Telemetry | already POSTs to `settings.telemetry_host` (defaults `macbook.local`) — no change needed | same |
| RL concurrency | 1 mission at a time | 2+ missions simultaneously across both nodes |
| Fine-tune dispatch (`mlx_lora`, `sft`, `dpo`, `grpo`) | `mlx_lora`/`sft` run wherever `default_backend` resolves (MacBook today); `dpo`/`grpo` don't exist yet | always pinned to Mac Mini; launch fails immediately if Mac Mini unavailable — no fallback |

### Implementation Plan

**Ray plumbing (needed for both use cases):**

1. Add `backend/sandbox/ray_sandbox.py` — implements `SandboxBase`, submits training script as a `ray.remote` task
2. Add `ASTRA_SANDBOX_BACKEND=ray` env var (default: `subprocess`, joining existing `subprocess`/`ssh` options)
3. `SandboxManager` selects backend based on config (same pattern `_detect_backend()` already uses for `ssh`)
4. Mac Mini needs: Python, astra dependencies, Ray worker started with `--resources='{"mac_mini": 1}'`, access to shared or synced mission data path
5. Mission data (`data/missions/`) accessible on both nodes via **NFS mount** for RL missions (decided over rsync — avoids staleness/race conditions if mission dir changes mid-run; simpler to operate on a 2-node LAN than a rsync-before-launch step). Note `SSHSandbox` currently gets away with rsync-back-on-exit instead of a live mount — checkpoints under `checkpoints/` are only synced post-hoc when the sandbox exits, never live during a run. Same limitation applies to `RaySandbox` unless NFS is used.
6. Confirm `SandboxBase` contract (launch, cancel, log streaming, exit status) can be satisfied by a Ray remote task. `RaySandbox.cancel()` needs its own implementation (`ray.cancel`), not a passthrough of `SubprocessSandbox`/`SSHSandbox`'s terminate sequence.
7. Telemetry already POSTs to `settings.telemetry_host` (defaults `macbook.local`) — no code change needed here.
8. Verify `LoopStateMachine`/`PivotEngine` plateau and best-score tracking still works when an RL sandbox runs on the Mac Mini. Telemetry-based state (`telemetry.jsonl`, read via HTTP POST) is already node-agnostic and fine. The open risk is checkpoint-file state (`best_score.txt`, `checkpoints/best_model.zip`): only correct after sync-back on exit, so mid-run reads from the head node would be stale unless step 5's NFS mount is used instead.

**Fine-tune task types + Mac-Mini pinning:**

9. Add `dpo`/`grpo` task types: `recipes/ensemble_dpo_v1.yaml` / `recipes/ensemble_grpo_v1.yaml` (base model, adapter warm-start, LoRA rank/scale, beta/epochs for DPO, num_generations/clip_epsilon/reward_schema for GRPO — mirroring `ensemble/finetune/dpo_train.py` and `grpo_train.py`'s existing CLI args)
10. Add `_DPO_TEMPLATE` / `_GRPO_TEMPLATE` to `code_generator.py` — thin wrappers that `subprocess.run` the *existing* `ensemble/finetune/dpo_train.py` / `grpo_train.py` scripts rather than having the LLM regenerate DPO/GRPO loss math from a prompt (same reasoning as Phase 23's deterministic `_inject_curriculum`: complex training-loop logic should be reused/deterministic, not LLM-authored per iteration)
11. Define `_FINETUNE_TASK_TYPES = {"sft", "mlx_lora", "dpo", "grpo"}` in `SandboxManager` — any mission with `task_type` in this set is dispatched via `RaySandbox` with `resources={"mac_mini": 1}`, unconditionally, regardless of `ASTRA_SANDBOX_BACKEND`
12. Pre-launch availability check: if `ray.cluster_resources().get("mac_mini", 0) < 1`, fail the mission launch immediately with a clear error ("Mac Mini unavailable — fine-tune missions require the Ray worker node") — do not route through `ErrorAnalyzer`'s script-healing path, and do not fall back to `SubprocessSandbox` on the MacBook
13. Manifest/evaluator changes: `dpo`/`grpo` missions track `pass_rate` against the eval oracle (not `eval_loss`, since DPO/GRPO loss isn't directly comparable to SFT loss) — add to the higher-is-better metric set in `manifest_generator.py`, new benchmark requirement analogous to the existing `eval_loss ≤ 1.5` one
14. Adapter sync-back: after a Mac-Mini fine-tune run completes, rsync just the adapter output directory (`ensemble/finetune/adapters/<mission_id>/`) back to `astra/data/missions/<mission_id>/checkpoints/` — scoped rsync, not a full NFS mount, since fine-tune missions don't need live mid-run checkpoint visibility the way RL pivot detection does
15. Verify runtime parity before enabling this for `sft`: `mlx_lora`/`dpo`/`grpo` all use MLX, which the Mac Mini already runs (via `ensemble/finetune`'s `~/finetune-env`), but `sft` uses `trl.SFTTrainer` (PyTorch/`transformers`/`trl`) — confirm those deps exist on the Mac Mini before assuming this rule applies uniformly across all four task types

### Practical Result

- Two RL missions train in parallel — one per machine — when both nodes are free
- Fine-tune missions (`mlx_lora`, `sft`, `dpo`, `grpo`) always run on the Mac Mini, never silently fall back to the MacBook if it's unavailable — launch fails loudly instead
- No changes to the planning/eval/manifest loop — only the sandbox layer and (for `dpo`/`grpo`) two new task-type templates change
- Ray handles fault tolerance for the free-scheduled RL case: if Mac Mini goes offline, RL jobs requeue to MacBook — this does **not** apply to pinned fine-tune jobs, which fail instead of requeuing elsewhere
