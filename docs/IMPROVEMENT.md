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

Astra currently runs one mission at a time — the training sandbox is a local subprocess on the MacBook Air. A second machine (Mac Mini) sits idle while training runs. Ray enables parallel mission dispatch across both machines without changing astra's orchestration logic.

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
- All telemetry from both nodes posts back to astra's backend on the MacBook (port 8200)
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
| `SubprocessSandbox` | local subprocess | `RaySandbox` submits task to cluster |
| LLM inference | MacBook only | MacBook only (MLX is single-machine) |
| Telemetry | localhost | both nodes POST to MacBook:8200 |
| Concurrency | 1 mission | 2+ missions simultaneously |

### Implementation Plan

1. Add `backend/sandbox/ray_sandbox.py` — implements `SandboxBase`, submits training script as a `ray.remote` task
2. Add `ASTRA_SANDBOX_BACKEND=ray` env var (default: `subprocess`)
3. `SandboxManager` selects backend based on config
4. Mac Mini needs: Python, astra dependencies, Ray worker, access to shared or synced mission data path
5. Mission data (`data/missions/`) must be accessible on both nodes — either via NFS mount or rsync before sandbox launch

### Practical Result

- Two missions train in parallel — one per machine
- If MacBook is busy, new missions queue to Mac Mini automatically
- No changes to the planning/eval/manifest loop — only the sandbox layer changes
- Ray handles fault tolerance: if Mac Mini goes offline, jobs requeue to MacBook
