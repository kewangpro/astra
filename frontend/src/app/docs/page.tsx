import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "Docs — ASTRA",
  description: "Tasks and the algorithms that train them",
};

type Algorithm = {
  name: string;
  taskType: string;
  body: string;
};

type Task = {
  id: string;
  task: string;
  summary: string;
  algorithms: Algorithm[];
};

const TASKS: Task[] = [
  {
    id: "environment-rl",
    task: "Environment RL",
    summary:
      "A policy learns from the environment’s own reward. This is the only task that may switch algorithms when a run plateaus. Environments include Snake, Tetris, 2048, the MinAtar suite, Grid Pac-Man, AgentGym, and a standard Gymnasium id.",
    algorithms: [
      {
        name: "PPO",
        taskType: "rl",
        body: "Clipped surrogate policy gradient (Stable-Baselines3). The usual start for discrete control such as Snake.",
      },
      {
        name: "DQN",
        taskType: "rl",
        body: "Off-policy Q-learning with a replay buffer. The recipe default for several MinAtar games, including Seaquest.",
      },
      {
        name: "A2C",
        taskType: "rl",
        body: "Synchronous advantage actor-critic. A lighter on-policy alternative to PPO.",
      },
      {
        name: "SAC",
        taskType: "rl",
        body: "Off-policy actor-critic with an entropy bonus. For continuous actions.",
      },
      {
        name: "TD3",
        taskType: "rl",
        body: "Twin delayed deterministic policy gradient. For continuous actions.",
      },
    ],
  },
  {
    id: "reinforcement-fine-tuning",
    task: "Reinforcement fine-tuning",
    summary:
      "A language-model policy is updated from the routing oracle in eval_cases.yaml. The algorithm stays fixed for the mission. There is no language-model PPO trainer; PPO runs only under Environment RL.",
    algorithms: [
      {
        name: "GRPO",
        taskType: "grpo",
        body: "Sample a group of completions, score each with the oracle (skill match, partial credit, malformed JSON), and apply a group-relative advantage with a PPO-style clip.",
      },
      {
        name: "DPO",
        taskType: "dpo",
        body: "The same oracle labels one chosen completion and one rejected completion. The update is the offline preference loss, so a batch still teaches when every rollout in a GRPO group would have scored the same.",
      },
    ],
  },
  {
    id: "supervised-fine-tuning",
    task: "Supervised fine-tuning",
    summary:
      "The model fits labeled completions. There is no policy-gradient reward, and the algorithm stays fixed.",
    algorithms: [
      {
        name: "SFT",
        taskType: "sft",
        body: "Remote MLX supervised fine-tune on routing demonstrations, with a held-out split.",
      },
      {
        name: "SFT",
        taskType: "mlx_lora",
        body: "The same supervised update on the local Apple Silicon machine, via MLX LoRA.",
      },
      {
        name: "distillation",
        taskType: "distill",
        body: "A stronger teacher writes the completions, then the student is trained with SFT. The ceiling is what the teacher already gets right.",
      },
      {
        name: "rejection sampling",
        taskType: "rft",
        body: "Sample K completions from the student, keep the ones the oracle marks correct, and run SFT on the survivors. No teacher. This is supervised fine-tuning, not reinforcement fine-tuning.",
      },
      {
        name: "STaR",
        taskType: "star",
        body: "Roll out, write a hint-guided rationale for the misses, then LoRA on those traces.",
      },
    ],
  },
  {
    id: "on-policy-distillation",
    task: "On-policy distillation",
    summary:
      "The student samples its own rollouts. A teacher grades those states. The algorithm stays fixed, and the ceiling is still the teacher.",
    algorithms: [
      {
        name: "OPD",
        taskType: "opd",
        body: "Grade each student token with reverse KL against a teacher that scores those same states. Credit is dense on the student’s own trajectory.",
      },
    ],
  },
  {
    id: "calibrated-decision",
    task: "Calibrated decision",
    summary:
      "The decision is a distribution over the closed skill set, not a sampled skill and not a JSON plan. The reported pass rate is readout accuracy. best/ is the lowest Brier checkpoint.",
    algorithms: [
      {
        name: "RLCD",
        taskType: "rlcd",
        body: "Sample rationales, read the skill distribution after Answer:, and score it with Brier. Stage 1 trains that readout. Stage 2 reinforces the rationale with the same score and a KL pull back to the stage-1 weights.",
      },
    ],
  },
  {
    id: "prompt-edit",
    task: "Prompt edit",
    summary: "No weights change. The production conductor prompt is never edited.",
    algorithms: [
      {
        name: "rule append",
        taskType: "prompt",
        body: "Propose extra routing rules on a copy of the conductor prompt, then score that copy.",
      },
    ],
  },
  {
    id: "classical-ml",
    task: "Classical ML",
    summary: "A tabular model, not a language-model LoRA.",
    algorithms: [
      {
        name: "random forest",
        taskType: "ml",
        body: "sklearn on a built-in set such as iris, digits, wine, or breast cancer, or on a file path named in the goal.",
      },
    ],
  },
  {
    id: "retrieval",
    task: "Retrieval",
    summary:
      "Local sentence-transformer training. These workers do not run on the Mini LoRA host.",
    algorithms: [
      {
        name: "embedding",
        taskType: "embed",
        body: "Triplet loss so domain memory retrieves the right neighborhood.",
      },
      {
        name: "reranker",
        taskType: "rerank",
        body: "Pairwise ranking loss applied after retrieval.",
      },
    ],
  },
  {
    id: "generative-recommendation",
    task: "Generative recommendation",
    summary:
      "Sequential transduction on user–item histories. From-scratch PyTorch HSTU in Astra (LSTM-style train loop) — not Mini MLX.",
    algorithms: [
      {
        name: "HSTU",
        taskType: "hstu",
        body: "Next-item HSTU on MovieLens-1M (auto-download); goal metric is Hit Rate@10.",
      },
    ],
  },
];

export default function DocsPage() {
  return (
    <div className="max-w-7xl mx-auto px-6 py-8">
      <div className="mb-8">
        <p className="text-[10px] tracking-[0.25em] uppercase text-[#14b8a6] mb-2">
          reference
        </p>
        <h1 className="text-lg font-semibold text-[#e2e8f0] tracking-wide">Docs</h1>
        <p className="text-xs text-[#94a3b8] mt-2 max-w-2xl leading-relaxed">
          A task is the objective. An algorithm is the update rule under that task.
          The goal bar selects the algorithm; <span className="text-[#14b8a6]">task_type</span> is
          the worker that runs it.
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-[220px_minmax(0,1fr)] gap-8">
        <nav aria-label="Tasks" className="lg:sticky lg:top-6 lg:self-start">
          <p className="text-[10px] tracking-widest uppercase text-[#64748b] mb-3">Tasks</p>
          <ol className="space-y-1">
            {TASKS.map((task, i) => (
              <li key={task.id}>
                <a
                  href={`#${task.id}`}
                  className="flex items-baseline gap-2 px-2 py-1.5 rounded text-xs text-[#94a3b8] hover:text-[#e2e8f0] hover:bg-[#1e293b] transition-colors"
                >
                  <span className="text-[#14b8a6] tabular-nums w-4">{String(i + 1).padStart(2, "0")}</span>
                  {task.task}
                </a>
              </li>
            ))}
          </ol>
        </nav>

        <div className="space-y-6">
          {TASKS.map((task) => (
            <section
              key={task.id}
              id={task.id}
              className="scroll-mt-6 border border-[rgba(20,184,166,0.15)] bg-[#1e293b]/70 rounded-lg"
            >
              <header className="px-5 py-4 border-b border-[rgba(20,184,166,0.1)]">
                <h2 className="text-sm font-semibold text-[#e2e8f0]">{task.task}</h2>
                <p className="text-xs text-[#94a3b8] mt-2 leading-relaxed">{task.summary}</p>
              </header>
              <ul>
                {task.algorithms.map((algo) => (
                  <li
                    key={`${task.id}-${algo.name}-${algo.taskType}`}
                    className="px-5 py-4 border-t border-[rgba(20,184,166,0.08)] first:border-t-0 grid grid-cols-1 sm:grid-cols-[180px_minmax(0,1fr)] gap-2 sm:gap-6"
                  >
                    <div>
                      <h3 className="text-xs text-[#e2e8f0]">{algo.name}</h3>
                      <p className="text-[10px] text-[#14b8a6] mt-1 tracking-wide">{algo.taskType}</p>
                    </div>
                    <p className="text-xs text-[#94a3b8] leading-relaxed">{algo.body}</p>
                  </li>
                ))}
              </ul>
            </section>
          ))}
        </div>
      </div>
    </div>
  );
}
