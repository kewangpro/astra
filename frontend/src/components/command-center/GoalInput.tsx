"use client";

import { useState } from "react";
import { useCreateMission, useRunMission } from "@/lib/hooks/useMissions";
import { useRouter } from "next/navigation";

const DOMAINS = ["general", "vision", "nlp", "rl", "tabular"];

export function GoalInput() {
  const [goal, setGoal] = useState("");
  const [domain, setDomain] = useState("general");
  const router = useRouter();
  const create = useCreateMission();
  const run = useRunMission();

  const submit = async () => {
    if (!goal.trim()) return;
    const mission = await create.mutateAsync({ goal: goal.trim(), domain });
    await run.mutateAsync(mission.id);
    router.push(`/missions/${mission.id}`);
  };

  const loading = create.isPending || run.isPending;

  return (
    <div className="bg-[#0d0d1a] border border-[rgba(20,184,166,0.2)] rounded-lg p-6">
      <label className="block text-xs text-[#64748b] tracking-widest mb-3 uppercase">
        Training Goal
      </label>
      <textarea
        value={goal}
        onChange={(e) => setGoal(e.target.value)}
        placeholder="e.g. Train a ResNet-18 to achieve > 92% accuracy on CIFAR-10 within 20 iterations"
        rows={3}
        className="w-full bg-[#12122a] border border-[rgba(20,184,166,0.15)] rounded px-4 py-3
                   text-sm text-[#e2e8f0] placeholder-[#334155] resize-none
                   focus:outline-none focus:border-[#14b8a6] transition-colors"
        onKeyDown={(e) => {
          if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) submit();
        }}
      />
      <div className="mt-3 flex items-center gap-3">
        <select
          value={domain}
          onChange={(e) => setDomain(e.target.value)}
          className="bg-[#12122a] border border-[rgba(20,184,166,0.15)] rounded px-3 py-2
                     text-sm text-[#94a3b8] focus:outline-none focus:border-[#14b8a6] transition-colors"
        >
          {DOMAINS.map((d) => (
            <option key={d} value={d}>
              {d}
            </option>
          ))}
        </select>
        <button
          onClick={submit}
          disabled={loading || !goal.trim()}
          className="ml-auto flex items-center gap-2 px-5 py-2 rounded
                     bg-[#14b8a6] hover:bg-[#0d9488] disabled:opacity-40 disabled:cursor-not-allowed
                     text-[#070710] text-sm font-semibold transition-colors"
        >
          {loading ? (
            <>
              <span className="inline-block w-3 h-3 border-2 border-[#070710] border-t-transparent rounded-full animate-spin" />
              Launching…
            </>
          ) : (
            "Launch Mission ⌘↵"
          )}
        </button>
      </div>
    </div>
  );
}
