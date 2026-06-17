"use client";

import { useState } from "react";
import { useCreateMission, useRunMission } from "@/lib/hooks/useMissions";
import { useRouter } from "next/navigation";

const MAX = 280;

export function GoalInput() {
  const [goal, setGoal] = useState("");
  const [focused, setFocused] = useState(false);
  const router = useRouter();
  const create = useCreateMission();
  const run = useRunMission();

  const submit = async () => {
    if (!goal.trim()) return;
    const mission = await create.mutateAsync({ goal: goal.trim(), taskType: "rl" });
    await run.mutateAsync(mission.id);
    router.push(`/missions/${mission.id}`);
  };

  const loading = create.isPending || run.isPending;
  const remaining = MAX - goal.length;

  return (
    <div
      className="rounded-lg transition-all duration-300"
      style={{
        background: "#1e293b",
        border: focused
          ? "1px solid rgba(20,184,166,0.5)"
          : "1px solid rgba(20,184,166,0.15)",
        boxShadow: focused
          ? "0 0 0 1px rgba(20,184,166,0.1), 0 0 40px rgba(20,184,166,0.05)"
          : "none",
      }}
    >
      {/* Terminal header bar */}
      <div className="flex items-center gap-2 px-4 py-2.5 border-b border-[rgba(20,184,166,0.08)]">
        <span className="text-[10px] text-[#64748b] tracking-widest uppercase">
          mission.objective
        </span>
      </div>

      {/* Input area */}
      <div className="flex gap-3 px-4 py-4">
        <span
          className="text-[#14b8a6] text-sm mt-0.5 select-none shrink-0"
          style={{ opacity: focused ? 1 : 0.4 }}
        >
          &gt;_
        </span>
        <textarea
          value={goal}
          onChange={(e) => setGoal(e.target.value.slice(0, MAX))}
          onFocus={() => setFocused(true)}
          onBlur={() => setFocused(false)}
          placeholder="define training objective — e.g. ResNet-18 achieving >92% on CIFAR-10 in 20 iterations"
          rows={2}
          className="flex-1 bg-transparent text-sm text-[#e2e8f0] placeholder-[#2d3748]
                     resize-none focus:outline-none leading-relaxed"
          onKeyDown={(e) => {
            if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) submit();
          }}
        />
      </div>

      {/* Footer */}
      <div className="flex items-center gap-3 px-4 py-2.5 border-t border-[rgba(20,184,166,0.08)]">
        <span
          className="text-[10px]"
          style={{ color: remaining < 40 ? "#f87171" : "#334155" }}
        >
          {remaining}
        </span>

        <div className="flex-1" />

        <span className="text-[10px] text-[#64748b] hidden sm:block">⌘↵ to launch</span>

        <button
          onClick={submit}
          disabled={loading || !goal.trim()}
          className="flex items-center gap-2 px-4 py-1.5 rounded text-xs font-semibold
                     transition-all duration-200 disabled:opacity-30 disabled:cursor-not-allowed"
          style={{
            background: goal.trim() && !loading ? "#14b8a6" : "rgba(20,184,166,0.1)",
            color: goal.trim() && !loading ? "#0f172a" : "#14b8a6",
            border: "1px solid rgba(20,184,166,0.3)",
          }}
        >
          {loading ? (
            <>
              <span className="w-3 h-3 border border-current border-t-transparent rounded-full animate-spin" />
              launching
            </>
          ) : (
            "launch mission"
          )}
        </button>
      </div>
    </div>
  );
}
