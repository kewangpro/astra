"use client";

import Link from "next/link";
import { useMissions, useRunMission } from "@/lib/hooks/useMissions";
import type { Mission } from "@/lib/api";

const STATUS_STYLES: Record<string, string> = {
  pending:    "text-[#94a3b8] border-[#334155]",
  planning:   "text-[#60a5fa] border-[#60a5fa]/30",
  running:    "text-[#14b8a6] border-[#14b8a6]/40 animate-pulse-teal",
  paused:     "text-[#fbbf24] border-[#fbbf24]/30",
  evaluating: "text-[#a78bfa] border-[#a78bfa]/30",
  completed:  "text-[#4ade80] border-[#4ade80]/30",
  failed:     "text-[#f87171] border-[#f87171]/30",
};

function MissionCard({ m }: { m: Mission }) {
  const run = useRunMission();
  const style = STATUS_STYLES[m.status] ?? STATUS_STYLES.pending;

  return (
    <Link href={`/missions/${m.id}`} className="block group">
      <div className="bg-[#0d0d1a] border border-[rgba(20,184,166,0.12)] rounded-lg p-4
                      hover:border-[rgba(20,184,166,0.35)] hover:bg-[#12122a] transition-all">
        <div className="flex items-start justify-between gap-2 mb-3">
          <span className="text-[10px] text-[#475569] tracking-widest uppercase">
            #{m.id}
          </span>
          <span className={`text-[10px] px-2 py-0.5 rounded border ${style} uppercase tracking-wider`}>
            {m.status}
          </span>
        </div>
        <p className="text-sm text-[#cbd5e1] leading-relaxed line-clamp-2 mb-3">
          {m.goal}
        </p>
        <div className="flex items-center justify-between text-[11px] text-[#475569]">
          <span>iter {m.iteration}</span>
          {m.best_metric !== null && (
            <span className="text-[#14b8a6]">{(m.best_metric * 100).toFixed(1)}%</span>
          )}
        </div>
        {m.status === "pending" && (
          <button
            onClick={(e) => {
              e.preventDefault();
              run.mutate(m.id);
            }}
            className="mt-3 w-full text-[11px] py-1.5 rounded border border-[rgba(20,184,166,0.25)]
                       text-[#14b8a6] hover:bg-[rgba(20,184,166,0.1)] transition-colors"
          >
            Run
          </button>
        )}
      </div>
    </Link>
  );
}

export function MissionsGrid() {
  const { data: missions, isLoading, error } = useMissions();

  if (isLoading)
    return (
      <div className="text-center py-16 text-[#334155] text-sm">
        Loading missions…
      </div>
    );
  if (error)
    return (
      <div className="text-center py-16 text-[#f87171] text-sm">
        Backend unreachable — start with <code>make run</code>
      </div>
    );
  if (!missions?.length)
    return (
      <div className="text-center py-16 text-[#334155] text-sm">
        No missions yet. Launch one above.
      </div>
    );

  return (
    <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4">
      {[...missions].reverse().map((m) => (
        <MissionCard key={m.id} m={m} />
      ))}
    </div>
  );
}
