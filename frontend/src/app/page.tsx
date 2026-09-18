"use client";

import Link from "next/link";
import { GoalInput } from "@/components/command-center/GoalInput";
import { MissionsGrid } from "@/components/command-center/MissionsGrid";
import { NodesPanel } from "@/components/command-center/NodesPanel";
import { useMissions } from "@/lib/hooks/useMissions";

function GlobalStats() {
  const { data: missions } = useMissions();
  if (!missions?.length) return null;

  const counts = missions.reduce<Record<string, number>>((a, m) => {
    a[m.status] = (a[m.status] ?? 0) + 1;
    return a;
  }, {});

  const runningCount =
    (counts.running ?? 0) +
    (counts.planning ?? 0) +
    (counts.evaluating ?? 0) +
    (counts.pending ?? 0);
  const stalledCount = (counts.stalled ?? 0) + (counts.paused ?? 0);

  const stats = [
    { label: "Total", value: missions.length },
    { label: "Running", value: runningCount, color: "#14b8a6" },
    { label: "Completed", value: counts.completed ?? 0, color: "#4ade80", href: "/completed" },
    { label: "Stalled", value: stalledCount, color: "#fb923c" },
    { label: "Failed", value: counts.failed ?? 0, color: "#f87171" },
  ];

  return (
    <div className="flex gap-6 text-xs text-[#94a3b8]">
      {stats.map((s) =>
        s.href ? (
          <Link
            key={s.label}
            href={s.href}
            className="flex items-center gap-1.5 hover:underline transition-all group"
            title="View Completed Missions Archive"
          >
            {s.color && (
              <span
                className="inline-block w-1.5 h-1.5 rounded-full"
                style={{ background: s.color }}
              />
            )}
            <span style={{ color: s.color ?? "#475569" }} className="font-semibold">
              {s.value}
            </span>
            <span className="group-hover:text-[#e2e8f0]">{s.label} ↗</span>
          </Link>
        ) : (
          <div key={s.label} className="flex items-center gap-1.5">
            {s.color && (
              <span
                className="inline-block w-1.5 h-1.5 rounded-full"
                style={{ background: s.color }}
              />
            )}
            <span style={{ color: s.color ?? "#475569" }}>{s.value}</span>
            <span>{s.label}</span>
          </div>
        )
      )}
    </div>
  );
}

export default function CommandCenter() {
  const { data: missions } = useMissions();
  const completedCount = missions?.filter((m) => m.status === "completed").length ?? 0;

  return (
    <div className="max-w-7xl mx-auto px-6 py-8 space-y-8">
      <div className="flex items-end justify-between">
        <div>
          <h1 className="text-lg font-semibold text-[#e2e8f0] tracking-wide">
            Command Center
          </h1>
          <p className="text-xs text-[#94a3b8] mt-0.5">
            Define a training goal and launch an autonomous mission
          </p>
        </div>
        <GlobalStats />
      </div>

      <GoalInput />

      <NodesPanel />

      <div>
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center gap-2">
            <h2 className="text-xs text-[#94a3b8] tracking-widest uppercase font-medium">
              Operational Board
            </h2>
            <span className="text-[10px] text-[#64748b]">
              (Running, Stalled, Failed)
            </span>
          </div>
          {completedCount > 0 && (
            <Link
              href="/completed"
              className="inline-flex items-center gap-1.5 text-xs text-[#4ade80] hover:text-[#86efac] bg-[#4ade80]/10 hover:bg-[#4ade80]/15 border border-[#4ade80]/20 px-3 py-1 rounded transition-colors font-medium"
            >
              <span>🏆</span> View Completed Missions ({completedCount}) →
            </Link>
          )}
        </div>
        <MissionsGrid />
      </div>
    </div>
  );
}

