"use client";

import { GoalInput } from "@/components/command-center/GoalInput";
import { MissionsGrid } from "@/components/command-center/MissionsGrid";
import { useMissions } from "@/lib/hooks/useMissions";

function GlobalStats() {
  const { data: missions } = useMissions();
  if (!missions?.length) return null;

  const counts = missions.reduce<Record<string, number>>((a, m) => {
    a[m.status] = (a[m.status] ?? 0) + 1;
    return a;
  }, {});

  const stats = [
    { label: "Total", value: missions.length },
    { label: "Running", value: counts.running ?? 0, color: "#14b8a6" },
    { label: "Completed", value: counts.completed ?? 0, color: "#4ade80" },
    { label: "Failed", value: counts.failed ?? 0, color: "#f87171" },
  ];

  return (
    <div className="flex gap-6 text-xs text-[#94a3b8]">
      {stats.map((s) => (
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
      ))}
    </div>
  );
}

export default function CommandCenter() {
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

      <div>
        <h2 className="text-xs text-[#94a3b8] tracking-widest uppercase mb-4">
          Missions
        </h2>
        <MissionsGrid />
      </div>
    </div>
  );
}
