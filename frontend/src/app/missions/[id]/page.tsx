"use client";

import { use } from "react";
import Link from "next/link";
import { useMission, useMetrics } from "@/lib/hooks/useMissions";
import { useTelemetry } from "@/lib/hooks/useTelemetry";
import { MetricGap } from "@/components/hud/MetricGap";
import { MetricChart } from "@/components/hud/MetricChart";
import { LogStream } from "@/components/hud/LogStream";
import { PivotTimeline } from "@/components/hud/PivotTimeline";
import { ApprovalPanel } from "@/components/approvals/ApprovalPanel";

const STATUS_COLOR: Record<string, string> = {
  pending:    "#94a3b8",
  planning:   "#60a5fa",
  running:    "#14b8a6",
  paused:     "#fbbf24",
  evaluating: "#a78bfa",
  completed:  "#4ade80",
  failed:     "#f87171",
};

export default function MissionHUD({
  params,
}: {
  params: Promise<{ id: string }>;
}) {
  const { id } = use(params);
  const missionId = Number(id);

  const { data: mission, isLoading } = useMission(missionId);
  const { data: metrics = [] } = useMetrics(missionId);
  const { events, connected } = useTelemetry(missionId);

  if (isLoading)
    return (
      <div className="flex items-center justify-center h-96 text-[#334155] text-sm">
        Loading mission…
      </div>
    );

  if (!mission)
    return (
      <div className="flex flex-col items-center justify-center h-96 gap-4 text-[#f87171] text-sm">
        Mission not found.
        <Link href="/" className="text-[#14b8a6] text-xs hover:underline">
          ← Command Center
        </Link>
      </div>
    );

  const statusColor = STATUS_COLOR[mission.status] ?? "#94a3b8";

  return (
    <div className="max-w-7xl mx-auto px-6 py-8 space-y-6">
      {/* Header */}
      <div className="flex items-start justify-between gap-4">
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-3 mb-1">
            <Link href="/" className="text-[#475569] text-xs hover:text-[#14b8a6] transition-colors">
              ← missions
            </Link>
            <span className="text-[#334155] text-xs">/ #{missionId}</span>
          </div>
          <p className="text-[#e2e8f0] text-sm leading-relaxed line-clamp-2">
            {mission.goal}
          </p>
        </div>
        <span
          className="shrink-0 text-[11px] px-3 py-1 rounded border uppercase tracking-widest"
          style={{ color: statusColor, borderColor: `${statusColor}40` }}
        >
          {mission.status}
        </span>
      </div>

      {/* Approval gate — shown prominently when pending */}
      <ApprovalPanel missionId={missionId} />

      {/* Metric row */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <MetricGap mission={mission} />
        <div className="md:col-span-2">
          <MetricChart metrics={metrics} />
        </div>
      </div>

      {/* Log + Pivots */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
        <div className="lg:col-span-2">
          <LogStream events={events} connected={connected} />
        </div>
        <div>
          <PivotTimeline events={events} />
        </div>
      </div>
    </div>
  );
}
