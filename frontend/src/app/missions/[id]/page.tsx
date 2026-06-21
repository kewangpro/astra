"use client";

import { use, useState } from "react";
import Link from "next/link";
import { useMission } from "@/lib/hooks/useMissions";
import { useTelemetry } from "@/lib/hooks/useTelemetry";
import { MetricGap } from "@/components/hud/MetricGap";
import { MetricChart } from "@/components/hud/MetricChart";
import { LogStream } from "@/components/hud/LogStream";
import { PivotTimeline } from "@/components/hud/PivotTimeline";
import { CritiqueTrace } from "@/components/hud/CritiqueTrace";
import { SnakePlayer } from "@/components/hud/SnakePlayer";
import { TetrisPlayer } from "@/components/hud/TetrisPlayer";
import { ApprovalPanel } from "@/components/approvals/ApprovalPanel";
import type { TelemetryEvent } from "@/lib/api";

function SidebarLayout({
  events,
  connected,
  missionStatus,
}: {
  events: TelemetryEvent[];
  connected: boolean;
  missionStatus: string;
}) {
  const hasPivots = events.some((e) => e.type === "pivot");
  const hasCritiques = events.some((e) => e.type === "critique");
  const hasSidebar = hasPivots || hasCritiques;

  return (
    <div className={`grid grid-cols-1 gap-4 ${hasSidebar ? "lg:grid-cols-3" : ""}`}>
      <div className={hasSidebar ? "lg:col-span-2" : ""}>
        <LogStream
          events={events}
          connected={connected}
          missionStatus={missionStatus}
          className={hasSidebar ? "h-[36rem]" : ""}
        />
      </div>
      {hasSidebar && (
        <div className="space-y-4">
          {hasCritiques && <CritiqueTrace events={events} />}
          {hasPivots && <PivotTimeline events={events} />}
        </div>
      )}
    </div>
  );
}

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
  const { id: missionId } = use(params);

  const { data: mission, isLoading } = useMission(missionId);
  const { events, connected } = useTelemetry(missionId);
  const [autoApproveMode, setAutoApproveModeRaw] = useState(() => {
    try { return localStorage.getItem(`auto-approve:${missionId}`) === "1"; } catch { return false; }
  });
  const setAutoApproveMode = (on: boolean) => {
    setAutoApproveModeRaw(on);
    try { localStorage.setItem(`auto-approve:${missionId}`, on ? "1" : "0"); } catch {}
  };

  if (isLoading)
    return (
      <div className="flex items-center justify-center h-96 text-[#64748b] text-sm">
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
            <Link href="/" className="text-[#94a3b8] text-xs hover:text-[#14b8a6] transition-colors">
              ← missions
            </Link>
            <span className="text-[#64748b] text-xs">/ #{missionId}</span>
          </div>
          <p className="text-[#e2e8f0] text-sm leading-relaxed line-clamp-2">
            {mission.goal}
          </p>
        </div>
        <div className="flex items-center gap-2 shrink-0">
          <span
            className="text-[11px] px-3 py-1 rounded border uppercase tracking-widest"
            style={{ color: statusColor, borderColor: `${statusColor}40` }}
          >
            {mission.status}
          </span>
          <button
            onClick={() => setAutoApproveMode(!autoApproveMode)}
            title={autoApproveMode ? "Disable auto-approve" : "Enable auto-approve"}
            className={`text-[11px] px-2.5 py-1 rounded border transition-colors ${
              autoApproveMode
                ? "border-[#38bdf8]/50 text-[#38bdf8] bg-[#38bdf8]/10 hover:bg-[#38bdf8]/20"
                : "border-[#475569] text-[#64748b] hover:border-[#38bdf8]/40 hover:text-[#38bdf8]"
            }`}
          >
            ⚡
          </button>
        </div>
      </div>

      {/* Approval gate — shown prominently when pending */}
      <ApprovalPanel missionId={missionId} autoApproveMode={autoApproveMode} onAutoApproveModeChange={setAutoApproveMode} />

      {/* Metric row */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <MetricGap mission={mission} events={events} />
        <div className="md:col-span-2">
          <MetricChart events={events} targetMetric={mission.target_metric} />
        </div>
      </div>

      {/* Live agent viewer — shown for custom env missions */}
      {mission.goal.includes("Snake-v0") && (
        <SnakePlayer missionId={missionId} envId="Snake-v0" />
      )}
      {mission.goal.includes("Tetris-v0") && (
        <TetrisPlayer missionId={missionId} envId="Tetris-v0" />
      )}

      {/* Log + Critic Trace + Pivots */}
      <SidebarLayout events={events} connected={connected} missionStatus={mission.status} />
    </div>
  );
}
