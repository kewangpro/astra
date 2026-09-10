"use client";

import { useMemo } from "react";
import { useRouter } from "next/navigation";
import {
  TrendingUp,
  Target,
  Server,
  Play,
  Square,
  ArrowUpRight,
} from "lucide-react";
import { useMissions, useRunMission, useCancelMission } from "@/lib/hooks/useMissions";
import type { Mission } from "@/lib/api";
import { parseTs } from "@/lib/date";

const STATUS_COLOR: Record<string, string> = {
  pending:    "#475569",
  planning:   "#60a5fa",
  running:    "#14b8a6",
  paused:     "#fbbf24",
  evaluating: "#a78bfa",
  completed:  "#4ade80",
  failed:     "#f87171",
  stalled:    "#fb923c",
};

function fmtTs(iso: string): string {
  return parseTs(iso).toLocaleString([], {
    month: "numeric",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
}

function formatRelativeTime(iso: string): string {
  try {
    const d = parseTs(iso);
    const now = Date.now();
    const diffSec = Math.floor((now - d.getTime()) / 1000);
    if (diffSec < 60) return "just now";
    const diffMin = Math.floor(diffSec / 60);
    if (diffMin < 60) return `${diffMin}m ago`;
    const diffHour = Math.floor(diffMin / 60);
    if (diffHour < 24) return `${diffHour}h ago`;
    const diffDay = Math.floor(diffHour / 24);
    if (diffDay < 7) return `${diffDay}d ago`;
    return d.toLocaleDateString([], { month: "short", day: "numeric" });
  } catch {
    return iso;
  }
}

function getTargetProgress(m: Mission): { targetKey: string; targetVal: number; progressPct: number } | null {
  if (!m.target_metric) return null;
  const entries = Object.entries(m.target_metric);
  if (entries.length === 0) return null;
  const [key, targetVal] = entries[0];
  if (typeof targetVal !== "number" || targetVal <= 0) return null;

  const currentNum = m.best_metric_value ? parseFloat(m.best_metric_value) : null;
  if (currentNum === null || Number.isNaN(currentNum)) return null;

  const pct = Math.min(100, Math.max(0, (currentNum / targetVal) * 100));
  return { targetKey: key, targetVal, progressPct: Math.round(pct) };
}

// run/stop are both mission-control actions — keep them visually consistent
// (matching STATUS_COLOR.running) instead of tracking the card's current
// per-status color, which made "run" (pending, gray) and "stop"
// (running/planning/evaluating, teal/blue/purple) look mismatched.
const ACTION_COLOR = STATUS_COLOR.running;

function SkeletonCard() {
  return (
    <div className="bg-[#1e293b] border border-[rgba(255,255,255,0.04)] rounded-lg p-3.5 animate-pulse space-y-3">
      <div className="flex justify-between items-center">
        <div className="h-2.5 w-14 bg-[#2d3f57] rounded" />
        <div className="h-2.5 w-14 bg-[#2d3f57] rounded" />
      </div>
      <div className="h-2.5 bg-[#2d3f57] rounded mb-1" />
      <div className="h-2.5 bg-[#2d3f57] rounded w-3/4 mb-3" />
      <div className="h-10 bg-[#2d3f57]/30 rounded" />
      <div className="flex justify-between pt-1 border-t border-[rgba(255,255,255,0.04)]">
        <div className="h-2 w-10 bg-[#2d3f57] rounded" />
        <div className="h-2 w-10 bg-[#2d3f57] rounded" />
      </div>
    </div>
  );
}

function MissionCard({ m }: { m: Mission }) {
  const run = useRunMission();
  const cancel = useCancelMission();
  const router = useRouter();
  const color = STATUS_COLOR[m.status] ?? STATUS_COLOR.pending;
  const isRunning = m.status === "running" || m.status === "planning" || m.status === "evaluating";
  const isPending = m.status === "pending";
  const targetProgress = getTargetProgress(m);

  const bestValFormatted = useMemo(() => {
    if (m.best_metric_value === null || m.best_metric_value === undefined) return null;
    const num = parseFloat(m.best_metric_value);
    return Number.isFinite(num) ? num.toFixed(2) : m.best_metric_value;
  }, [m.best_metric_value]);

  function handleStop(e: React.MouseEvent) {
    e.preventDefault();
    e.stopPropagation();
    fetch(`/api/agent/missions/${m.id}/cancel`, { method: "POST" })
      .then(() => cancel.reset())
      .catch(() => null);
  }

  function handleRun(e: React.MouseEvent) {
    e.preventDefault();
    e.stopPropagation();
    run.mutate(m.id, {
      onSuccess: () => router.push(`/missions/${m.id}`),
    });
  }

  return (
    <div
      className="group cursor-pointer"
      onClick={() => router.push(`/missions/${m.id}`)}
    >
      <div
        className="relative rounded-lg p-3.5 transition-all duration-200 overflow-hidden"
        style={{
          background: "#1e293b",
          border: "1px solid rgba(255,255,255,0.05)",
          borderLeft: `2px solid ${color}`,
        }}
      >
        {/* Running shimmer */}
        {isRunning && (
          <div
            className="absolute inset-0 opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none"
            style={{
              background: `linear-gradient(135deg, transparent 40%, ${color}06 100%)`,
            }}
          />
        )}

        {/* Header: ID, Task Type, Remote Host, Live Status */}
        <div className="flex items-start justify-between gap-2 mb-2.5">
          <div className="flex items-center gap-1.5 flex-wrap">
            <span className="text-[10px] text-[#64748b] tracking-widest font-mono">
              #{m.id.length > 8 ? m.id.slice(0, 8) : m.id}
            </span>
            {m.task_type && (
              <span
                className="text-[9px] px-1 py-0.5 rounded-sm text-[#94a3b8] uppercase tracking-wider font-medium"
                style={{ background: "rgba(255,255,255,0.05)" }}
              >
                {m.task_type}
              </span>
            )}
            {m.host && (
              <span
                className="inline-flex items-center gap-1 text-[9px] px-1 py-0.5 rounded-sm text-[#94a3b8]"
                style={{ background: "rgba(255,255,255,0.05)" }}
                title={`Running on ${m.host}`}
              >
                <Server className="w-2.5 h-2.5 text-[#64748b]" />
                <span className="max-w-[70px] truncate">{m.host}</span>
              </span>
            )}
          </div>
          <span
            className="inline-flex items-center gap-1 text-[10px] px-1.5 py-0.5 rounded-sm uppercase tracking-widest font-medium shrink-0"
            style={{
              color,
              background: `${color}15`,
            }}
          >
            {isRunning && (
              <span className="relative flex h-1.5 w-1.5">
                <span
                  className="animate-ping absolute inline-flex h-full w-full rounded-full opacity-75"
                  style={{ backgroundColor: color }}
                />
                <span
                  className="relative inline-flex rounded-full h-1.5 w-1.5"
                  style={{ backgroundColor: color }}
                />
              </span>
            )}
            {m.status}
          </span>
        </div>

        {/* Goal text */}
        <p className="text-[13px] text-[#94a3b8] leading-relaxed line-clamp-2 mb-3 group-hover:text-[#cbd5e1] transition-colors">
          {m.goal}
        </p>

        {/* Metric & Progress Display */}
        <div
          className="rounded p-2 mb-3 space-y-1.5"
          style={{
            background: "rgba(0,0,0,0.2)",
            border: "1px solid rgba(255,255,255,0.03)",
          }}
        >
          <div className="flex items-center justify-between text-[11px]">
            <span className="text-[10px] text-[#64748b] flex items-center gap-1">
              <TrendingUp className="w-3 h-3 text-[#64748b]" />
              metric
            </span>
            <span className="font-medium font-mono" style={{ color: bestValFormatted !== null ? color : "#64748b" }}>
              {bestValFormatted !== null ? bestValFormatted : "—"}
            </span>
          </div>

          {targetProgress ? (
            <div className="space-y-1 pt-1 border-t border-[rgba(255,255,255,0.04)]">
              <div className="flex justify-between text-[9px] text-[#64748b]">
                <span className="flex items-center gap-1">
                  <Target className="w-2.5 h-2.5 text-[#64748b]" />
                  target {targetProgress.targetVal}
                </span>
                <span className="font-mono text-[#94a3b8]">{targetProgress.progressPct}%</span>
              </div>
              <div
                className="w-full rounded-full h-1 overflow-hidden"
                style={{ background: "rgba(255,255,255,0.06)" }}
              >
                <div
                  className="h-full rounded-full transition-all duration-300"
                  style={{
                    width: `${targetProgress.progressPct}%`,
                    backgroundColor: color,
                  }}
                />
              </div>
            </div>
          ) : (
            <div className="flex items-center justify-between text-[10px] text-[#64748b] pt-0.5 border-t border-[rgba(255,255,255,0.04)]">
              <span>iter {m.current_iteration}</span>
              {m.best_metric_iteration !== null && m.best_metric_iteration !== undefined && (
                <span className="text-[#475569] font-mono">(best @ {m.best_metric_iteration})</span>
              )}
            </div>
          )}
        </div>

        {/* Footer: Date & Controls */}
        <div className="flex items-center justify-between text-[9px] text-[#475569] tracking-wide pt-1 border-t border-[rgba(255,255,255,0.04)]">
          <span title={`created ${fmtTs(m.created_at)}${m.completed_at ? ` · ended ${fmtTs(m.completed_at)}` : ""}`}>
            {formatRelativeTime(m.created_at)}
          </span>

          <div className="flex items-center gap-2">
            {isPending && (
              <button
                onClick={handleRun}
                disabled={run.isPending}
                className="inline-flex items-center gap-1 px-2 py-0.5 rounded-sm border text-[10px] transition-colors disabled:opacity-40"
                style={{
                  borderColor: `${ACTION_COLOR}30`,
                  color: ACTION_COLOR,
                }}
                onMouseEnter={(e) => {
                  (e.currentTarget as HTMLButtonElement).style.background = `${ACTION_COLOR}10`;
                }}
                onMouseLeave={(e) => {
                  (e.currentTarget as HTMLButtonElement).style.background = "transparent";
                }}
              >
                <Play className="w-2.5 h-2.5 fill-current" />
                run
              </button>
            )}

            {isRunning && (
              <button
                onClick={handleStop}
                disabled={cancel.isPending}
                className="inline-flex items-center gap-1 px-2 py-0.5 rounded-sm border text-[10px] transition-colors disabled:opacity-40"
                style={{
                  borderColor: `${ACTION_COLOR}30`,
                  color: ACTION_COLOR,
                }}
                onMouseEnter={(e) => {
                  (e.currentTarget as HTMLButtonElement).style.background = `${ACTION_COLOR}10`;
                }}
                onMouseLeave={(e) => {
                  (e.currentTarget as HTMLButtonElement).style.background = "transparent";
                }}
              >
                <Square className="w-2.5 h-2.5 fill-current" />
                stop
              </button>
            )}

            <span className="text-[#64748b] group-hover:text-[#cbd5e1] transition-colors">
              <ArrowUpRight className="w-3 h-3" />
            </span>
          </div>
        </div>
      </div>
    </div>
  );
}

// Kanban columns layout in workflow order.
const MISSION_COLUMNS: { key: string; label: string; match: (s: string) => boolean }[] = [
  { key: "running",   label: "Running",   match: (s) => !["completed", "failed", "stalled", "paused"].includes(s) },
  { key: "completed", label: "Completed", match: (s) => s === "completed" },
  { key: "stalled",   label: "Stalled",   match: (s) => s === "stalled" || s === "paused" },
  { key: "failed",    label: "Failed",    match: (s) => s === "failed" },
];

function KanbanColumn({
  label,
  missions,
}: {
  label: string;
  missions: Mission[];
}) {
  return (
    <div className="flex flex-col min-w-0 bg-[#0f172a]/30 rounded-xl p-3 border border-[rgba(255,255,255,0.03)]">
      <div className="flex items-center justify-between pb-2 mb-3 border-b border-[rgba(255,255,255,0.04)]">
        <h3 className="text-[10px] text-[#64748b] tracking-widest uppercase flex items-center gap-2 font-medium">
          {label}
          <span className="text-[#475569]">{missions.length}</span>
        </h3>
      </div>
      <div
        className="space-y-3 overflow-y-auto max-h-[calc(100vh-250px)] pr-1"
        style={{
          scrollbarWidth: "thin",
          scrollbarColor: "rgba(255,255,255,0.1) transparent",
        }}
      >
        {missions.length === 0 ? (
          <div
            className="py-12 text-center border border-dashed rounded-lg"
            style={{ borderColor: "rgba(255,255,255,0.04)" }}
          >
            <span className="text-[10px] text-[#475569] tracking-widest uppercase">
              No missions
            </span>
          </div>
        ) : (
          missions.map((m) => <MissionCard key={m.id} m={m} />)
        )}
      </div>
    </div>
  );
}

export function MissionsGrid() {
  const { data: missions, isLoading, error } = useMissions();

  if (isLoading)
    return (
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 items-start">
        {Array.from({ length: 4 }).map((_, colIdx) => (
          <div key={colIdx} className="space-y-3 bg-[#0f172a]/30 rounded-xl p-3 border border-[rgba(255,255,255,0.03)]">
            <div className="h-2 w-14 bg-[#2d3f57] rounded mb-3 animate-pulse" />
            <SkeletonCard />
            <SkeletonCard />
          </div>
        ))}
      </div>
    );

  if (error)
    return (
      <div className="text-center py-16 text-[#f87171] text-xs tracking-widest">
        BACKEND_UNREACHABLE — run <code className="text-[#e2e8f0]">make run</code>
      </div>
    );

  if (!missions?.length)
    return (
      <div className="py-16 flex flex-col items-center gap-3">
        <div className="flex gap-1">
          {[...Array(5)].map((_, i) => (
            <div
              key={i}
              className="w-0.5 rounded-full bg-[#2d3f57]"
              style={{ height: 12 + (i % 3) * 6 }}
            />
          ))}
        </div>
        <span className="text-[10px] text-[#64748b] tracking-widest uppercase">
          No missions — define an objective above
        </span>
      </div>
    );

  // Newest first within each group, by creation time.
  const ordered = [...missions].sort(
    (a, b) => parseTs(b.created_at).getTime() - parseTs(a.created_at).getTime(),
  );

  return (
    <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 items-start">
      {MISSION_COLUMNS.map((col) => (
        <KanbanColumn
          key={col.key}
          label={col.label}
          missions={ordered.filter((m) => col.match(m.status))}
        />
      ))}
    </div>
  );
}

export const MissionsKanban = MissionsGrid;
