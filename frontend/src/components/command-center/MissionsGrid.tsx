"use client";

import { useRouter } from "next/navigation";
import { useMissions, useRunMission, useCancelMission } from "@/lib/hooks/useMissions";
import type { Mission } from "@/lib/api";

const STATUS_COLOR: Record<string, string> = {
  pending:    "#475569",
  planning:   "#60a5fa",
  running:    "#14b8a6",
  paused:     "#fbbf24",
  evaluating: "#a78bfa",
  completed:  "#4ade80",
  failed:     "#f87171",
};

// run/stop are both mission-control actions — keep them visually consistent
// (matching STATUS_COLOR.running) instead of tracking the card's current
// per-status color, which made "run" (pending, gray) and "stop"
// (running/planning/evaluating, teal/blue/purple) look mismatched.
const ACTION_COLOR = STATUS_COLOR.running;

function SkeletonCard() {
  return (
    <div className="bg-[#1e293b] border border-[rgba(255,255,255,0.04)] rounded-lg p-4 animate-pulse">
      <div className="flex justify-between mb-3">
        <div className="h-2 w-6 bg-[#2d3f57] rounded" />
        <div className="h-2 w-14 bg-[#2d3f57] rounded" />
      </div>
      <div className="h-2 bg-[#2d3f57] rounded mb-2" />
      <div className="h-2 bg-[#2d3f57] rounded w-3/4 mb-4" />
      <div className="flex justify-between">
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

  function handleStop(e: React.MouseEvent) {
    e.preventDefault();
    e.stopPropagation();
    fetch(`/api/agent/missions/${m.id}/cancel`, { method: "POST" })
      .then(() => cancel.reset())
      .catch(() => null);
  }

  return (
    <div
      className="group cursor-pointer"
      onClick={() => router.push(`/missions/${m.id}`)}
    >
      <div
        className="relative rounded-lg p-4 transition-all duration-200 overflow-hidden"
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

        <div className="flex items-start justify-between gap-2 mb-3">
          <span className="text-[10px] text-[#64748b] tracking-widest">#{m.id}</span>
          <span
            className="text-[10px] px-1.5 py-0.5 rounded-sm uppercase tracking-widest font-medium"
            style={{
              color,
              background: `${color}15`,
            }}
          >
            {m.status}
          </span>
        </div>

        <p className="text-[13px] text-[#94a3b8] leading-relaxed line-clamp-2 mb-4 group-hover:text-[#cbd5e1] transition-colors">
          {m.goal}
        </p>

        <div className="flex items-center justify-between">
          <span className="text-[10px] text-[#64748b]">iter {m.current_iteration}</span>
          {m.best_metric_value !== null && (
            <span className="text-[11px] font-medium" style={{ color }}>
              {Number.isFinite(parseFloat(m.best_metric_value))
                ? parseFloat(m.best_metric_value).toFixed(2)
                : m.best_metric_value}
            </span>
          )}
        </div>

        {m.status === "pending" && (
          <button
            onClick={(e) => {
              e.stopPropagation();
              run.mutate(m.id, {
                onSuccess: () => router.push(`/missions/${m.id}`),
              });
            }}
            disabled={run.isPending}
            className="mt-3 w-full text-[11px] py-1.5 rounded-sm border transition-colors disabled:opacity-40"
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
            run
          </button>
        )}
        {isRunning && (
          <button
            onClick={handleStop}
            className="mt-3 w-full text-[11px] py-1.5 rounded-sm border transition-colors"
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
            stop
          </button>
        )}
      </div>
    </div>
  );
}

export function MissionsGrid() {
  const { data: missions, isLoading, error } = useMissions();

  if (isLoading)
    return (
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4">
        {Array.from({ length: 4 }).map((_, i) => <SkeletonCard key={i} />)}
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

  return (
    <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4">
      {[...missions].reverse().map((m) => (
        <MissionCard key={m.id} m={m} />
      ))}
    </div>
  );
}
