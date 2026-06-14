"use client";

import type { TelemetryEvent } from "@/lib/api";

interface Props {
  events: TelemetryEvent[];
}

export function PivotTimeline({ events }: Props) {
  const pivots = events.filter(
    (e) =>
      e.event === "pivot" ||
      e.event === "loop.pivot" ||
      e.event.includes("pivot")
  );

  if (!pivots.length) return null;

  return (
    <div className="bg-[#1e293b] border border-[rgba(20,184,166,0.15)] rounded-lg p-5">
      <div className="text-xs text-[#94a3b8] tracking-widest uppercase mb-4">
        Pivot History
      </div>
      <ol className="relative border-l border-[rgba(20,184,166,0.15)] space-y-5 pl-6">
        {pivots.map((p, i) => (
          <li key={i} className="relative">
            <span className="absolute -left-[22px] top-1 w-3 h-3 rounded-full bg-[#fbbf24] border-2 border-[#0f172a]" />
            <div className="text-[11px] text-[#fbbf24] mb-0.5">
              iter {String(p.data.iteration ?? "?")} — pivot triggered
            </div>
            <div className="text-[11px] text-[#64748b]">
              {p.data.reason ? String(p.data.reason) : "plateau detected"}
            </div>
            <div className="text-[10px] text-[#64748b] mt-0.5">
              {new Date(p.ts).toLocaleString()}
            </div>
          </li>
        ))}
      </ol>
    </div>
  );
}
