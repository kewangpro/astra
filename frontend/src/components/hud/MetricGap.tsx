"use client";

import type { Mission } from "@/lib/api";

interface Props {
  mission: Mission;
  target?: number;
}

export function MetricGap({ mission, target = 0.92 }: Props) {
  const current = mission.best_metric ?? 0;
  const gap = Math.max(0, target - current);
  const pct = Math.min(100, (current / target) * 100);

  return (
    <div className="bg-[#0d0d1a] border border-[rgba(20,184,166,0.15)] rounded-lg p-5">
      <div className="flex items-baseline justify-between mb-1">
        <span className="text-xs text-[#475569] tracking-widest uppercase">Metric Gap</span>
        <span className="text-[#14b8a6] text-xs">target {(target * 100).toFixed(0)}%</span>
      </div>

      <div className="flex items-end gap-4 mt-3">
        <div>
          <div className="text-3xl font-semibold text-[#e2e8f0]">
            {(current * 100).toFixed(2)}
            <span className="text-base text-[#64748b] ml-1">%</span>
          </div>
          <div className="text-xs text-[#f87171] mt-1">
            Δ −{(gap * 100).toFixed(2)}% to close
          </div>
        </div>
        <div className="flex-1 text-right">
          <div className="text-xs text-[#475569] mb-1">iter {mission.iteration}</div>
        </div>
      </div>

      <div className="mt-4 h-1.5 bg-[#1a1a35] rounded-full overflow-hidden">
        <div
          className="h-full bg-[#14b8a6] rounded-full transition-all duration-700"
          style={{ width: `${pct}%` }}
        />
      </div>
    </div>
  );
}
