"use client";

import type { Mission } from "@/lib/api";

interface Props {
  mission: Mission;
  target?: number;
}

function ArcGauge({ pct, achieved }: { pct: number; achieved: boolean }) {
  const R = 52;
  const cx = 64;
  const cy = 68;
  // 210° arc: from 195° to 345° (bottom-left sweeping up and over to bottom-right)
  const startAngle = 195;
  const totalAngle = 210;

  function polar(angleDeg: number, r = R) {
    const rad = ((angleDeg - 90) * Math.PI) / 180;
    return { x: cx + r * Math.cos(rad), y: cy + r * Math.sin(rad) };
  }

  const start = polar(startAngle);
  const end = polar(startAngle + totalAngle);
  const fillEnd = polar(startAngle + totalAngle * Math.min(pct / 100, 1));
  const largeArc = totalAngle > 180 ? 1 : 0;
  const fillLarge = totalAngle * (pct / 100) > 180 ? 1 : 0;

  const trackPath = `M ${start.x} ${start.y} A ${R} ${R} 0 ${largeArc} 1 ${end.x} ${end.y}`;
  const fillPath =
    pct > 0
      ? `M ${start.x} ${start.y} A ${R} ${R} 0 ${fillLarge} 1 ${fillEnd.x} ${fillEnd.y}`
      : null;

  const strokeColor = achieved ? "#4ade80" : "#14b8a6";
  const glowColor = achieved ? "rgba(74,222,128,0.4)" : "rgba(20,184,166,0.4)";

  return (
    <svg width="128" height="88" viewBox="0 0 128 88" fill="none">
      <defs>
        <filter id="glow">
          <feGaussianBlur stdDeviation="2.5" result="blur" />
          <feMerge>
            <feMergeNode in="blur" />
            <feMergeNode in="SourceGraphic" />
          </feMerge>
        </filter>
      </defs>
      {/* Track */}
      <path
        d={trackPath}
        stroke="rgba(255,255,255,0.05)"
        strokeWidth="6"
        strokeLinecap="round"
        fill="none"
      />
      {/* Fill */}
      {fillPath && (
        <path
          d={fillPath}
          stroke={strokeColor}
          strokeWidth="6"
          strokeLinecap="round"
          fill="none"
          filter="url(#glow)"
          style={{
            filter: `drop-shadow(0 0 6px ${glowColor})`,
          }}
        />
      )}
      {/* Tick marks at target */}
      <circle cx={end.x} cy={end.y} r="2" fill="rgba(20,184,166,0.3)" />
    </svg>
  );
}

export function MetricGap({ mission, target = 0.92 }: Props) {
  const current = parseFloat(mission.best_metric_value ?? "0");
  const gap = Math.max(0, target - current);
  const pct = Math.min(100, (current / target) * 100);
  const achieved = gap <= 0;

  return (
    <div className="bg-[#1e293b] border border-[rgba(20,184,166,0.15)] rounded-lg p-5">
      <div className="flex items-baseline justify-between mb-2">
        <span className="text-xs text-[#94a3b8] tracking-widest uppercase">Metric Gap</span>
        <span className="text-[#94a3b8] text-[10px]">target {(target * 100).toFixed(0)}%</span>
      </div>

      <div className="flex items-center gap-4">
        <div className="relative shrink-0">
          <ArcGauge pct={pct} achieved={achieved} />
          <div className="absolute inset-0 flex flex-col items-center justify-center pb-2">
            <span
              className="text-2xl font-semibold leading-none"
              style={{ color: achieved ? "#4ade80" : "#e2e8f0" }}
            >
              {(current * 100).toFixed(1)}
            </span>
            <span className="text-[10px] text-[#94a3b8] mt-0.5">%</span>
          </div>
        </div>

        <div className="flex-1 min-w-0">
          <div className="text-xs text-[#94a3b8] mb-1">iter {mission.current_iteration}</div>
          {achieved ? (
            <div className="text-xs text-[#4ade80] font-medium">Target achieved</div>
          ) : (
            <div className="text-xs text-[#f87171]">
              −{(gap * 100).toFixed(2)}% to close
            </div>
          )}
          <div className="mt-2 text-[10px] text-[#64748b]">
            {pct.toFixed(0)}% of target
          </div>
        </div>
      </div>
    </div>
  );
}
