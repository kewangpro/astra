"use client";

import {
  LineChart, Line, ResponsiveContainer, Tooltip, ReferenceLine, YAxis,
} from "recharts";
import type { Mission, TelemetryEvent } from "@/lib/api";

interface Props {
  mission: Mission;
  events?: TelemetryEvent[];
}

function ArcGauge({ pct, achieved }: { pct: number; achieved: boolean }) {
  const R = 52;
  const cx = 64;
  const cy = 68;
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
      <path
        d={trackPath}
        stroke="rgba(255,255,255,0.05)"
        strokeWidth="6"
        strokeLinecap="round"
        fill="none"
      />
      {fillPath && (
        <path
          d={fillPath}
          stroke={strokeColor}
          strokeWidth="6"
          strokeLinecap="round"
          fill="none"
          filter="url(#glow)"
          style={{ filter: `drop-shadow(0 0 6px ${glowColor})` }}
        />
      )}
      <circle cx={end.x} cy={end.y} r="2" fill="rgba(20,184,166,0.3)" />
    </svg>
  );
}

export function MetricGap({ mission, events = [] }: Props) {
  const tm = mission.target_metric;
  const [metricName, targetValue] = tm && Object.keys(tm).length > 0
    ? [Object.keys(tm)[0], Object.values(tm)[0] as number]
    : ["metric", 0.92];

  const isRaw = targetValue > 1;

  const best = parseFloat(mission.best_metric_value ?? "0");
  const bestIter = mission.best_metric_iteration ?? null;
  const current = mission.current_metric_value != null
    ? parseFloat(mission.current_metric_value)
    : null;
  const currentIter = mission.current_iteration;

  // For raw positive targets (e.g. lines_cleared=20), clamp best to [0, ∞) so a
  // contaminated mean_reward seed doesn't show -600% of target.
  const displayBest = isRaw ? Math.max(0, best) : best;
  const pct = targetValue > 0 ? Math.min(100, (displayBest / targetValue) * 100) : 0;
  const gap = Math.max(0, targetValue - displayBest);
  const achieved = gap <= 0;

  const fmt = (v: number) =>
    isRaw ? v.toFixed(1) : `${(v * 100).toFixed(1)}%`;
  const formatTarget = isRaw ? targetValue.toFixed(0) : `${(targetValue * 100).toFixed(0)}%`;
  const formatGap = isRaw
    ? `−${gap.toFixed(1)} to close`
    : `−${(gap * 100).toFixed(2)}% to close`;

  const showCurrent = current != null && Math.abs(current - best) > 0.05;

  // Build sparkline data from target metric telemetry events
  const sparkData = events
    .filter((e) => (e.type === "metric" || e.type === "backfill") && e.name === metricName)
    .map((e) => ({ step: e.step ?? 0, value: e.value as number }))
    .sort((a, b) => a.step - b.step);

  const hasSparkData = sparkData.length > 1;

  return (
    <div className="bg-[#1e293b] border border-[rgba(20,184,166,0.15)] rounded-lg p-5">
      {/* Header */}
      <div className="flex items-baseline justify-between mb-3">
        <span className="text-xs text-[#94a3b8] tracking-widest uppercase">Metric Gap</span>
        <span className="text-[#94a3b8] text-[10px]">
          {metricName} · target {formatTarget}
        </span>
      </div>

      <div className="flex items-center gap-4">
        {/* Arc gauge + gap summary below it */}
        <div className="shrink-0 flex flex-col items-center gap-1">
          <div className="relative">
            <ArcGauge pct={pct} achieved={achieved} />
            <div className="absolute inset-0 flex flex-col items-center justify-center pt-6">
              <span
                className="text-2xl font-semibold leading-none"
                style={{ color: achieved ? "#4ade80" : "#e2e8f0" }}
              >
                {fmt(displayBest)}
              </span>
            </div>
          </div>
          {/* Gap summary — below the arc */}
          <div className="text-center">
            {achieved ? (
              <div className="text-xs text-[#4ade80] font-medium">Target achieved</div>
            ) : (
              <div className="text-xs text-[#f87171]">{formatGap}</div>
            )}
            <div className="text-[10px] text-[#64748b]">{pct.toFixed(0)}% of target</div>
          </div>
        </div>

        <div className="flex-1 min-w-0 space-y-2">
          {/* Best iter */}
          <div className="text-[10px] text-[#64748b]">
            best at iter {bestIter ?? "—"}
          </div>

          {/* Current iter — only show score when it differs from best */}
          <div className="text-[10px] text-[#94a3b8]">
            {showCurrent
              ? `iter ${currentIter}: ${fmt(current!)}`
              : `current iter: ${currentIter}`}
          </div>

          {/* Sparkline — target metric history */}
          {hasSparkData && (
            <div className="pt-1">
              <div className="text-[9px] text-[#64748b] mb-1 uppercase tracking-widest">
                {metricName} history
              </div>
              <ResponsiveContainer width="100%" height={52}>
                <LineChart data={sparkData} margin={{ top: 2, right: 4, bottom: 0, left: 0 }}>
                  <YAxis domain={["auto", "auto"]} hide />
                  <Tooltip
                    contentStyle={{
                      background: "#0f172a",
                      border: "1px solid rgba(20,184,166,0.2)",
                      borderRadius: 4,
                      fontSize: 10,
                      color: "#94a3b8",
                      padding: "2px 6px",
                    }}
                    formatter={(v: unknown) => [
                      isRaw
                        ? (v as number).toFixed(1)
                        : `${((v as number) * 100).toFixed(1)}%`,
                      metricName,
                    ]}
                    labelFormatter={() => ""}
                  />
                  {targetValue > 0 && (
                    <ReferenceLine
                      y={targetValue}
                      stroke="rgba(20,184,166,0.3)"
                      strokeDasharray="3 3"
                    />
                  )}
                  <Line
                    type="monotone"
                    dataKey="value"
                    stroke="#14b8a6"
                    strokeWidth={1.5}
                    dot={false}
                    isAnimationActive={false}
                  />
                </LineChart>
              </ResponsiveContainer>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
