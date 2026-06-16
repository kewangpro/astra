"use client";

import {
  ResponsiveContainer,
  AreaChart,
  Area,
  XAxis,
  YAxis,
  Tooltip,
  ReferenceLine,
  CartesianGrid,
} from "recharts";
import type { TelemetryEvent } from "@/lib/api";

interface Props {
  events: TelemetryEvent[];
  targetMetric?: Record<string, number> | null;
}

const COLORS = ["#14b8a6", "#60a5fa", "#a78bfa", "#fbbf24", "#4ade80"];
const COLORS_DIM = ["#1e4a47", "#1e3356", "#3b2e56", "#4a3a10", "#1a4030"];

export function MetricChart({ events, targetMetric }: Props) {
  const metricEvents = events.filter(
    (e) => (e.type === "metric" || e.type === "backfill") && e.name != null && e.value != null
  );

  // Track which steps are "live" (current run) vs historical (backfill)
  const liveSteps = new Set(
    events
      .filter((e) => e.type === "metric" && e.name != null && e.value != null)
      .map((e) => e.step ?? e.iteration ?? 0)
  );

  const byStep = metricEvents.reduce<Record<number, Record<string, unknown>>>(
    (acc, e) => {
      const step = e.step ?? e.iteration ?? 0;
      if (!acc[step]) acc[step] = { step };
      const suffix = liveSteps.has(step) ? "_live" : "_hist";
      acc[step][`${e.name!}${suffix}`] = e.value!;
      // Keep both so the tooltip can show value regardless of which series
      acc[step][`_val_${e.name!}`] = e.value!;
      return acc;
    },
    {}
  );

  const data = Object.values(byStep).sort(
    (a, b) => (a.step as number) - (b.step as number)
  );

  const names = [...new Set(metricEvents.map((e) => e.name!))];
  const hasLive = liveSteps.size > 0;

  // Resolve target value and display mode from targetMetric dict
  const [targetName, targetValue] =
    targetMetric && Object.keys(targetMetric).length > 0
      ? [Object.keys(targetMetric)[0], Object.values(targetMetric)[0] as number]
      : [null, 0.92];
  const isRaw = targetValue > 1;

  const maxObserved = Math.max(
    0,
    ...data.map((d) => (targetName ? ((d[`_val_${targetName}`] as number) ?? 0) : 0))
  );
  const yDomain: [number, number] = isRaw
    ? [0, Math.max(targetValue * 1.1, maxObserved)]
    : [0, 1];

  const yFormatter = isRaw
    ? (v: number) => v.toFixed(0)
    : (v: number) => `${(v * 100).toFixed(0)}%`;

  const tooltipFormatter = (v: unknown, name: unknown) => {
    const val = v as number;
    const label = String(name).replace(/_(live|hist)$/, "");
    const formatted = isRaw ? val.toFixed(1) : `${(val * 100).toFixed(2)}%`;
    return [formatted, label] as [string, string];
  };

  const targetLabel = isRaw
    ? `target ${targetValue.toFixed(0)}`
    : `target ${(targetValue * 100).toFixed(0)}%`;

  if (!data.length)
    return (
      <div
        className="bg-[#1e293b] border border-[rgba(20,184,166,0.15)] rounded-lg p-5 h-56
                      flex flex-col items-center justify-center gap-2"
      >
        <div className="w-8 h-px bg-[rgba(20,184,166,0.2)]" />
        <span className="text-[#64748b] text-xs tracking-widest">NO METRICS YET</span>
        <div className="w-8 h-px bg-[rgba(20,184,166,0.2)]" />
      </div>
    );

  return (
    <div className="bg-[#1e293b] border border-[rgba(20,184,166,0.15)] rounded-lg p-5">
      <div className="flex items-center justify-between mb-4">
        <span className="text-xs text-[#94a3b8] tracking-widest uppercase">
          Metric History
        </span>
        <div className="flex items-center gap-4">
          {hasLive && (
            <div className="flex items-center gap-1.5">
              <span className="inline-block w-3 h-px bg-[#14b8a6]" />
              <span className="text-[10px] text-[#94a3b8]">current</span>
            </div>
          )}
          <div className="flex items-center gap-1.5">
            <span className="inline-block w-3 h-px bg-[#334155]" />
            <span className="text-[10px] text-[#64748b]">prior</span>
          </div>
        </div>
      </div>

      <svg width="0" height="0" style={{ position: "absolute" }}>
        <defs>
          {names.map((name, i) => (
            <linearGradient key={`${name}-live`} id={`grad-live-${i}`} x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor={COLORS[i % COLORS.length]} stopOpacity={0.2} />
              <stop offset="100%" stopColor={COLORS[i % COLORS.length]} stopOpacity={0} />
            </linearGradient>
          ))}
          {names.map((name, i) => (
            <linearGradient key={`${name}-hist`} id={`grad-hist-${i}`} x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor={COLORS_DIM[i % COLORS_DIM.length]} stopOpacity={0.4} />
              <stop offset="100%" stopColor={COLORS_DIM[i % COLORS_DIM.length]} stopOpacity={0} />
            </linearGradient>
          ))}
        </defs>
      </svg>

      <ResponsiveContainer width="100%" height={200}>
        <AreaChart data={data} margin={{ top: 4, right: 8, bottom: 0, left: -16 }}>
          <CartesianGrid
            strokeDasharray="2 4"
            stroke="rgba(255,255,255,0.03)"
            vertical={false}
          />
          <XAxis
            dataKey="step"
            tick={{ fontSize: 10, fill: "#334155" }}
            axisLine={false}
            tickLine={false}
          />
          <YAxis
            domain={yDomain}
            tickFormatter={yFormatter}
            tick={{ fontSize: 10, fill: "#334155" }}
            axisLine={false}
            tickLine={false}
          />
          <Tooltip
            contentStyle={{
              background: "#263347",
              border: "1px solid rgba(20,184,166,0.2)",
              borderRadius: 6,
              fontSize: 11,
              color: "#e2e8f0",
              boxShadow: "0 4px 24px rgba(0,0,0,0.4)",
            }}
            formatter={tooltipFormatter}
            cursor={{ stroke: "rgba(20,184,166,0.2)", strokeWidth: 1 }}
          />
          <ReferenceLine
            y={targetValue}
            stroke="rgba(20,184,166,0.35)"
            strokeDasharray="3 5"
            label={{
              value: targetLabel,
              position: "right",
              fontSize: 9,
              fill: "rgba(20,184,166,0.5)",
            }}
          />
          {/* Historical series — muted */}
          {names.map((name, i) => (
            <Area
              key={`${name}_hist`}
              type="monotone"
              dataKey={`${name}_hist`}
              stroke={COLORS_DIM[i % COLORS_DIM.length]}
              strokeWidth={1.5}
              fill={`url(#grad-hist-${i})`}
              dot={false}
              connectNulls={false}
              activeDot={false}
            />
          ))}
          {/* Live / current run series — bright */}
          {names.map((name, i) => (
            <Area
              key={`${name}_live`}
              type="monotone"
              dataKey={`${name}_live`}
              stroke={COLORS[i % COLORS.length]}
              strokeWidth={2}
              fill={`url(#grad-live-${i})`}
              dot={false}
              connectNulls={false}
              activeDot={{
                r: 4,
                fill: COLORS[i % COLORS.length],
                stroke: "#0f172a",
                strokeWidth: 2,
              }}
            />
          ))}
        </AreaChart>
      </ResponsiveContainer>
    </div>
  );
}
