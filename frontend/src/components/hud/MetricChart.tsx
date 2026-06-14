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
import type { Metric } from "@/lib/api";

interface Props {
  metrics: Metric[];
  target?: number;
}

const COLORS = ["#14b8a6", "#60a5fa", "#a78bfa", "#fbbf24", "#4ade80"];

export function MetricChart({ metrics, target = 0.92 }: Props) {
  const byIter = metrics.reduce<Record<number, Record<string, number>>>(
    (acc, m) => {
      if (!acc[m.iteration]) acc[m.iteration] = { iteration: m.iteration };
      acc[m.iteration][m.metric_name] = m.metric_value;
      return acc;
    },
    {}
  );
  const data = Object.values(byIter).sort((a, b) => a.iteration - b.iteration);
  const names = [...new Set(metrics.map((m) => m.metric_name))];

  if (!data.length)
    return (
      <div className="bg-[#1e293b] border border-[rgba(20,184,166,0.15)] rounded-lg p-5 h-56
                      flex flex-col items-center justify-center gap-2">
        <div className="w-8 h-px bg-[rgba(20,184,166,0.2)]" />
        <span className="text-[#64748b] text-xs tracking-widest">NO METRICS YET</span>
        <div className="w-8 h-px bg-[rgba(20,184,166,0.2)]" />
      </div>
    );

  return (
    <div className="bg-[#1e293b] border border-[rgba(20,184,166,0.15)] rounded-lg p-5">
      <div className="flex items-center justify-between mb-4">
        <span className="text-xs text-[#94a3b8] tracking-widest uppercase">Metric History</span>
        <div className="flex items-center gap-3">
          {names.map((name, i) => (
            <div key={name} className="flex items-center gap-1.5">
              <span
                className="inline-block w-3 h-px"
                style={{ background: COLORS[i % COLORS.length] }}
              />
              <span className="text-[10px] text-[#94a3b8]">{name}</span>
            </div>
          ))}
        </div>
      </div>

      <svg width="0" height="0" style={{ position: "absolute" }}>
        <defs>
          {names.map((name, i) => (
            <linearGradient key={name} id={`grad-${i}`} x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor={COLORS[i % COLORS.length]} stopOpacity={0.18} />
              <stop offset="100%" stopColor={COLORS[i % COLORS.length]} stopOpacity={0} />
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
            dataKey="iteration"
            tick={{ fontSize: 10, fill: "#334155" }}
            axisLine={false}
            tickLine={false}
          />
          <YAxis
            domain={[0, 1]}
            tickFormatter={(v: number) => `${(v * 100).toFixed(0)}%`}
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
            formatter={(v: unknown) => `${((v as number) * 100).toFixed(2)}%`}
            cursor={{ stroke: "rgba(20,184,166,0.2)", strokeWidth: 1 }}
          />
          <ReferenceLine
            y={target}
            stroke="rgba(20,184,166,0.35)"
            strokeDasharray="3 5"
            label={{
              value: `target ${(target * 100).toFixed(0)}%`,
              position: "right",
              fontSize: 9,
              fill: "rgba(20,184,166,0.5)",
            }}
          />
          {names.map((name, i) => (
            <Area
              key={name}
              type="monotone"
              dataKey={name}
              stroke={COLORS[i % COLORS.length]}
              strokeWidth={2}
              fill={`url(#grad-${i})`}
              dot={false}
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
