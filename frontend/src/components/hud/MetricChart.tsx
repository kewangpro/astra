"use client";

import {
  ResponsiveContainer,
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  ReferenceLine,
} from "recharts";
import type { Metric } from "@/lib/api";

interface Props {
  metrics: Metric[];
  target?: number;
}

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

  const COLORS = ["#14b8a6", "#60a5fa", "#a78bfa", "#fbbf24", "#4ade80"];

  if (!data.length)
    return (
      <div className="bg-[#0d0d1a] border border-[rgba(20,184,166,0.15)] rounded-lg p-5 h-56
                      flex items-center justify-center text-[#334155] text-sm">
        No metrics yet…
      </div>
    );

  return (
    <div className="bg-[#0d0d1a] border border-[rgba(20,184,166,0.15)] rounded-lg p-5">
      <div className="text-xs text-[#475569] tracking-widest uppercase mb-4">
        Metric History
      </div>
      <ResponsiveContainer width="100%" height={200}>
        <LineChart data={data} margin={{ top: 4, right: 8, bottom: 0, left: -16 }}>
          <XAxis
            dataKey="iteration"
            tick={{ fontSize: 10, fill: "#475569" }}
            axisLine={false}
            tickLine={false}
          />
          <YAxis
            domain={[0, 1]}
            tickFormatter={(v: number) => `${(v * 100).toFixed(0)}%`}
            tick={{ fontSize: 10, fill: "#475569" }}
            axisLine={false}
            tickLine={false}
          />
          <Tooltip
            contentStyle={{
              background: "#12122a",
              border: "1px solid rgba(20,184,166,0.2)",
              borderRadius: 6,
              fontSize: 11,
              color: "#e2e8f0",
            }}
            formatter={(v: number) => `${(v * 100).toFixed(2)}%`}
          />
          <ReferenceLine
            y={target}
            stroke="rgba(20,184,166,0.4)"
            strokeDasharray="4 4"
          />
          {names.map((name, i) => (
            <Line
              key={name}
              type="monotone"
              dataKey={name}
              stroke={COLORS[i % COLORS.length]}
              strokeWidth={2}
              dot={false}
              activeDot={{ r: 4 }}
            />
          ))}
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
