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

// Training-signal metric names — shown continuously in history, separate from
// the goal metric (shown in MetricGap instead). mean_reward for RL; loss for
// dpo/grpo (grpo_train.py prints it every --steps-per-report steps, dpo_train.py
// once per epoch — same relationship pass_rate has to mean_reward).
const TRAINING_SIGNAL_NAMES = ["mean_reward", "loss"];

export function MetricChart({ events, targetMetric }: Props) {
  // MetricHistory shows the training signal only.
  // For RL tasks the training signal is mean_reward; for dpo/grpo it's loss;
  // the goal metric (food_eaten, lines_cleared, pass_rate, etc.) is a separate
  // series shown in MetricGap — exclude it here. For ML/SFT tasks the goal
  // metric (accuracy, eval_loss) IS the training signal, so we keep everything.
  const goalMetricName = targetMetric && Object.keys(targetMetric).length > 0
    ? Object.keys(targetMetric)[0]
    : null;
  const trainingSignalEvent = events.find(
    (e) => (e.type === "metric" || e.type === "backfill") && TRAINING_SIGNAL_NAMES.includes(e.name ?? "")
  );
  const hasTrainingSignal = trainingSignalEvent != null;
  // Only exclude goal metric when a training signal is present and the goal
  // metric differs from it (i.e. it is a secondary eval metric, not the signal).
  const excludeGoalMetric =
    hasTrainingSignal && goalMetricName && goalMetricName !== trainingSignalEvent!.name;
  const metricEvents = events.filter(
    (e) =>
      (e.type === "metric" || e.type === "backfill") &&
      e.name != null &&
      e.value != null &&
      !(excludeGoalMetric && e.name === goalMetricName)
  );

  // Find all run-reset boundaries (step counter drops back to low value).
  const resetIndices: number[] = [0];
  for (let i = 1; i < metricEvents.length; i++) {
    const prev = metricEvents[i - 1].step ?? 0;
    const curr = metricEvents[i].step ?? 0;
    if (curr < prev) resetIndices.push(i);
  }
  const lastResetIdx = resetIndices[resetIndices.length - 1];

  // Only keep the last 3 runs so the current run isn't a tiny sliver.
  const MAX_RUNS = 3;
  const startIdx = resetIndices.length > MAX_RUNS
    ? resetIndices[resetIndices.length - MAX_RUNS]
    : 0;
  const visibleEvents = metricEvents.slice(startIdx);
  const visibleResetOffset = resetIndices.length > MAX_RUNS
    ? resetIndices.length - MAX_RUNS
    : 0;

  // Assign a unique x key per run to avoid step-number collisions across runs.
  let runOffset = 0;
  const chartEvents: Array<{ x: number; name: string; value: number; isLive: boolean }> = [];
  for (let i = 0; i < visibleEvents.length; i++) {
    const globalIdx = startIdx + i;
    const e = visibleEvents[i];
    if (i > 0) {
      const prev = visibleEvents[i - 1].step ?? 0;
      const curr = e.step ?? 0;
      if (curr < prev) runOffset += prev;
    }
    chartEvents.push({
      x: (e.step ?? 0) + runOffset,
      name: e.name!,
      value: e.value as number,
      isLive: globalIdx >= lastResetIdx,
    });
  }
  void visibleResetOffset;

  const byX = chartEvents.reduce<Record<number, Record<string, unknown>>>(
    (acc, e) => {
      if (!acc[e.x]) acc[e.x] = { step: e.x };
      const suffix = e.isLive ? "_live" : "_hist";
      acc[e.x][`${e.name}${suffix}`] = e.value;
      return acc;
    },
    {}
  );

  const data = Object.values(byX).sort(
    (a, b) => (a.step as number) - (b.step as number)
  );

  const hasLive = chartEvents.some((e) => e.isLive);

  const names = [...new Set(metricEvents.map((e) => e.name!))];

  // Resolve target value and display mode from targetMetric dict
  const [targetName, targetValue] =
    targetMetric && Object.keys(targetMetric).length > 0
      ? [Object.keys(targetMetric)[0], Object.values(targetMetric)[0] as number]
      : [null, 0.92];
  const isRaw = targetValue > 1;
  // Displaying a separate training signal (mean_reward, loss) rather than the
  // goal metric itself means the goal's own <=1 target scale is irrelevant to
  // this chart — e.g. pass_rate's target of 0.85 says nothing about the range
  // of "loss" values (which can exceed 1). Always treat as raw/adaptive in
  // that case, regardless of whether the (unshown) goal target is <=1.
  const displayRaw = excludeGoalMetric || isRaw;

  // Scale Y-axis from all live data so the chart is never blank when metric names
  // don't exactly match target_metric (e.g. target=lines_cleared but data=mean_reward),
  // and so it adapts correctly for signals unrelated to the goal's own scale (loss vs pass_rate).
  const maxObserved = Math.max(
    0,
    ...chartEvents.filter((e) => e.isLive).map((e) => e.value)
  );
  const yDomain: [number, number] = displayRaw
    ? [0, Math.max(excludeGoalMetric ? 1 : targetValue * 1.1, maxObserved * 1.1)]
    : [0, 1];

  const yFormatter = displayRaw
    ? (v: number) => v.toFixed(v < 10 ? 2 : 0)
    : (v: number) => `${(v * 100).toFixed(0)}%`;

  const tooltipFormatter = (v: unknown, name: unknown) => {
    const val = v as number;
    const label = String(name).replace(/_(live|hist)$/, "");
    const formatted = displayRaw ? val.toFixed(val < 10 ? 3 : 1) : `${(val * 100).toFixed(2)}%`;
    return [formatted, label] as [string, string];
  };

  const targetLabel = isRaw
    ? `target ${targetValue.toFixed(0)}`
    : `target ${(targetValue * 100).toFixed(0)}%`;

  if (!data.length) {
    // No pass_rate/loss data yet doesn't necessarily mean nothing is happening —
    // dpo's collect_pairs() phase can run 1hr+ before the first metric exists.
    // Surface the latest "Collecting preference pairs: ..." status event (same
    // event stream as "Sandbox running" etc.) instead of a bare placeholder.
    const collectingEvent = [...events]
      .reverse()
      .find((e) => e.type === "info" && (e.name ?? "").startsWith("Collecting preference pairs"));

    return (
      <div
        className="bg-[#1e293b] border border-[rgba(20,184,166,0.15)] rounded-lg p-5 h-56
                      flex flex-col items-center justify-center gap-2"
      >
        <div className="w-8 h-px bg-[rgba(20,184,166,0.2)]" />
        <span className="text-[#64748b] text-xs tracking-widest">
          {collectingEvent ? collectingEvent.name!.toUpperCase() : "NO METRICS YET"}
        </span>
        <div className="w-8 h-px bg-[rgba(20,184,166,0.2)]" />
      </div>
    );
  }

  return (
    <div className="bg-[#1e293b] border border-[rgba(20,184,166,0.15)] rounded-lg p-5">
      <div className="flex items-center justify-between mb-4">
        <span className="text-xs text-[#94a3b8] tracking-widest uppercase">
          Metric History
        </span>
        <div className="flex items-center gap-4">
          {hasLive && (
            <div className="flex items-center gap-1.5">
              <span className="inline-block w-3 h-px" style={{ background: COLORS[0] }} />
              <span className="text-[10px] text-[#94a3b8]">current</span>
            </div>
          )}
          <div className="flex items-center gap-1.5">
            <span className="inline-block w-3 h-px" style={{ background: COLORS[0], opacity: 0.35 }} />
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
              <stop offset="0%" stopColor={COLORS[i % COLORS.length]} stopOpacity={0.07} />
              <stop offset="100%" stopColor={COLORS[i % COLORS.length]} stopOpacity={0} />
            </linearGradient>
          ))}
        </defs>
      </svg>

      <ResponsiveContainer width="100%" height={200}>
        <AreaChart data={data} margin={{ top: 4, right: 10, bottom: 0, left: -16 }}>
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
            tickCount={6}
            tickFormatter={(v: number) =>
              v >= 1_000_000
                ? `${(v / 1_000_000).toFixed(1)}M`
                : v >= 1_000
                ? `${Math.round(v / 1_000)}K`
                : String(v)
            }
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
          {/* Only draw the target line when the chart is showing the goal metric
              directly (mean_reward missions). When the chart shows mean_reward as a
              proxy for a different goal (food_eaten, lines_cleared, etc.) the target
              value is on a different scale and the line is misleading. */}
          {!excludeGoalMetric && (
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
          )}
          {/* Historical series — same color as current but dimmed */}
          {names.map((name, i) => (
            <Area
              key={`${name}_hist`}
              type="monotone"
              dataKey={`${name}_hist`}
              stroke={COLORS[i % COLORS.length]}
              strokeWidth={1.5}
              strokeOpacity={0.35}
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
