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

// Training-signal metric names — shown continuously in history.
// mean_reward for RL; loss for dpo/grpo/distill (per-step on Mini stdout).
const TRAINING_SIGNAL_NAMES = ["mean_reward", "loss"];

function isFinetuneRateGoal(name: string | null): boolean {
  if (!name) return false;
  const n = name.toLowerCase();
  return n === "pass_rate" || n.endsWith("_rate");
}

export function MetricChart({ events, targetMetric }: Props) {
  // MetricHistory shows the training signal; for remote LM fine-tunes also
  // chart mid-run pass_rate evals alongside loss (Metric Gap stays the
  // iteration-level DB arc). RL still excludes food_eaten/lines_cleared when
  // mean_reward is the training signal (different scale / role).
  const goalMetricName = targetMetric && Object.keys(targetMetric).length > 0
    ? Object.keys(targetMetric)[0]
    : null;
  const trainingSignalEvent = events.find(
    (e) => (e.type === "metric" || e.type === "backfill") && TRAINING_SIGNAL_NAMES.includes(e.name ?? "")
  );
  const hasTrainingSignal = trainingSignalEvent != null;
  const showGoalWithLoss =
    hasTrainingSignal &&
    trainingSignalEvent!.name === "loss" &&
    isFinetuneRateGoal(goalMetricName);
  // Exclude goal when it differs from the training signal — unless we are
  // deliberately dual-charting loss + pass_rate for fine-tunes.
  const excludeGoalMetric =
    hasTrainingSignal &&
    !!goalMetricName &&
    goalMetricName !== trainingSignalEvent!.name &&
    !showGoalWithLoss;
  // End-of-iter summaries historically used step = iteration * 500000 (RL
  // sparkline ordering). On dual loss+pass_rate charts those become ghost
  // points at ~0.5M–3.5M with a couple of pass_rate dots. Drop them from the
  // history chart; MetricGap still buckets by iteration.
  const SYNTHETIC_ITER_STEP = 500_000;
  const metricEvents = events.filter((e) => {
    if (
      (e.type !== "metric" && e.type !== "backfill") ||
      e.name == null ||
      e.value == null ||
      (excludeGoalMetric && e.name === goalMetricName)
    ) {
      return false;
    }
    // Dual chart: only loss + goal rate (drop avg_loss / split rates).
    if (
      showGoalWithLoss &&
      e.name !== "loss" &&
      e.name !== goalMetricName
    ) {
      return false;
    }
    const step = e.step ?? 0;
    if (
      showGoalWithLoss &&
      e.name === goalMetricName &&
      step >= SYNTHETIC_ITER_STEP
    ) {
      return false;
    }
    return true;
  });

  // Find all run-reset boundaries — prefer the `iteration` field (present on
  // every backend-emitted metric event) over the old step-decrease heuristic.
  // DPO/GRPO loss steps can reset or collide across iterations; `iteration`
  // differing is the reliable run boundary.
  const resetIndices: number[] = [0];
  for (let i = 1; i < metricEvents.length; i++) {
    const prevEvent = metricEvents[i - 1];
    const currEvent = metricEvents[i];
    const boundary =
      prevEvent.iteration != null && currEvent.iteration != null
        ? currEvent.iteration !== prevEvent.iteration
        : (currEvent.step ?? 0) < (prevEvent.step ?? 0);
    if (boundary) resetIndices.push(i);
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
  // Advance by the max step *in the completed run*, not the previous event's
  // step — the last event is often a pass_rate with step 0..3, which used to
  // leave runs almost fully overlapped (two/three loss curves at x=30, 42, …).
  let runOffset = 0;
  let maxStepInRun = 0;
  const chartEvents: Array<{ x: number; name: string; value: number; isLive: boolean }> = [];
  for (let i = 0; i < visibleEvents.length; i++) {
    const globalIdx = startIdx + i;
    const e = visibleEvents[i];
    const step = e.step ?? 0;
    if (i > 0) {
      const prevEvent = visibleEvents[i - 1];
      const prev = prevEvent.step ?? 0;
      const boundary =
        prevEvent.iteration != null && e.iteration != null
          ? e.iteration !== prevEvent.iteration
          : step < prev;
      if (boundary) {
        runOffset += maxStepInRun;
        maxStepInRun = 0;
      }
    }
    maxStepInRun = Math.max(maxStepInRun, step);
    chartEvents.push({
      x: step + runOffset,
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
  // of "loss" values (which can exceed 1). Dual loss+pass_rate charts must
  // also stay raw: labeling loss 0.70 as "70%" is wrong even though both
  // series often sit in [0, 1].
  const displayRaw = excludeGoalMetric || showGoalWithLoss || isRaw;

  // Scale Y-axis from all live data so the chart is never blank when metric names
  // don't exactly match target_metric (e.g. target=lines_cleared but data=mean_reward),
  // and so it adapts correctly for signals unrelated to the goal's own scale (loss vs pass_rate).
  const maxObserved = Math.max(
    0,
    ...chartEvents.filter((e) => e.isLive).map((e) => e.value)
  );
  const yDomain: [number, number] = displayRaw
    ? [
        0,
        Math.max(
          showGoalWithLoss || excludeGoalMetric ? 1 : targetValue * 1.1,
          maxObserved * 1.1,
        ),
      ]
    : [0, 1];

  const yFormatter = displayRaw
    ? (v: number) => v.toFixed(v < 10 ? 2 : 0)
    : (v: number) => `${(v * 100).toFixed(0)}%`;

  const tooltipFormatter = (v: unknown, name: unknown) => {
    const val = v as number;
    const label = String(name).replace(/_(live|hist)$/, "");
    // Dual chart: format pass_rate as %; loss stays decimal.
    const asRate = showGoalWithLoss && isFinetuneRateGoal(label);
    const formatted = asRate
      ? `${(val * 100).toFixed(2)}%`
      : displayRaw
        ? val.toFixed(val < 10 ? 3 : 1)
        : `${(val * 100).toFixed(2)}%`;
    return [formatted, label] as [string, string];
  };

  const targetLabel = isRaw
    ? `target ${targetValue.toFixed(0)}`
    : `target ${(targetValue * 100).toFixed(0)}%`;

  const progressStatus = [...events]
    .reverse()
    .find(
      (e) =>
        (e.type === "info" || e.type === "success") &&
        typeof e.name === "string" &&
        (e.name.startsWith("Collecting preference pairs") ||
          e.name.startsWith("Training: step") ||
          e.name.startsWith("Rejection sampling") ||
          e.name.startsWith("On-policy sampling") ||
          e.name.startsWith("New best pass rate")),
    );

  if (!data.length) {
    // No pass_rate/loss yet — collection or early train can run a long time.
    // Surface the latest progress status instead of a bare placeholder.
    return (
      <div
        className="bg-[#1e293b] border border-[rgba(20,184,166,0.15)] rounded-lg p-5 h-56
                      flex flex-col items-center justify-center gap-2"
      >
        <div className="w-8 h-px bg-[rgba(20,184,166,0.2)]" />
        <span className="text-[#64748b] text-xs tracking-widest text-center px-4">
          {progressStatus?.name
            ? progressStatus.name.toUpperCase()
            : "NO METRICS YET"}
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
          {names.map((name, i) => (
            <div key={name} className="flex items-center gap-1.5">
              <span
                className="inline-block w-3 h-px"
                style={{ background: COLORS[i % COLORS.length] }}
              />
              <span className="text-[10px] text-[#94a3b8]">{name}</span>
            </div>
          ))}
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
