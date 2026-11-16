"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import type { TelemetryEvent } from "@/lib/api";
import { parseTs } from "@/lib/date";

interface Props {
  events: TelemetryEvent[];
  connected: boolean;
  missionStatus?: string;
  className?: string;
}

const LEVEL_STYLES: Record<string, { dot: string; text: string; label: string }> = {
  error:    { dot: "#f87171", text: "#f87171", label: "ERR" },
  warn:     { dot: "#fbbf24", text: "#fbbf24", label: "WRN" },
  info:     { dot: "#60a5fa", text: "#94a3b8", label: "INF" },
  success:  { dot: "#4ade80", text: "#86efac", label: "OK " },
  critique: { dot: "#a78bfa", text: "#c4b5fd", label: "CRT" },
  log:      { dot: "#64748b", text: "#94a3b8", label: "LOG" },
  "":       { dot: "#334155", text: "#64748b", label: "   " },
};

/** High-frequency progress in mirrored sandbox.log — collapse consecutive runs. */
const PROGRESS_LINE_RE = /^\s*\[\d+\/\d+\]|^Step\s+\d+\//;

function classify(type: string | undefined, name: string | undefined): string {
  if (type === "log") return "log";
  if (type === "error") return "error";
  if (type === "warn") return "warn";
  if (type === "success") return "success";
  if (type === "pivot") return "warn";
  if (type === "metric") return "success";
  if (type === "critique") return "critique";
  const s = `${name ?? ""}`;
  if (s.includes("error") || s.includes("failed") || s.includes("rejected")) return "error";
  if (s.includes("warn") || s.includes("approval") || s.includes("healing")) return "warn";
  if (s.includes("ready") || s.includes("complete") || s.includes("approved") || s.includes("achieved")) return "success";
  return "info";
}

const STATUS_LABEL: Record<string, string> = {
  planning:   "LLM generating plan…",
  running:    "AWAITING EVENTS",
  evaluating: "Evaluating results…",
  paused:     "PAUSED",
  pending:    "NOT STARTED",
  failed:     "MISSION FAILED",
  completed:  "MISSION COMPLETE",
};

/** High-frequency RL signals stay on Metric History; sparse fine-tune metrics
 *  still appear in the stream alongside mirrored sandbox.log lines. */
const STREAM_METRIC_NAMES = new Set([
  "pass_rate",
  "pass_rate_static_live",
  "pass_rate_heldout",
  "loss",
  "avg_loss",
  "train_loss",
  "eval_loss",
]);

function isStreamVisible(e: TelemetryEvent): boolean {
  if (e.type === "backfill_complete" || e.type === "backfill") return false;
  if (e.type === "metric") {
    return !!e.name && STREAM_METRIC_NAMES.has(e.name);
  }
  return true;
}

function formatStreamMsg(e: TelemetryEvent): string {
  if (e.type === "metric" && e.name != null && e.value != null) {
    const n = e.name;
    const v = e.value as number;
    const step = e.step != null ? ` (step ${e.step})` : "";
    if (
      n === "pass_rate" ||
      n === "pass_rate_static_live" ||
      n === "pass_rate_heldout" ||
      n.endsWith("_rate")
    ) {
      return `${n}: ${(v * 100).toFixed(1)}%${step}`;
    }
    return `${n}: ${typeof v === "number" ? v.toFixed(v < 10 ? 4 : 1) : v}${step}`;
  }
  if (e.name != null) {
    return e.value != null
      ? `${e.name}: ${e.value}${e.step != null ? ` (step ${e.step})` : ""}`
      : e.name;
  }
  return e.type ?? "";
}

function isProgressLog(e: TelemetryEvent): boolean {
  return e.type === "log" && PROGRESS_LINE_RE.test(e.name ?? "");
}

type CollapsedRow =
  | { kind: "single"; event: TelemetryEvent; key: number }
  | { kind: "group"; events: TelemetryEvent[]; key: number };

function collapseProgressRows(events: TelemetryEvent[]): CollapsedRow[] {
  const rows: CollapsedRow[] = [];
  let i = 0;
  while (i < events.length) {
    const e = events[i];
    if (!isProgressLog(e)) {
      rows.push({ kind: "single", event: e, key: i });
      i += 1;
      continue;
    }
    const group = [e];
    let j = i + 1;
    while (j < events.length && isProgressLog(events[j])) {
      group.push(events[j]);
      j += 1;
    }
    if (group.length === 1) {
      rows.push({ kind: "single", event: e, key: i });
    } else {
      rows.push({ kind: "group", events: group, key: i });
    }
    i = j;
  }
  return rows;
}

function EventRow({ e }: { e: TelemetryEvent }) {
  const level = classify(e.type, e.name);
  const style = LEVEL_STYLES[level];
  const ts = e.recorded_at ? parseTs(e.recorded_at) : null;
  const msg = formatStreamMsg(e);

  return (
    <div className="flex items-start gap-2 px-4 py-0.5 hover:bg-[rgba(255,255,255,0.02)] group">
      <span
        className="shrink-0 text-[10px] mt-0.5 w-6 text-right font-medium"
        style={{ color: style.dot }}
      >
        {style.label}
      </span>
      <span className="shrink-0 text-[10px] text-[#64748b] mt-0.5">
        {ts
          ? ts.toLocaleTimeString("en", {
              hour12: false,
              hour: "2-digit",
              minute: "2-digit",
              second: "2-digit",
            })
          : "--:--:--"}
      </span>
      <span className="text-[10px] text-[#94a3b8] mt-0.5 shrink-0 group-hover:text-[#64748b]">
        [{e.type}]
      </span>
      <span
        className="text-[11px] leading-relaxed break-all"
        style={{ color: style.text }}
      >
        {msg}
      </span>
    </div>
  );
}

function ProgressGroup({ events, rowKey }: { events: TelemetryEvent[]; rowKey: number }) {
  const [expanded, setExpanded] = useState(false);
  const latest = events[events.length - 1];
  const hidden = events.length - 1;

  if (expanded) {
    return (
      <div>
        {events.map((e, i) => (
          <EventRow key={`${rowKey}-${i}`} e={e} />
        ))}
        <button
          type="button"
          className="px-4 py-0.5 text-[10px] text-[#64748b] hover:text-[#94a3b8] tracking-wide"
          onClick={() => setExpanded(false)}
        >
          collapse {events.length} progress lines
        </button>
      </div>
    );
  }

  return (
    <div>
      <EventRow e={latest} />
      <button
        type="button"
        className="px-4 py-0.5 text-[10px] text-[#64748b] hover:text-[#94a3b8] tracking-wide"
        onClick={() => setExpanded(true)}
      >
        +{hidden} similar
      </button>
    </div>
  );
}

export function LogStream({ events, connected, missionStatus, className }: Props) {
  const bottomRef = useRef<HTMLDivElement>(null);
  const allVisible = events.filter(isStreamVisible);
  const visible = allVisible.slice(-100);
  const rows = useMemo(() => collapseProgressRows(visible), [visible]);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [visible.length]);

  return (
    <div
      className={`rounded-lg flex flex-col ${className || "h-96"}`}
      style={{
        background: "#1e293b",
        border: "1px solid rgba(20,184,166,0.12)",
      }}
    >
      <div className="flex items-center justify-between px-4 py-2.5 border-b border-[rgba(20,184,166,0.08)] shrink-0">
        <div className="flex items-center gap-2">
          <span className="text-[10px] text-[#64748b] tracking-widest uppercase">
            event stream
          </span>
          {allVisible.length > 0 && (
            <span className="text-[10px] text-[#64748b]">
              {allVisible.length > 100 ? "100+" : allVisible.length}
            </span>
          )}
        </div>
        <div className="flex items-center gap-1.5">
          <span
            className="inline-block w-1.5 h-1.5 rounded-full"
            style={{
              background: connected ? "#4ade80" : "#f87171",
              boxShadow: connected ? "0 0 6px #4ade80" : "none",
            }}
          />
          <span
            className="text-[10px] tracking-widest"
            style={{ color: connected ? "#4ade80" : "#f87171" }}
          >
            {connected ? "live" : "offline"}
          </span>
        </div>
      </div>

      <div className="flex-1 overflow-y-auto py-1">
        {visible.length === 0 ? (
          <div className="flex items-center gap-3 px-4 py-6 text-[#64748b]">
            <span className="text-[10px] tracking-widest">
              {STATUS_LABEL[missionStatus ?? ""] ?? "AWAITING EVENTS"}
            </span>
            <span className="inline-block w-1.5 h-3 bg-[#2d3f57] animate-pulse" />
          </div>
        ) : (
          rows.map((row) =>
            row.kind === "single" ? (
              <EventRow key={row.key} e={row.event} />
            ) : (
              <ProgressGroup key={row.key} events={row.events} rowKey={row.key} />
            ),
          )
        )}
        <div ref={bottomRef} />
      </div>
    </div>
  );
}
