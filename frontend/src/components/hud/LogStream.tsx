"use client";

import { useEffect, useRef } from "react";
import type { TelemetryEvent } from "@/lib/api";

interface Props {
  events: TelemetryEvent[];
  connected: boolean;
  missionStatus?: string;
}

const LEVEL_STYLES: Record<string, { dot: string; text: string; label: string }> = {
  error:    { dot: "#f87171", text: "#f87171", label: "ERR" },
  warn:     { dot: "#fbbf24", text: "#fbbf24", label: "WRN" },
  info:     { dot: "#60a5fa", text: "#94a3b8", label: "INF" },
  success:  { dot: "#4ade80", text: "#86efac", label: "OK " },
  critique: { dot: "#a78bfa", text: "#c4b5fd", label: "CRT" },
  "":       { dot: "#334155", text: "#64748b", label: "   " },
};

function classify(type: string | undefined, name: string | undefined): string {
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

export function LogStream({ events, connected, missionStatus }: Props) {
  const bottomRef = useRef<HTMLDivElement>(null);
  const visible = events.filter((e) => e.type !== "backfill_complete");

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [visible.length]);

  return (
    <div
      className="rounded-lg flex flex-col"
      style={{
        background: "#1e293b",
        border: "1px solid rgba(20,184,166,0.12)",
        height: "24rem",
      }}
    >
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-2.5 border-b border-[rgba(20,184,166,0.08)] shrink-0">
        <div className="flex items-center gap-2">
          <span className="text-[10px] text-[#64748b] tracking-widest uppercase">
            event stream
          </span>
          {visible.length > 0 && (
            <span className="text-[10px] text-[#64748b]">({visible.length})</span>
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

      {/* Log rows */}
      <div className="flex-1 overflow-y-auto py-1">
        {visible.length === 0 ? (
          <div className="flex items-center gap-3 px-4 py-6 text-[#64748b]">
            <span className="text-[10px] tracking-widest">
              {STATUS_LABEL[missionStatus ?? ""] ?? "AWAITING EVENTS"}
            </span>
            <span className="inline-block w-1.5 h-3 bg-[#2d3f57] animate-pulse" />
          </div>
        ) : (
          visible.map((e, i) => {
            const level = classify(e.type, e.name);
            const style = LEVEL_STYLES[level];
            const ts = e.recorded_at ? new Date(e.recorded_at) : null;
            const msg = e.name != null
              ? e.value != null
                ? `${e.name}: ${e.value}${e.step != null ? ` (step ${e.step})` : ""}`
                : e.name
              : e.type;

            return (
              <div
                key={i}
                className="flex items-start gap-2 px-4 py-0.5 hover:bg-[rgba(255,255,255,0.02)] group"
              >
                <span
                  className="shrink-0 text-[10px] mt-0.5 w-6 text-right font-medium"
                  style={{ color: style.dot }}
                >
                  {style.label}
                </span>
                <span className="shrink-0 text-[10px] text-[#64748b] mt-0.5">
                  {ts ? ts.toLocaleTimeString("en", {
                    hour12: false,
                    hour: "2-digit",
                    minute: "2-digit",
                    second: "2-digit",
                  }) : "--:--:--"}
                </span>
                <span
                  className="text-[10px] text-[#94a3b8] mt-0.5 shrink-0 group-hover:text-[#64748b]"
                >
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
          })
        )}
        <div ref={bottomRef} />
      </div>
    </div>
  );
}
