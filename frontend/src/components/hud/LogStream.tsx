"use client";

import { useEffect, useRef } from "react";
import type { TelemetryEvent } from "@/lib/api";

interface Props {
  events: TelemetryEvent[];
  connected: boolean;
}

const LEVEL_STYLES: Record<string, { dot: string; text: string; label: string }> = {
  error:   { dot: "#f87171", text: "#f87171", label: "ERR" },
  warn:    { dot: "#fbbf24", text: "#fbbf24", label: "WRN" },
  info:    { dot: "#60a5fa", text: "#94a3b8", label: "INF" },
  success: { dot: "#4ade80", text: "#86efac", label: "OK " },
  "":      { dot: "#334155", text: "#64748b", label: "   " },
};

function classify(event: string): string {
  if (event.includes("error") || event.includes("failed")) return "error";
  if (event.includes("warn")) return "warn";
  if (event.includes("complete") || event.includes("success") || event.includes("metric"))
    return "success";
  if (event.includes("start") || event.includes("launch") || event.includes("loop"))
    return "info";
  return "";
}

export function LogStream({ events, connected }: Props) {
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [events.length]);

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
          {events.length > 0 && (
            <span className="text-[10px] text-[#64748b]">({events.length})</span>
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
        {events.length === 0 ? (
          <div className="flex items-center gap-3 px-4 py-6 text-[#64748b]">
            <span className="text-[10px] tracking-widest">AWAITING EVENTS</span>
            <span className="inline-block w-1.5 h-3 bg-[#2d3f57] animate-pulse" />
          </div>
        ) : (
          events.map((e, i) => {
            const level = classify(e.event);
            const style = LEVEL_STYLES[level];
            const msg =
              e.data.message != null ? String(e.data.message) : JSON.stringify(e.data);

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
                  {new Date(e.ts).toLocaleTimeString("en", {
                    hour12: false,
                    hour: "2-digit",
                    minute: "2-digit",
                    second: "2-digit",
                  })}
                </span>
                <span
                  className="text-[10px] text-[#94a3b8] mt-0.5 shrink-0 group-hover:text-[#64748b]"
                >
                  [{e.event}]
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
