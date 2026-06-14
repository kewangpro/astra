"use client";

import { useEffect, useRef } from "react";
import type { TelemetryEvent } from "@/lib/api";

interface Props {
  events: TelemetryEvent[];
  connected: boolean;
}

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
    <div className="bg-[#0d0d1a] border border-[rgba(20,184,166,0.15)] rounded-lg flex flex-col h-64">
      <div className="flex items-center justify-between px-4 py-2 border-b border-[rgba(20,184,166,0.1)]">
        <span className="text-xs text-[#475569] tracking-widest uppercase">Event Log</span>
        <span className={`text-[10px] ${connected ? "text-[#4ade80]" : "text-[#f87171]"}`}>
          {connected ? "● live" : "○ offline"}
        </span>
      </div>
      <div className="flex-1 overflow-y-auto px-4 py-2 space-y-0.5">
        {events.length === 0 ? (
          <div className="text-[#334155] text-xs pt-4">Waiting for events…</div>
        ) : (
          events.map((e, i) => (
            <div key={i} className={`log-entry ${classify(e.event)}`}>
              <span className="opacity-40">{new Date(e.ts).toLocaleTimeString()} </span>
              <span className="opacity-60">[{e.event}] </span>
              {e.data.message != null
                ? String(e.data.message)
                : JSON.stringify(e.data)}
            </div>
          ))
        )}
        <div ref={bottomRef} />
      </div>
    </div>
  );
}
