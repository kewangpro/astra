import { useState, useEffect, useRef } from "react";
import type { TelemetryEvent } from "@/lib/api";

const WS_BASE =
  typeof window !== "undefined"
    ? `ws://${window.location.hostname}:8200`
    : "ws://localhost:8200";

export function useTelemetry(missionId: string) {
  const [events, setEvents] = useState<TelemetryEvent[]>([]);
  const [connected, setConnected] = useState(false);
  const wsRef = useRef<WebSocket | null>(null);

  useEffect(() => {
    if (!missionId) return;

    const ws = new WebSocket(`${WS_BASE}/ws/missions/${missionId}/telemetry`);
    wsRef.current = ws;

    ws.onopen = () => setConnected(true);
    ws.onclose = () => setConnected(false);
    ws.onerror = () => setConnected(false);

    ws.onmessage = (e) => {
      try {
        const msg = JSON.parse(e.data as string);
        if (msg.type === "backfill_batch") {
          // All historical events in one shot — single state update.
          setEvents(msg.events as TelemetryEvent[]);
        } else if (msg.type !== "backfill_complete") {
          setEvents((prev) => [...prev, msg as TelemetryEvent]);
        }
      } catch {
        // ignore malformed frames
      }
    };

    return () => {
      ws.close();
    };
  }, [missionId]);

  return { events, connected };
}
