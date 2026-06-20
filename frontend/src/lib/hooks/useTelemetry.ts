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
        const evt: TelemetryEvent = JSON.parse(e.data as string);
        setEvents((prev) => {
          const next = [...prev, evt];
          if (next.length <= 500) return next;
          // Always preserve goal metric events (non-mean_reward, one per iteration).
          // Only trim high-frequency events (mean_reward, status, info).
          const keep = next.filter(
            (ev) => ev.type === "metric" && ev.name !== "mean_reward"
          );
          const trimmable = next.filter(
            (ev) => !(ev.type === "metric" && ev.name !== "mean_reward")
          );
          return [...keep, ...trimmable.slice(-(500 - keep.length))];
        });
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
