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
          // Only trim high-frequency mean_reward metrics; always preserve
          // goal-metric events (food_eaten, lines_cleared) and all non-metric
          // events (info, pivot, status, critique) so the event stream log
          // never goes blank after a long run.
          const keep = next.filter(
            (ev) => !(ev.type === "metric" && ev.name === "mean_reward")
          );
          const trimmable = next.filter(
            (ev) => ev.type === "metric" && ev.name === "mean_reward"
          );
          return [...keep, ...trimmable.slice(-Math.max(0, 500 - keep.length))];
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
