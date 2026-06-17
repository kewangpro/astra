"use client";

import { useEffect, useRef, useState, useCallback } from "react";

const GRID = 16;
const CELL = 20; // px per cell
const CANVAS_SIZE = GRID * CELL;

const WS_BASE =
  typeof window !== "undefined"
    ? `ws://${window.location.hostname}:8200`
    : "ws://localhost:8200";

interface Frame {
  type: "frame" | "episode_end" | "error";
  grid?: number[];
  episode?: number;
  step?: number;
  episode_reward?: number;
  total_reward?: number;
  done?: boolean;
  message?: string;
}

function drawFrame(ctx: CanvasRenderingContext2D, grid: number[]) {
  ctx.fillStyle = "#0f172a";
  ctx.fillRect(0, 0, CANVAS_SIZE, CANVAS_SIZE);

  for (let i = 0; i < GRID * GRID; i++) {
    const row = Math.floor(i / GRID);
    const col = i % GRID;
    const val = grid[i];
    const x = col * CELL;
    const y = row * CELL;

    if (val === 1.0) {
      // Snake head
      ctx.fillStyle = "#14b8a6";
      ctx.fillRect(x + 1, y + 1, CELL - 2, CELL - 2);
    } else if (val === 0.5) {
      // Snake body
      ctx.fillStyle = "#0d6b61";
      ctx.fillRect(x + 2, y + 2, CELL - 4, CELL - 4);
    } else if (val === -1.0) {
      // Food
      ctx.fillStyle = "#f87171";
      const cx = x + CELL / 2;
      const cy = y + CELL / 2;
      ctx.beginPath();
      ctx.arc(cx, cy, CELL / 2 - 3, 0, Math.PI * 2);
      ctx.fill();
    }
  }

  // Grid lines (subtle)
  ctx.strokeStyle = "rgba(20,184,166,0.04)";
  ctx.lineWidth = 0.5;
  for (let i = 0; i <= GRID; i++) {
    ctx.beginPath(); ctx.moveTo(i * CELL, 0); ctx.lineTo(i * CELL, CANVAS_SIZE); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(0, i * CELL); ctx.lineTo(CANVAS_SIZE, i * CELL); ctx.stroke();
  }
}

interface Props {
  missionId: string;
  envId?: string;
}

export function SnakePlayer({ missionId, envId = "Snake-v0" }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const wsRef = useRef<WebSocket | null>(null);
  const [playing, setPlaying] = useState(false);
  const [episode, setEpisode] = useState(0);
  const [episodeReward, setEpisodeReward] = useState(0);
  const [bestReward, setBestReward] = useState<number | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  const stop = useCallback(() => {
    wsRef.current?.close();
    wsRef.current = null;
    setPlaying(false);
    setLoading(false);
  }, []);

  const start = useCallback(() => {
    if (wsRef.current) return;
    setError(null);
    setLoading(true);

    const ws = new WebSocket(
      `${WS_BASE}/ws/missions/${missionId}/play?env_id=${envId}&fps=12`
    );
    wsRef.current = ws;

    ws.onopen = () => {
      setPlaying(true);
      setLoading(false);
    };

    ws.onmessage = (e) => {
      const frame: Frame = JSON.parse(e.data as string);
      if (frame.type === "frame" && frame.grid) {
        const ctx = canvasRef.current?.getContext("2d");
        if (ctx) drawFrame(ctx, frame.grid);
        setEpisodeReward(frame.episode_reward ?? 0);
        if (frame.episode) setEpisode(frame.episode);
      } else if (frame.type === "episode_end") {
        const r = frame.total_reward ?? 0;
        setBestReward((prev) => (prev === null || r > prev ? r : prev));
      } else if (frame.type === "error") {
        setError(frame.message ?? "Unknown error");
        stop();
      }
    };

    ws.onerror = () => {
      setError("Connection failed");
      stop();
    };

    ws.onclose = () => {
      setPlaying(false);
      setLoading(false);
      wsRef.current = null;
    };
  }, [missionId, envId, stop]);

  // Cleanup on unmount
  useEffect(() => () => { wsRef.current?.close(); }, []);

  return (
    <div
      className="rounded-lg p-4 space-y-3"
      style={{ background: "#1e293b", border: "1px solid rgba(20,184,166,0.15)" }}
    >
      {/* Header */}
      <div className="flex items-center justify-between">
        <span className="text-[10px] text-[#64748b] tracking-widest uppercase">
          agent.play — {envId}
        </span>
        <div className="flex items-center gap-3">
          {bestReward !== null && (
            <span className="text-[10px] text-[#14b8a6]">
              best {bestReward.toFixed(1)}
            </span>
          )}
          <button
            onClick={playing ? stop : start}
            disabled={loading}
            className="text-[11px] px-3 py-1 rounded border transition-colors disabled:opacity-40"
            style={{
              color: playing ? "#f87171" : "#14b8a6",
              borderColor: playing ? "rgba(248,113,113,0.4)" : "rgba(20,184,166,0.4)",
              background: playing ? "rgba(248,113,113,0.08)" : "rgba(20,184,166,0.08)",
            }}
          >
            {loading ? "loading…" : playing ? "■ stop" : "▶ watch"}
          </button>
        </div>
      </div>

      {/* Canvas */}
      <div className="flex justify-center">
        <canvas
          ref={canvasRef}
          width={CANVAS_SIZE}
          height={CANVAS_SIZE}
          style={{
            imageRendering: "pixelated",
            borderRadius: 4,
            background: "#0f172a",
            border: "1px solid rgba(20,184,166,0.08)",
          }}
        />
      </div>

      {/* Stats */}
      {playing && (
        <div className="flex justify-between text-[10px] text-[#64748b]">
          <span>episode {episode}</span>
          <span>reward {episodeReward.toFixed(1)}</span>
        </div>
      )}

      {error && (
        <p className="text-[10px] text-[#f87171]">{error}</p>
      )}
    </div>
  );
}
