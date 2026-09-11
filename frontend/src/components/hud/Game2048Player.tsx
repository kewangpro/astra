"use client";

import { useEffect, useRef, useState, useCallback } from "react";

const SIZE = 4;
const CANVAS_SIZE = 320;
const PADDING = 10;
const GAP = 10;
const TILE_SIZE = (CANVAS_SIZE - PADDING * 2 - GAP * (SIZE - 1)) / SIZE;

const WS_BASE =
  typeof window !== "undefined"
    ? `ws://${window.location.hostname}:8200`
    : "ws://localhost:8200";

const TILE_COLORS: Record<number, { bg: string; text: string }> = {
  0:    { bg: "rgba(51, 65, 85, 0.35)", text: "transparent" },
  2:    { bg: "#334155", text: "#f1f5f9" },
  4:    { bg: "#475569", text: "#f1f5f9" },
  8:    { bg: "#d97706", text: "#ffffff" },
  16:   { bg: "#ea580c", text: "#ffffff" },
  32:   { bg: "#dc2626", text: "#ffffff" },
  64:   { bg: "#e11d48", text: "#ffffff" },
  128:  { bg: "#0284c7", text: "#ffffff" },
  256:  { bg: "#2563eb", text: "#ffffff" },
  512:  { bg: "#7c3aed", text: "#ffffff" },
  1024: { bg: "#059669", text: "#ffffff" },
  2048: { bg: "#eab308", text: "#ffffff" },
  4096: { bg: "#f59e0b", text: "#ffffff" },
};

interface Frame {
  type: "frame" | "episode_end" | "error";
  grid?: number[];
  episode?: number;
  step?: number;
  episode_reward?: number;
  total_reward?: number;
  score?: number;
  max_tile?: number;
  done?: boolean;
  message?: string;
}

function drawBoard(ctx: CanvasRenderingContext2D, grid: number[]) {
  ctx.fillStyle = "#0f172a";
  ctx.fillRect(0, 0, CANVAS_SIZE, CANVAS_SIZE);

  // Background container with rounded corners
  ctx.fillStyle = "#1e293b";
  ctx.beginPath();
  ctx.roundRect(0, 0, CANVAS_SIZE, CANVAS_SIZE, 8);
  ctx.fill();

  for (let r = 0; r < SIZE; r++) {
    for (let c = 0; c < SIZE; c++) {
      const val = grid[r * SIZE + c] || 0;
      const x = PADDING + c * (TILE_SIZE + GAP);
      const y = PADDING + r * (TILE_SIZE + GAP);
      const style = TILE_COLORS[val] ?? { bg: "#b45309", text: "#ffffff" };

      // Draw tile background
      ctx.fillStyle = style.bg;
      ctx.beginPath();
      ctx.roundRect(x, y, TILE_SIZE, TILE_SIZE, 6);
      ctx.fill();

      // Draw value if non-zero
      if (val > 0) {
        ctx.fillStyle = style.text;
        ctx.textAlign = "center";
        ctx.textBaseline = "middle";

        let fontSize = 24;
        if (val >= 1024) fontSize = 16;
        else if (val >= 128) fontSize = 20;

        ctx.font = `bold ${fontSize}px ui-sans-serif, system-ui, sans-serif`;
        ctx.fillText(val.toString(), x + TILE_SIZE / 2, y + TILE_SIZE / 2 + 1);
      }
    }
  }
}

interface Props {
  missionId: string;
  envId?: string;
}

export function Game2048Player({ missionId, envId = "Game2048-v0" }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const wsRef = useRef<WebSocket | null>(null);

  const [playing, setPlaying] = useState(false);
  const [episode, setEpisode] = useState(0);
  const [score, setScore] = useState(0);
  const [maxTile, setMaxTile] = useState(0);
  const [bestScore, setBestScore] = useState<number | null>(null);
  const [step, setStep] = useState(0);
  const [speed, setSpeed] = useState(8);
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
      `${WS_BASE}/ws/missions/${missionId}/play?env_id=${envId}&fps=${speed}`
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
        if (ctx) drawBoard(ctx, frame.grid);

        if (frame.step !== undefined) setStep(frame.step);
        if (frame.score !== undefined) setScore(frame.score);
        if (frame.max_tile !== undefined) {
          setMaxTile(frame.max_tile);
          setBestScore((prev) => (prev === null ? frame.score! : Math.max(prev, frame.score!)));
        }
        if (frame.episode) setEpisode(frame.episode);
      } else if (frame.type === "episode_end") {
        if (frame.score !== undefined) {
          setBestScore((prev) => (prev === null ? frame.score! : Math.max(prev, frame.score!)));
        }
      } else if (frame.type === "error") {
        setError(frame.message ?? "Inference error");
        stop();
      }
    };

    ws.onerror = () => {
      setError("WebSocket connection failed");
      stop();
    };

    ws.onclose = () => {
      setPlaying(false);
      setLoading(false);
      wsRef.current = null;
    };
  }, [missionId, envId, speed, stop]);

  // Initial draw
  useEffect(() => {
    const ctx = canvasRef.current?.getContext("2d");
    if (ctx) drawBoard(ctx, new Array(16).fill(0));
  }, []);

  useEffect(() => () => { wsRef.current?.close(); }, []);

  return (
    <div className="bg-[#1e293b] border border-[rgba(20,184,166,0.15)] rounded-lg p-5 space-y-4">
      {/* Header */}
      <div className="flex flex-wrap items-center justify-between gap-3 border-b border-[rgba(255,255,255,0.05)] pb-3">
        <div>
          <div className="flex items-center gap-2">
            <span className="text-[10px] font-mono tracking-widest text-[#64748b] uppercase">
              AGENT.PLAY
            </span>
            <span className="text-[10px] font-mono px-1.5 py-0.5 rounded bg-[#0f172a] text-[#14b8a6] border border-[rgba(20,184,166,0.2)]">
              {envId}
            </span>
            {playing && (
              <span className="inline-flex items-center gap-1 text-[10px] font-mono text-emerald-400 bg-emerald-500/10 px-1.5 py-0.5 rounded border border-emerald-500/20">
                <span className="relative flex h-1.5 w-1.5">
                  <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75"></span>
                  <span className="relative inline-flex rounded-full h-1.5 w-1.5 bg-emerald-500"></span>
                </span>
                LIVE
              </span>
            )}
          </div>
          <h3 className="text-xs font-semibold text-[#e2e8f0] tracking-wide mt-1">
            2048 Live Player
          </h3>
        </div>

        {/* Metrics Bar */}
        <div className="flex items-center gap-4 text-xs font-mono">
          <div className="text-right">
            <span className="text-[10px] text-[#64748b] block uppercase">Score</span>
            <span className="text-amber-400 font-semibold">{score}</span>
          </div>
          <div className="text-right">
            <span className="text-[10px] text-[#64748b] block uppercase">Max Tile</span>
            <span className="text-emerald-400 font-semibold">{maxTile || "—"}</span>
          </div>
          <div className="text-right">
            <span className="text-[10px] text-[#64748b] block uppercase">Best</span>
            <span className="text-teal-400 font-semibold">{bestScore !== null ? bestScore : "—"}</span>
          </div>
          <div className="text-right">
            <span className="text-[10px] text-[#64748b] block uppercase">Episode</span>
            <span className="text-[#94a3b8]">{episode}</span>
          </div>
        </div>
      </div>

      {/* Stage: Canvas + Controls */}
      <div className="flex flex-col md:flex-row items-center justify-center gap-6 my-1">
        <canvas
          ref={canvasRef}
          width={CANVAS_SIZE}
          height={CANVAS_SIZE}
          className="rounded-lg shadow-inner bg-[#0f172a] border border-[rgba(20,184,166,0.12)]"
        />

        {/* Controls Sidebar */}
        <div className="flex flex-col gap-3.5 w-full md:w-44 shrink-0">
          {error && (
            <div className="p-2.5 rounded bg-rose-500/10 border border-rose-500/20 text-rose-400 text-xs">
              {error}
            </div>
          )}

          <button
            onClick={playing ? stop : start}
            disabled={loading}
            className={`w-full py-2 px-3 rounded text-xs font-semibold transition-all flex items-center justify-center gap-1.5 ${
              playing
                ? "bg-rose-500/20 hover:bg-rose-500/30 text-rose-400 border border-rose-500/30"
                : "bg-teal-500 hover:bg-teal-400 text-[#0f172a] shadow-sm"
            } disabled:opacity-40`}
          >
            {loading ? "Connecting…" : playing ? "■ Stop" : "▶ Watch Game"}
          </button>

          <div className="pt-2 border-t border-[rgba(255,255,255,0.05)] space-y-1.5">
            <div className="flex justify-between text-[10px] text-[#64748b]">
              <span>Playback Speed</span>
              <span className="font-mono text-[#94a3b8]">{speed} fps</span>
            </div>
            <input
              type="range"
              min={2}
              max={24}
              value={speed}
              onChange={(e) => setSpeed(Number(e.target.value))}
              className="w-full h-1 bg-[#0f172a] rounded-lg appearance-none cursor-pointer accent-teal-500"
            />
          </div>

          <div className="text-[10px] text-[#64748b] bg-[#0f172a]/50 p-2 rounded border border-[rgba(255,255,255,0.03)] space-y-0.5">
            <div className="flex justify-between">
              <span>Grid:</span>
              <span className="font-mono text-[#94a3b8]">4x4</span>
            </div>
            <div className="flex justify-between">
              <span>Step:</span>
              <span className="font-mono text-[#94a3b8]">{step}</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
