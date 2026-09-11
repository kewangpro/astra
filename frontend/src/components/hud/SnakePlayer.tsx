"use client";

import { useEffect, useRef, useState, useCallback } from "react";

const GRID = 16;
const CELL = 20; // px per cell
const CANVAS_SIZE = GRID * CELL;

const WS_BASE =
  typeof window !== "undefined"
    ? `ws://${window.location.hostname}:8200`
    : "ws://localhost:8200";

import { PolicyInspector, PolicyTelemetry } from "./PolicyInspector";

interface Frame {
  type: "frame" | "episode_end" | "error";
  grid?: number[];
  episode?: number;
  step?: number;
  episode_reward?: number;
  total_reward?: number;
  food_eaten?: number;
  done?: boolean;
  message?: string;
  q_values?: Record<string, number>;
  action_probs?: Record<string, number>;
  entropy?: number | null;
  selected_action?: string;
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

  // Subtle grid lines
  ctx.strokeStyle = "rgba(20,184,166,0.06)";
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
  const [step, setStep] = useState(0);
  const [episodeReward, setEpisodeReward] = useState(0);
  const [foodEaten, setFoodEaten] = useState(0);
  const [bestReward, setBestReward] = useState<number | null>(null);
  const [speed, setSpeed] = useState(12);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [policyTelemetry, setPolicyTelemetry] = useState<PolicyTelemetry | null>(null);

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
        if (ctx) drawFrame(ctx, frame.grid);
        setEpisodeReward(frame.episode_reward ?? 0);
        if (frame.food_eaten !== undefined) setFoodEaten(frame.food_eaten);
        if (frame.step !== undefined) setStep(frame.step);
        if (frame.episode) setEpisode(frame.episode);
        if (frame.q_values || frame.action_probs || frame.selected_action) {
          setPolicyTelemetry({
            q_values: frame.q_values,
            action_probs: frame.action_probs,
            entropy: frame.entropy,
            selected_action: frame.selected_action,
          });
        }
      } else if (frame.type === "episode_end") {
        setFoodEaten(0);
        const r = frame.total_reward ?? 0;
        setBestReward((prev) => (prev === null || r > prev ? r : prev));
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

  // Initial canvas draw
  useEffect(() => {
    const ctx = canvasRef.current?.getContext("2d");
    if (ctx) drawFrame(ctx, new Array(GRID * GRID).fill(0));
  }, []);

  // Cleanup on unmount
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
            Snake Live Player
          </h3>
        </div>

        {/* Metrics Bar */}
        <div className="flex items-center gap-4 text-xs font-mono">
          <div className="text-right">
            <span className="text-[10px] text-[#64748b] block uppercase">Food</span>
            <span className="text-teal-400 font-semibold">{foodEaten}</span>
          </div>
          <div className="text-right">
            <span className="text-[10px] text-[#64748b] block uppercase">Reward</span>
            <span className="text-amber-400 font-semibold">{episodeReward.toFixed(1)}</span>
          </div>
          <div className="text-right">
            <span className="text-[10px] text-[#64748b] block uppercase">Best</span>
            <span className="text-emerald-400 font-semibold">{bestReward !== null ? bestReward.toFixed(1) : "—"}</span>
          </div>
          <div className="text-right">
            <span className="text-[10px] text-[#64748b] block uppercase">Episode</span>
            <span className="text-[#94a3b8]">{episode}</span>
          </div>
        </div>
      </div>

      {/* Stage: Canvas + Control Panel */}
      <div className="flex flex-col md:flex-row items-center justify-center gap-6 my-1">
        <canvas
          ref={canvasRef}
          width={CANVAS_SIZE}
          height={CANVAS_SIZE}
          className="rounded-lg shadow-inner bg-[#0f172a] border border-[rgba(20,184,166,0.12)]"
          style={{ imageRendering: "pixelated" }}
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
              min={4}
              max={24}
              value={speed}
              onChange={(e) => setSpeed(Number(e.target.value))}
              className="w-full h-1 bg-[#0f172a] rounded-lg appearance-none cursor-pointer accent-teal-500"
            />
          </div>

          <div className="text-[10px] text-[#64748b] bg-[#0f172a]/50 p-2 rounded border border-[rgba(255,255,255,0.03)] space-y-0.5">
            <div className="flex justify-between">
              <span>Grid:</span>
              <span className="font-mono text-[#94a3b8]">16x16</span>
            </div>
            <div className="flex justify-between">
              <span>Step:</span>
              <span className="font-mono text-[#94a3b8]">{step}</span>
            </div>
          </div>
        </div>
      </div>

      <PolicyInspector telemetry={policyTelemetry} />
    </div>
  );
}
