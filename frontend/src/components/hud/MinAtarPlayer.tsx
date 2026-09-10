"use client";

import { useEffect, useRef, useState, useCallback } from "react";

const ROWS = 10;
const COLS = 10;
const CELL = 24;
const W = COLS * CELL;
const H = ROWS * CELL;

const WS_BASE =
  typeof window !== "undefined"
    ? `ws://${window.location.hostname}:8200`
    : "ws://localhost:8200";

const BRICK_ROW_COLORS = [
  "#f87171", // Row 1: red
  "#fbbf24", // Row 2: amber
  "#38bdf8", // Row 3: sky
  "#a855f7", // Row 4: purple
];

interface Frame {
  type: "frame" | "episode_end" | "error";
  grid?: number[];
  episode?: number;
  step?: number;
  episode_reward?: number;
  total_reward?: number;
  score?: number;
  bricks_cleared?: number;
  done?: boolean;
  message?: string;
}

function drawMinAtar(ctx: CanvasRenderingContext2D, grid: number[]) {
  ctx.fillStyle = "#0f172a";
  ctx.fillRect(0, 0, W, H);

  // Subtle grid lines
  ctx.strokeStyle = "rgba(255, 255, 255, 0.04)";
  ctx.lineWidth = 0.5;
  for (let c = 0; c <= COLS; c++) {
    ctx.beginPath();
    ctx.moveTo(c * CELL, 0);
    ctx.lineTo(c * CELL, H);
    ctx.stroke();
  }
  for (let r = 0; r <= ROWS; r++) {
    ctx.beginPath();
    ctx.moveTo(0, r * CELL);
    ctx.lineTo(W, r * CELL);
    ctx.stroke();
  }

  // Draw entities: 0=empty, 1=paddle, 2=ball, 3=brick
  for (let r = 0; r < ROWS; r++) {
    for (let c = 0; c < COLS; c++) {
      const cell = grid[r * COLS + c] || 0;
      const x = c * CELL;
      const y = r * CELL;

      if (cell === 1) {
        // Paddle (green bar)
        ctx.fillStyle = "#4ade80";
        ctx.shadowColor = "rgba(74, 222, 128, 0.5)";
        ctx.shadowBlur = 6;
        ctx.fillRect(x + 1, y + 6, CELL - 2, CELL - 12);
        ctx.shadowBlur = 0;
      } else if (cell === 2) {
        // Ball (white circle/square)
        ctx.fillStyle = "#ffffff";
        ctx.shadowColor = "rgba(255, 255, 255, 0.8)";
        ctx.shadowBlur = 8;
        ctx.beginPath();
        ctx.arc(x + CELL / 2, y + CELL / 2, CELL / 3, 0, Math.PI * 2);
        ctx.fill();
        ctx.shadowBlur = 0;
      } else if (cell === 3) {
        // Brick
        const color = BRICK_ROW_COLORS[(r - 1) % BRICK_ROW_COLORS.length] || "#f87171";
        ctx.fillStyle = color;
        ctx.beginPath();
        ctx.roundRect(x + 2, y + 4, CELL - 4, CELL - 8, 3);
        ctx.fill();
      }
    }
  }
}

interface Props {
  missionId: string;
  envId?: string;
}

export function MinAtarPlayer({ missionId, envId = "MinAtar-Breakout-v0" }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const wsRef = useRef<WebSocket | null>(null);

  const [playing, setPlaying] = useState(false);
  const [episode, setEpisode] = useState(0);
  const [score, setScore] = useState(0);
  const [bricksCleared, setBricksCleared] = useState(0);
  const [step, setStep] = useState(0);
  const [speed, setSpeed] = useState(12);
  const [error, setError] = useState<string | null>(null);
  const [connected, setConnected] = useState(false);

  const framesRef = useRef<Frame[]>([]);
  const animFrameRef = useRef<number | null>(null);
  const lastDrawRef = useRef<number>(0);

  const stopPlayback = useCallback(() => {
    setPlaying(false);
    if (animFrameRef.current) cancelAnimationFrame(animFrameRef.current);
    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }
    setConnected(false);
  }, []);

  const connect = useCallback(() => {
    setError(null);
    framesRef.current = [];

    const url = `${WS_BASE}/ws/missions/${missionId}/play?env_id=${envId}&fps=${speed}`;
    const ws = new WebSocket(url);
    wsRef.current = ws;

    ws.onopen = () => {
      setConnected(true);
      setPlaying(true);
    };

    ws.onmessage = (evt) => {
      try {
        const msg: Frame = JSON.parse(evt.data);
        if (msg.type === "error") {
          setError(msg.message ?? "Inference error");
          stopPlayback();
          return;
        }
        if (msg.type === "frame" || msg.type === "episode_end") {
          framesRef.current.push(msg);
        }
      } catch {}
    };

    ws.onerror = () => {
      setError("WebSocket connection failed.");
      stopPlayback();
    };

    ws.onclose = () => {
      setConnected(false);
    };
  }, [missionId, envId, speed, stopPlayback]);

  // Animation loop
  useEffect(() => {
    if (!playing) return;

    const interval = 1000 / speed;

    function loop(now: number) {
      if (now - lastDrawRef.current >= interval && framesRef.current.length > 0) {
        const frame = framesRef.current.shift()!;
        lastDrawRef.current = now;

        if (frame.type === "frame" && frame.grid) {
          const canvas = canvasRef.current;
          if (canvas) {
            const ctx = canvas.getContext("2d");
            if (ctx) drawMinAtar(ctx, frame.grid);
          }
          if (frame.step !== undefined) setStep(frame.step);
          if (frame.score !== undefined) setScore(frame.score);
          if (frame.bricks_cleared !== undefined) setBricksCleared(frame.bricks_cleared);
        } else if (frame.type === "episode_end") {
          setEpisode((e) => e + 1);
        }
      }
      animFrameRef.current = requestAnimationFrame(loop);
    }

    animFrameRef.current = requestAnimationFrame(loop);
    return () => {
      if (animFrameRef.current) cancelAnimationFrame(animFrameRef.current);
    };
  }, [playing, speed]);

  // Initial draw
  useEffect(() => {
    const canvas = canvasRef.current;
    if (canvas) {
      const ctx = canvas.getContext("2d");
      if (ctx) drawMinAtar(ctx, new Array(100).fill(0));
    }
  }, []);

  return (
    <div className="bg-[#1e293b] border border-[rgba(255,255,255,0.05)] rounded-lg p-5">
      <div className="flex items-center justify-between mb-4">
        <div>
          <h3 className="text-xs font-semibold text-[#e2e8f0] tracking-wide flex items-center gap-2">
            <span>MinAtar Breakout Live Player</span>
            <span className="text-[10px] text-[#64748b] font-mono">10x10 Arcade</span>
          </h3>
          <p className="text-[11px] text-[#94a3b8] mt-0.5">
            Model inference running on Apple Silicon / CPU
          </p>
        </div>

        <div className="flex items-center gap-4 text-xs font-mono">
          <div className="text-right">
            <span className="text-[10px] text-[#64748b] block uppercase">Score</span>
            <span className="text-teal-400 font-semibold">{score}</span>
          </div>
          <div className="text-right">
            <span className="text-[10px] text-[#64748b] block uppercase">Bricks</span>
            <span className="text-rose-400 font-semibold">{bricksCleared}</span>
          </div>
          <div className="text-right">
            <span className="text-[10px] text-[#64748b] block uppercase">Step</span>
            <span className="text-[#94a3b8]">{step}</span>
          </div>
        </div>
      </div>

      <div className="flex flex-col sm:flex-row items-center gap-6 justify-center my-2">
        <canvas
          ref={canvasRef}
          width={W}
          height={H}
          className="rounded-lg shadow-inner bg-[#0f172a] border border-[#334155]/60"
        />

        <div className="flex flex-col gap-3 min-w-[160px]">
          {error && (
            <div className="p-2 rounded bg-rose-500/10 border border-rose-500/20 text-rose-400 text-xs">
              {error}
            </div>
          )}

          <div className="flex flex-col gap-2">
            {!playing ? (
              <button
                onClick={connect}
                className="px-4 py-2 rounded bg-teal-500 hover:bg-teal-400 text-[#0f172a] font-semibold text-xs transition-colors"
              >
                Watch Game
              </button>
            ) : (
              <button
                onClick={stopPlayback}
                className="px-4 py-2 rounded bg-rose-500/20 hover:bg-rose-500/30 text-rose-400 border border-rose-500/30 text-xs font-medium transition-colors"
              >
                Stop
              </button>
            )}
          </div>

          <div className="pt-3 border-t border-[rgba(255,255,255,0.05)] space-y-1.5">
            <div className="flex justify-between text-[10px] text-[#64748b]">
              <span>Speed</span>
              <span className="font-mono">{speed} fps</span>
            </div>
            <input
              type="range"
              min={4}
              max={30}
              value={speed}
              onChange={(e) => setSpeed(Number(e.target.value))}
              className="w-full h-1 bg-[#0f172a] rounded-lg appearance-none cursor-pointer accent-teal-500"
            />
          </div>
        </div>
      </div>
    </div>
  );
}
