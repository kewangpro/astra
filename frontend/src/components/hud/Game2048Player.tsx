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
            if (ctx) drawBoard(ctx, frame.grid);
          }
          if (frame.step !== undefined) setStep(frame.step);
          if (frame.score !== undefined) setScore(frame.score);
          if (frame.max_tile !== undefined) {
            setMaxTile(frame.max_tile);
            setBestScore((prev) => (prev === null ? frame.score! : Math.max(prev, frame.score!)));
          }
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
      if (ctx) drawBoard(ctx, new Array(16).fill(0));
    }
  }, []);

  return (
    <div className="bg-[#1e293b] border border-[rgba(255,255,255,0.05)] rounded-lg p-5">
      <div className="flex items-center justify-between mb-4">
        <div>
          <h3 className="text-xs font-semibold text-[#e2e8f0] tracking-wide flex items-center gap-2">
            <span>2048 Live Player</span>
            <span className="text-[10px] text-[#64748b] font-mono">Game2048-v0</span>
          </h3>
          <p className="text-[11px] text-[#94a3b8] mt-0.5">
            Model inference running on Apple Silicon / CPU
          </p>
        </div>

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
            <span className="text-[10px] text-[#64748b] block uppercase">Step</span>
            <span className="text-[#94a3b8]">{step}</span>
          </div>
        </div>
      </div>

      <div className="flex flex-col sm:flex-row items-center gap-6 justify-center my-2">
        <canvas
          ref={canvasRef}
          width={CANVAS_SIZE}
          height={CANVAS_SIZE}
          className="rounded-lg shadow-inner bg-[#0f172a]"
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
              min={2}
              max={24}
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
