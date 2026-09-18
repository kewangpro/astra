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

import { PolicyInspector, PolicyTelemetry } from "./PolicyInspector";

interface Frame {
  type: "frame" | "episode_end" | "error";
  grid?: number[];
  episode?: number;
  step?: number;
  episode_reward?: number;
  total_reward?: number;
  score?: number;
  bricks_cleared?: number;
  aliens_killed?: number;
  asteroids_hit?: number;
  crossings?: number;
  collisions?: number;
  divers_saved?: number;
  enemies_killed?: number;
  oxygen?: number;
  done?: boolean;
  message?: string;
  q_values?: Record<string, number>;
  action_probs?: Record<string, number>;
  entropy?: number | null;
  selected_action?: string;
}

function drawMinAtar(ctx: CanvasRenderingContext2D, grid: number[], envId: string) {
  const isFreeway = envId.toLowerCase().includes("freeway");
  const isSeaquest = envId.toLowerCase().includes("seaquest");

  ctx.fillStyle = isSeaquest ? "#091428" : "#0f172a";
  ctx.fillRect(0, 0, W, H);

  // Subtle grid lines
  ctx.strokeStyle = isSeaquest ? "rgba(56, 189, 248, 0.08)" : "rgba(20, 184, 166, 0.06)";
  ctx.lineWidth = 0.5;
  for (let c = 0; c <= COLS; c++) {
    ctx.beginPath(); ctx.moveTo(c * CELL, 0); ctx.lineTo(c * CELL, H); ctx.stroke();
  }
  for (let r = 0; r <= ROWS; r++) {
    ctx.beginPath(); ctx.moveTo(0, r * CELL); ctx.lineTo(W, r * CELL); ctx.stroke();
  }

  for (let r = 0; r < ROWS; r++) {
    for (let c = 0; c < COLS; c++) {
      const cell = grid[r * COLS + c] || 0;
      const x = c * CELL;
      const y = r * CELL;

      if (cell === 0) continue;

      if (isFreeway) {
        if (cell === 4) {
          // Safe Sidewalk
          ctx.fillStyle = "rgba(45, 212, 191, 0.12)";
          ctx.fillRect(x, y + 2, CELL, CELL - 4);
          ctx.strokeStyle = "rgba(45, 212, 191, 0.4)";
          ctx.setLineDash([2, 4]);
          ctx.strokeRect(x, y + 2, CELL, CELL - 4);
          ctx.setLineDash([]);
        } else if (cell === 1) {
          // Chicken (bright yellow/lime chicken)
          ctx.fillStyle = "#facc15";
          ctx.shadowColor = "rgba(250, 204, 21, 0.8)";
          ctx.shadowBlur = 8;
          ctx.beginPath();
          ctx.arc(x + CELL / 2, y + CELL / 2, CELL / 3, 0, Math.PI * 2);
          ctx.fill();
          // Beak
          ctx.fillStyle = "#f97316";
          ctx.fillRect(x + CELL / 2 - 2, y + 3, 4, 3);
          ctx.shadowBlur = 0;
        } else if (cell === 2) {
          // Car moving left (rose red)
          ctx.fillStyle = "#f43f5e";
          ctx.shadowColor = "rgba(244, 63, 94, 0.5)";
          ctx.shadowBlur = 6;
          ctx.beginPath();
          ctx.roundRect(x + 2, y + 5, CELL - 4, CELL - 10, 3);
          ctx.fill();
          // Headlight
          ctx.fillStyle = "#ffffff";
          ctx.fillRect(x + 3, y + 7, 2, CELL - 14);
          ctx.shadowBlur = 0;
        } else if (cell === 3) {
          // Car moving right (amber orange)
          ctx.fillStyle = "#f59e0b";
          ctx.shadowColor = "rgba(245, 158, 11, 0.5)";
          ctx.shadowBlur = 6;
          ctx.beginPath();
          ctx.roundRect(x + 2, y + 5, CELL - 4, CELL - 10, 3);
          ctx.fill();
          // Headlight
          ctx.fillStyle = "#ffffff";
          ctx.fillRect(x + CELL - 5, y + 7, 2, CELL - 14);
          ctx.shadowBlur = 0;
        } else if (cell === 5) {
          // Collision flash
          ctx.fillStyle = "#ef4444";
          ctx.shadowColor = "rgba(239, 68, 68, 0.9)";
          ctx.shadowBlur = 10;
          ctx.fillRect(x + 1, y + 1, CELL - 2, CELL - 2);
          ctx.shadowBlur = 0;
        }
      } else if (isSeaquest) {
        if (cell === 6) {
          // Surface water / Oxygen bar
          ctx.fillStyle = "rgba(56, 189, 248, 0.25)";
          ctx.fillRect(x, y + 14, CELL, CELL - 14);
          ctx.strokeStyle = "#38bdf8";
          ctx.lineWidth = 1;
          ctx.beginPath(); ctx.moveTo(x, y + 14); ctx.lineTo(x + CELL, y + 14); ctx.stroke();
        } else if (cell === 1) {
          // Player Submarine (Teal)
          ctx.fillStyle = "#14b8a6";
          ctx.shadowColor = "rgba(20, 184, 166, 0.8)";
          ctx.shadowBlur = 8;
          ctx.beginPath();
          ctx.roundRect(x + 2, y + 5, CELL - 4, CELL - 10, 4);
          ctx.fill();
          // Periscope
          ctx.fillRect(x + CELL / 2 - 1, y + 2, 3, 4);
          ctx.shadowBlur = 0;
        } else if (cell === 2) {
          // Enemy (shark / enemy sub)
          ctx.fillStyle = "#f43f5e";
          ctx.shadowColor = "rgba(244, 63, 94, 0.6)";
          ctx.shadowBlur = 6;
          ctx.beginPath();
          ctx.roundRect(x + 3, y + 6, CELL - 6, CELL - 12, 3);
          ctx.fill();
          ctx.shadowBlur = 0;
        } else if (cell === 3) {
          // Diver (Amber / gold swimmer)
          ctx.fillStyle = "#facc15";
          ctx.shadowColor = "rgba(250, 204, 21, 0.7)";
          ctx.shadowBlur = 6;
          ctx.beginPath();
          ctx.arc(x + CELL / 2, y + CELL / 2, CELL / 3.5, 0, Math.PI * 2);
          ctx.fill();
          ctx.shadowBlur = 0;
        } else if (cell === 4) {
          // Player Torpedo (Cyan)
          ctx.fillStyle = "#38bdf8";
          ctx.shadowColor = "rgba(56, 189, 248, 0.9)";
          ctx.shadowBlur = 8;
          ctx.fillRect(x + 4, y + CELL / 2 - 1.5, CELL - 8, 3);
          ctx.shadowBlur = 0;
        } else if (cell === 5) {
          // Enemy Torpedo (Orange)
          ctx.fillStyle = "#fb923c";
          ctx.shadowColor = "rgba(251, 146, 60, 0.8)";
          ctx.shadowBlur = 6;
          ctx.fillRect(x + 4, y + CELL / 2 - 1.5, CELL - 8, 3);
          ctx.shadowBlur = 0;
        } else if (cell === 7) {
          // Explosion
          ctx.fillStyle = "#ef4444";
          ctx.fillRect(x + 2, y + 2, CELL - 4, CELL - 4);
        }
      } else {
        // Breakout / Space Invaders / Asteroids
        if (cell === 1) {
          // Player (paddle, cannon, ship)
          ctx.fillStyle = "#4ade80";
          ctx.shadowColor = "rgba(74, 222, 128, 0.5)";
          ctx.shadowBlur = 6;
          ctx.fillRect(x + 1, y + 6, CELL - 2, CELL - 12);
          ctx.shadowBlur = 0;
        } else if (cell === 2) {
          // Ball / Alien / Asteroid
          ctx.fillStyle = "#ffffff";
          ctx.shadowColor = "rgba(255, 255, 255, 0.8)";
          ctx.shadowBlur = 8;
          ctx.beginPath();
          ctx.arc(x + CELL / 2, y + CELL / 2, CELL / 3, 0, Math.PI * 2);
          ctx.fill();
          ctx.shadowBlur = 0;
        } else if (cell === 3) {
          // Brick or bomb
          const color = BRICK_ROW_COLORS[(r - 1) % BRICK_ROW_COLORS.length] || "#f87171";
          ctx.fillStyle = color;
          ctx.beginPath();
          ctx.roundRect(x + 2, y + 4, CELL - 4, CELL - 8, 3);
          ctx.fill();
        } else if (cell === 4) {
          // Laser / Bullet
          ctx.fillStyle = "#38bdf8";
          ctx.shadowColor = "rgba(56, 189, 248, 0.9)";
          ctx.shadowBlur = 6;
          ctx.fillRect(x + CELL / 2 - 1.5, y + 2, 3, CELL - 4);
          ctx.shadowBlur = 0;
        } else if (cell === 5) {
          // Shield
          ctx.fillStyle = "#38bdf8";
          ctx.fillRect(x + 2, y + 4, CELL - 4, CELL - 8);
        }
      }
    }
  }
}

function initialGrid(envId: string): number[] {
  const g = new Array(100).fill(0);
  const isFreeway = envId.toLowerCase().includes("freeway");
  const isSeaquest = envId.toLowerCase().includes("seaquest");

  if (isFreeway) {
    // Goal & Start Sidewalks
    for (let c = 0; c < 10; c++) {
      g[0 * 10 + c] = 4;
      g[9 * 10 + c] = 4;
    }
    // Chicken at row 9 col 4
    g[9 * 10 + 4] = 1;
    // Sample traffic cars
    g[2 * 10 + 2] = 2;
    g[4 * 10 + 7] = 3;
    g[6 * 10 + 5] = 2;
    g[8 * 10 + 1] = 3;
    return g;
  }

  if (isSeaquest) {
    // Surface water at row 0
    for (let c = 0; c < 10; c++) g[0 * 10 + c] = 6;
    // Submarine at row 5 col 2
    g[5 * 10 + 2] = 1;
    // Enemy at row 3 col 7
    g[3 * 10 + 7] = 2;
    // Diver at row 7 col 8
    g[7 * 10 + 8] = 3;
    return g;
  }

  // Breakout default
  for (let r = 1; r <= 3; r++) {
    for (let c = 0; c < 10; c++) g[r * 10 + c] = 3;
  }
  g[9 * 10 + 4] = 1;
  g[9 * 10 + 5] = 1;
  g[6 * 10 + 4] = 2;
  return g;
}

function getGameTitle(envId: string): string {
  const lower = envId.toLowerCase();
  if (lower.includes("freeway")) return "MinAtar Freeway Live Player";
  if (lower.includes("seaquest")) return "MinAtar Seaquest Live Player";
  if (lower.includes("space")) return "MinAtar Space Invaders Live Player";
  if (lower.includes("asteroid")) return "MinAtar Asteroids Live Player";
  return "MinAtar Breakout Live Player";
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
  const [bestScore, setBestScore] = useState<number | null>(null);
  const [statA, setStatA] = useState<{ label: string; value: number | string }>({ label: "Count", value: 0 });
  const [statB, setStatB] = useState<{ label: string; value: number | string } | null>(null);
  const [step, setStep] = useState(0);
  const [speed, setSpeed] = useState(14);
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
        if (ctx) drawMinAtar(ctx, frame.grid, envId);

        if (frame.step !== undefined) setStep(frame.step);
        if (frame.score !== undefined) {
          setScore(frame.score);
          setBestScore((prev) => (prev === null ? frame.score! : Math.max(prev, frame.score!)));
        }
        if (frame.crossings !== undefined) {
          setStatA({ label: "Crossings", value: frame.crossings });
          setStatB({ label: "Collisions", value: frame.collisions ?? 0 });
        } else if (frame.divers_saved !== undefined) {
          setStatA({ label: "Divers", value: frame.divers_saved });
          setStatB({ label: "Oxygen", value: frame.oxygen ?? 0 });
        } else if (frame.aliens_killed !== undefined) {
          setStatA({ label: "Aliens", value: frame.aliens_killed });
        } else if (frame.asteroids_hit !== undefined) {
          setStatA({ label: "Asteroids", value: frame.asteroids_hit });
        } else if (frame.bricks_cleared !== undefined) {
          setStatA({ label: "Bricks", value: frame.bricks_cleared });
        }

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
    if (ctx) drawMinAtar(ctx, initialGrid(envId), envId);
  }, [envId]);

  useEffect(() => () => { wsRef.current?.close(); }, []);

  const isFreeway = envId.toLowerCase().includes("freeway");
  const isSeaquest = envId.toLowerCase().includes("seaquest");

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
            {getGameTitle(envId)}
          </h3>
        </div>

        {/* Metrics Bar */}
        <div className="flex items-center gap-4 text-xs font-mono">
          <div className="text-right">
            <span className="text-[10px] text-[#64748b] block uppercase">Score</span>
            <span className="text-teal-400 font-semibold">{score}</span>
          </div>
          <div className="text-right">
            <span className="text-[10px] text-[#64748b] block uppercase">{statA.label}</span>
            <span className="text-rose-400 font-semibold">{statA.value}</span>
          </div>
          {statB && (
            <div className="text-right">
              <span className="text-[10px] text-[#64748b] block uppercase">{statB.label}</span>
              <span className="text-sky-400 font-semibold">{statB.value}</span>
            </div>
          )}
          <div className="text-right">
            <span className="text-[10px] text-[#64748b] block uppercase">Best</span>
            <span className="text-emerald-400 font-semibold">{bestScore !== null ? bestScore : "—"}</span>
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
          width={W}
          height={H}
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
              min={4}
              max={30}
              value={speed}
              onChange={(e) => setSpeed(Number(e.target.value))}
              className="w-full h-1 bg-[#0f172a] rounded-lg appearance-none cursor-pointer accent-teal-500"
            />
          </div>

          <div className="text-[10px] text-[#64748b] bg-[#0f172a]/50 p-2 rounded border border-[rgba(255,255,255,0.03)] space-y-0.5">
            <div className="flex justify-between">
              <span>Grid:</span>
              <span className="font-mono text-[#94a3b8]">10x10</span>
            </div>
            <div className="flex justify-between">
              <span>Step:</span>
              <span className="font-mono text-[#94a3b8]">{step}</span>
            </div>
            {isFreeway && (
              <div className="flex justify-between">
                <span>Lanes:</span>
                <span className="font-mono text-emerald-400">8 traffic</span>
              </div>
            )}
            {isSeaquest && (
              <div className="flex justify-between">
                <span>Capacity:</span>
                <span className="font-mono text-teal-400">6 divers</span>
              </div>
            )}
          </div>
        </div>
      </div>

      <PolicyInspector telemetry={policyTelemetry} />
    </div>
  );
}
