"use client";

import { useEffect, useRef, useState, useCallback } from "react";

const ROWS = 20;
const COLS = 10;
const CELL = 18;
const W = COLS * CELL;
const H = ROWS * CELL;

const WS_BASE =
  typeof window !== "undefined"
    ? `ws://${window.location.hostname}:8200`
    : "ws://localhost:8200";

const PIECE_COLORS = [
  "#06b6d4", // I — cyan
  "#fbbf24", // O — yellow
  "#a855f7", // T — purple
  "#4ade80", // S — green
  "#f87171", // Z — red
  "#60a5fa", // J — blue
  "#f97316", // L — orange
];
const PIECE_NAMES = ["I", "O", "T", "S", "Z", "J", "L"];
const FALLBACK_COLOR = "#14b8a6";

import { PolicyInspector, PolicyTelemetry } from "./PolicyInspector";

interface Frame {
  type: "frame" | "episode_end" | "error";
  grid?: number[];
  episode?: number;
  step?: number;
  episode_reward?: number;
  total_reward?: number;
  done?: boolean;
  message?: string;
  lines_cleared_last?: number;
  lines_cleared?: number;
  highlight_rows?: number[];
  q_values?: Record<string, number>;
  action_probs?: Record<string, number>;
  entropy?: number | null;
  selected_action?: string;
}

function drawFrame(
  ctx: CanvasRenderingContext2D,
  obs: number[],
  cellColors: (string | null)[],
  highlightRows: number[],
) {
  ctx.fillStyle = "#0f172a";
  ctx.fillRect(0, 0, W, H);

  for (let r = 0; r < ROWS; r++) {
    for (let c = 0; c < COLS; c++) {
      if (obs[r * COLS + c] <= 0.5) continue;
      const x = c * CELL;
      const y = r * CELL;
      ctx.fillStyle = cellColors[r * COLS + c] ?? FALLBACK_COLOR;
      ctx.fillRect(x + 1, y + 1, CELL - 2, CELL - 2);
      ctx.fillStyle = "rgba(255,255,255,0.18)";
      ctx.fillRect(x + 1, y + 1, CELL - 2, 2);
    }
  }

  // Highlight rows being cleared
  if (highlightRows.length > 0) {
    ctx.fillStyle = "rgba(250,204,21,0.65)";
    for (const row of highlightRows) {
      ctx.fillRect(0, row * CELL, W, CELL);
    }
  }

  // Grid lines
  ctx.strokeStyle = "rgba(20,184,166,0.06)";
  ctx.lineWidth = 0.5;
  for (let i = 0; i <= COLS; i++) {
    ctx.beginPath(); ctx.moveTo(i * CELL, 0); ctx.lineTo(i * CELL, H); ctx.stroke();
  }
  for (let i = 0; i <= ROWS; i++) {
    ctx.beginPath(); ctx.moveTo(0, i * CELL); ctx.lineTo(W, i * CELL); ctx.stroke();
  }
}

interface Props {
  missionId: string;
  envId?: string;
}

export function TetrisPlayer({ missionId, envId = "Tetris-v0" }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const wsRef = useRef<WebSocket | null>(null);
  const cellColorsRef = useRef<(string | null)[]>(new Array(ROWS * COLS).fill(null));
  const prevBoardRef = useRef<number[] | null>(null);

  const [playing, setPlaying] = useState(false);
  const [episode, setEpisode] = useState(0);
  const [step, setStep] = useState(0);
  const [episodeReward, setEpisodeReward] = useState(0);
  const [linesCleared, setLinesCleared] = useState(0);
  const [bestReward, setBestReward] = useState<number | null>(null);
  const [currentPiece, setCurrentPiece] = useState<string | null>(null);
  const [nextPiece, setNextPiece] = useState<string | null>(null);
  const [speed, setSpeed] = useState(10);
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
    cellColorsRef.current = new Array(ROWS * COLS).fill(null);
    prevBoardRef.current = null;

    const ws = new WebSocket(
      `${WS_BASE}/ws/missions/${missionId}/play?env_id=${envId}&fps=${speed}`
    );
    wsRef.current = ws;

    ws.onopen = () => { setPlaying(true); setLoading(false); };

    ws.onmessage = (e) => {
      const frame: Frame = JSON.parse(e.data as string);
      if (frame.type === "frame" && frame.grid) {
        const obs = frame.grid;
        const board = obs.slice(0, 200);
        const highlightRows = frame.highlight_rows ?? [];

        const curIdx = obs.slice(200, 207).indexOf(1);
        const curColor = curIdx >= 0 ? PIECE_COLORS[curIdx] : FALLBACK_COLOR;

        const prevBoard = prevBoardRef.current;
        const cellColors = cellColorsRef.current;

        const linesCleared = frame.lines_cleared_last ?? 0;
        if (linesCleared > 0 && prevBoard) {
          for (let r = ROWS - 1; r >= 0; r--) {
            for (let c = 0; c < COLS; c++) {
              cellColors[r * COLS + c] =
                r >= linesCleared ? cellColors[(r - linesCleared) * COLS + c] : null;
            }
          }
        }

        for (let i = 0; i < ROWS * COLS; i++) {
          const filled = board[i] > 0.5;
          const wasFilled = prevBoard ? prevBoard[i] > 0.5 : false;
          if (filled && !wasFilled) cellColors[i] = curColor;
          else if (!filled) cellColors[i] = null;
        }
        prevBoardRef.current = board;

        const ctx = canvasRef.current?.getContext("2d");
        if (ctx) drawFrame(ctx, obs, cellColors, highlightRows);

        setEpisodeReward(frame.episode_reward ?? 0);
        if (frame.lines_cleared !== undefined) setLinesCleared(frame.lines_cleared);
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

        const nxtIdx = obs.slice(207, 214).indexOf(1);
        setCurrentPiece(curIdx >= 0 ? PIECE_NAMES[curIdx] : null);
        setNextPiece(nxtIdx >= 0 ? PIECE_NAMES[nxtIdx] : null);
      } else if (frame.type === "episode_end") {
        cellColorsRef.current = new Array(ROWS * COLS).fill(null);
        prevBoardRef.current = null;
        setLinesCleared(0);
        const r = frame.total_reward ?? 0;
        setBestReward((prev) => (prev === null || r > prev ? r : prev));
      } else if (frame.type === "error") {
        setError(frame.message ?? "Inference error");
        stop();
      }
    };

    ws.onerror = () => { setError("WebSocket connection failed"); stop(); };
    ws.onclose = () => { setPlaying(false); setLoading(false); wsRef.current = null; };
  }, [missionId, envId, speed, stop]);

  // Initial draw
  useEffect(() => {
    const ctx = canvasRef.current?.getContext("2d");
    if (ctx) drawFrame(ctx, new Array(ROWS * COLS).fill(0), new Array(ROWS * COLS).fill(null), []);
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
            Tetris Live Player
          </h3>
        </div>

        {/* Metrics Bar */}
        <div className="flex items-center gap-4 text-xs font-mono">
          <div className="text-right">
            <span className="text-[10px] text-[#64748b] block uppercase">Lines</span>
            <span className="text-teal-400 font-semibold">{linesCleared}</span>
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

      {/* Stage: Canvas + Pieces + Control Panel */}
      <div className="flex flex-col md:flex-row items-center justify-center gap-6 my-1">
        <div className="flex gap-3 items-start justify-center">
          <canvas
            ref={canvasRef}
            width={W}
            height={H}
            className="rounded-lg shadow-inner bg-[#0f172a] border border-[rgba(20,184,166,0.12)] shrink-0"
            style={{ imageRendering: "pixelated" }}
          />

          {/* Piece Previews */}
          <div className="flex flex-col gap-2.5 pt-1 min-w-[56px]">
            <div className="bg-[#0f172a]/70 p-2 rounded border border-[rgba(255,255,255,0.04)] text-center space-y-1">
              <p className="text-[9px] text-[#64748b] uppercase tracking-wider font-mono">Current</p>
              <div
                className="w-8 h-8 mx-auto rounded flex items-center justify-center text-xs font-bold transition-all"
                style={{
                  background: currentPiece ? PIECE_COLORS[PIECE_NAMES.indexOf(currentPiece)] + "26" : "transparent",
                  border: currentPiece ? `1px solid ${PIECE_COLORS[PIECE_NAMES.indexOf(currentPiece)]}66` : "1px dashed rgba(255,255,255,0.1)",
                  color: currentPiece ? PIECE_COLORS[PIECE_NAMES.indexOf(currentPiece)] : "#475569",
                }}
              >
                {currentPiece || "—"}
              </div>
            </div>

            <div className="bg-[#0f172a]/70 p-2 rounded border border-[rgba(255,255,255,0.04)] text-center space-y-1">
              <p className="text-[9px] text-[#64748b] uppercase tracking-wider font-mono">Next</p>
              <div
                className="w-8 h-8 mx-auto rounded flex items-center justify-center text-xs font-bold transition-all opacity-75"
                style={{
                  background: nextPiece ? PIECE_COLORS[PIECE_NAMES.indexOf(nextPiece)] + "22" : "transparent",
                  border: nextPiece ? `1px solid ${PIECE_COLORS[PIECE_NAMES.indexOf(nextPiece)]}44` : "1px dashed rgba(255,255,255,0.1)",
                  color: nextPiece ? PIECE_COLORS[PIECE_NAMES.indexOf(nextPiece)] : "#475569",
                }}
              >
                {nextPiece || "—"}
              </div>
            </div>
          </div>
        </div>

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
              max={20}
              value={speed}
              onChange={(e) => setSpeed(Number(e.target.value))}
              className="w-full h-1 bg-[#0f172a] rounded-lg appearance-none cursor-pointer accent-teal-500"
            />
          </div>

          <div className="text-[10px] text-[#64748b] bg-[#0f172a]/50 p-2 rounded border border-[rgba(255,255,255,0.03)] space-y-0.5">
            <div className="flex justify-between">
              <span>Grid:</span>
              <span className="font-mono text-[#94a3b8]">10x20</span>
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
