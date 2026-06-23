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
  highlight_rows?: number[];
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

  // Highlight only the rows being cleared
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
  // Per-cell color memory: cells keep the color of the piece that placed them
  const cellColorsRef = useRef<(string | null)[]>(new Array(ROWS * COLS).fill(null));
  const prevBoardRef = useRef<number[] | null>(null);

  const [playing, setPlaying] = useState(false);
  const [episode, setEpisode] = useState(0);
  const [episodeReward, setEpisodeReward] = useState(0);
  const [bestReward, setBestReward] = useState<number | null>(null);
  const [currentPiece, setCurrentPiece] = useState<string | null>(null);
  const [nextPiece, setNextPiece] = useState<string | null>(null);
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
    // Reset per-cell state
    cellColorsRef.current = new Array(ROWS * COLS).fill(null);
    prevBoardRef.current = null;

    const ws = new WebSocket(
      `${WS_BASE}/ws/missions/${missionId}/play?env_id=${envId}&fps=8`
    );
    wsRef.current = ws;

    ws.onopen = () => { setPlaying(true); setLoading(false); };

    ws.onmessage = (e) => {
      const frame: Frame = JSON.parse(e.data as string);
      if (frame.type === "frame" && frame.grid) {
        const obs = frame.grid;
        const board = obs.slice(0, 200);
        const highlightRows = frame.highlight_rows ?? [];

        // Determine current piece color for newly placed cells
        const curIdx = obs.slice(200, 207).indexOf(1);
        const curColor = curIdx >= 0 ? PIECE_COLORS[curIdx] : FALLBACK_COLOR;

        const prevBoard = prevBoardRef.current;
        const cellColors = cellColorsRef.current;

        // When lines are cleared the board shifts down — approximate by shifting color
        // memory down by the cleared count so existing pieces keep their color.
        const linesCleared = frame.lines_cleared_last ?? 0;
        if (linesCleared > 0 && prevBoard) {
          for (let r = ROWS - 1; r >= 0; r--) {
            for (let c = 0; c < COLS; c++) {
              cellColors[r * COLS + c] =
                r >= linesCleared ? cellColors[(r - linesCleared) * COLS + c] : null;
            }
          }
        }

        // Paint newly filled cells with the current piece color; clear emptied cells
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
        if (frame.episode) setEpisode(frame.episode);

        const nxtIdx = obs.slice(207, 214).indexOf(1);
        setCurrentPiece(curIdx >= 0 ? PIECE_NAMES[curIdx] : null);
        setNextPiece(nxtIdx >= 0 ? PIECE_NAMES[nxtIdx] : null);
      } else if (frame.type === "episode_end") {
        // Reset color memory between episodes
        cellColorsRef.current = new Array(ROWS * COLS).fill(null);
        prevBoardRef.current = null;
        const r = frame.total_reward ?? 0;
        setBestReward((prev) => (prev === null || r > prev ? r : prev));
      } else if (frame.type === "error") {
        setError(frame.message ?? "Unknown error");
        stop();
      }
    };

    ws.onerror = () => { setError("Connection failed"); stop(); };
    ws.onclose = () => { setPlaying(false); setLoading(false); wsRef.current = null; };
  }, [missionId, envId, stop]);

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
            <span className="text-[10px] text-[#14b8a6]">best {bestReward.toFixed(1)}</span>
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

      {/* Canvas + piece info side by side */}
      <div className="flex gap-4 items-start justify-center">
        <canvas
          ref={canvasRef}
          width={W}
          height={H}
          style={{
            imageRendering: "pixelated",
            borderRadius: 4,
            background: "#0f172a",
            border: "1px solid rgba(20,184,166,0.08)",
            flexShrink: 0,
          }}
        />

        {/* Sidebar: current / next piece */}
        <div className="space-y-3 pt-1 min-w-[64px]">
          {currentPiece && (
            <div className="space-y-1">
              <p className="text-[9px] text-[#64748b] uppercase tracking-widest">current</p>
              <div
                className="w-8 h-8 rounded flex items-center justify-center text-sm font-bold"
                style={{
                  background: PIECE_COLORS[PIECE_NAMES.indexOf(currentPiece)] + "33",
                  border: `1px solid ${PIECE_COLORS[PIECE_NAMES.indexOf(currentPiece)]}66`,
                  color: PIECE_COLORS[PIECE_NAMES.indexOf(currentPiece)],
                }}
              >
                {currentPiece}
              </div>
            </div>
          )}
          {nextPiece && (
            <div className="space-y-1">
              <p className="text-[9px] text-[#64748b] uppercase tracking-widest">next</p>
              <div
                className="w-8 h-8 rounded flex items-center justify-center text-sm font-bold opacity-60"
                style={{
                  background: PIECE_COLORS[PIECE_NAMES.indexOf(nextPiece)] + "22",
                  border: `1px solid ${PIECE_COLORS[PIECE_NAMES.indexOf(nextPiece)]}44`,
                  color: PIECE_COLORS[PIECE_NAMES.indexOf(nextPiece)],
                }}
              >
                {nextPiece}
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Stats */}
      {playing && (
        <div className="flex justify-between text-[10px] text-[#64748b]">
          <span>episode {episode}</span>
          <span>reward {episodeReward.toFixed(1)}</span>
        </div>
      )}

      {error && <p className="text-[10px] text-[#f87171]">{error}</p>}
    </div>
  );
}
