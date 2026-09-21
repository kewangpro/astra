"use client";

import { useEffect, useRef, useState, useCallback } from "react";

const ROWS = 10;
const COLS = 10;
const CELL = 24;
const W = COLS * CELL;
const H = ROWS * CELL;

const BRICK_ROW_COLORS = [
  "#f87171", // Row 1: red
  "#fbbf24", // Row 2: amber
  "#38bdf8", // Row 3: sky
  "#a855f7", // Row 4: purple
];

import { PolicyInspector, PolicyTelemetry } from "./PolicyInspector";
import { policyPlayUrl } from "@/lib/playWs";

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
  gold_collected?: number;
  ghosts_eaten?: number;
  pellets_left?: number;
  done?: boolean;
  message?: string;
  ship_dir?: number;
  player_dir?: number;
  q_values?: Record<string, number>;
  action_probs?: Record<string, number>;
  entropy?: number | null;
  selected_action?: string;
}

function drawFreeway(ctx: CanvasRenderingContext2D, grid: number[]) {
  // 1. Sidewalks & Road Surface
  // Row 0: Goal Sidewalk (Safe Zone / Scoring Finish Line)
  ctx.fillStyle = "#064e3b"; // Rich emerald turf
  ctx.fillRect(0, 0, W, CELL);

  // Checkered finish line pattern on Row 0
  ctx.fillStyle = "#10b981";
  for (let c = 0; c < COLS; c++) {
    if (c % 2 === 0) {
      ctx.fillRect(c * CELL, 0, CELL, CELL - 3);
    }
  }
  ctx.fillStyle = "rgba(52, 211, 153, 0.5)";
  ctx.fillRect(0, CELL - 3, W, 3); // Bright mint finish curb

  // Rows 1..8: Deep Highway Asphalt
  ctx.fillStyle = "#090d16"; // Crisp dark asphalt
  ctx.fillRect(0, CELL, W, CELL * 8);

  // Top and bottom road curb rails
  ctx.strokeStyle = "#475569";
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  ctx.moveTo(0, CELL); ctx.lineTo(W, CELL);
  ctx.moveTo(0, CELL * 9); ctx.lineTo(W, CELL * 9);
  ctx.stroke();

  // Highway Lane Dividers
  for (let r = 1; r < 8; r++) {
    const y = (r + 1) * CELL;
    ctx.beginPath();
    if (r === 4) {
      // Center highway double yellow divider (between row 4 and 5)
      ctx.strokeStyle = "rgba(250, 204, 21, 0.7)";
      ctx.lineWidth = 1;
      ctx.setLineDash([4, 4]);
      ctx.moveTo(0, y - 1); ctx.lineTo(W, y - 1);
      ctx.moveTo(0, y + 1); ctx.lineTo(W, y + 1);
      ctx.stroke();
    } else {
      // Regular dashed white lane lines
      ctx.strokeStyle = "rgba(255, 255, 255, 0.2)";
      ctx.lineWidth = 0.8;
      ctx.setLineDash([3, 5]);
      ctx.moveTo(0, y); ctx.lineTo(W, y);
      ctx.stroke();
    }
  }
  ctx.setLineDash([]);

  // Faint directional lane arrows on the asphalt
  ctx.font = "8px monospace";
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  for (let r = 1; r <= 8; r++) {
    const y = r * CELL + CELL / 2;
    const isLeft = r % 2 === 1;
    ctx.fillStyle = isLeft ? "rgba(6, 182, 212, 0.09)" : "rgba(217, 70, 239, 0.09)";
    for (let c = 1; c < COLS; c += 3) {
      ctx.fillText(isLeft ? "◄" : "►", c * CELL + CELL / 2, y);
    }
  }

  // Row 9: Starting Sidewalk (Spawn Curb)
  ctx.fillStyle = "#1e293b"; // Sturdy concrete slate
  ctx.fillRect(0, CELL * 9, W, CELL);
  // Pavement slab dividers
  ctx.strokeStyle = "rgba(148, 163, 184, 0.2)";
  ctx.lineWidth = 1;
  for (let c = 1; c < COLS; c++) {
    ctx.beginPath();
    ctx.moveTo(c * CELL, CELL * 9);
    ctx.lineTo(c * CELL, H);
    ctx.stroke();
  }

  // Subtle grid overlay for crisp cell alignment
  ctx.strokeStyle = "rgba(255, 255, 255, 0.04)";
  ctx.lineWidth = 0.5;
  for (let c = 0; c <= COLS; c++) {
    ctx.beginPath(); ctx.moveTo(c * CELL, 0); ctx.lineTo(c * CELL, H); ctx.stroke();
  }

  // 2. Render Entities
  for (let r = 0; r < ROWS; r++) {
    for (let c = 0; c < COLS; c++) {
      const cell = grid[r * COLS + c] || 0;
      const x = c * CELL;
      const y = r * CELL;

      if (cell === 0 || cell === 4) continue;

      if (cell === 1) {
        // === PLAYER: CHICKEN ===
        // Radiant Neon Golden Yellow & White Character with Red Comb & Orange Beak
        const cx = x + CELL / 2;
        const cy = y + CELL / 2;

        ctx.shadowColor = "rgba(250, 204, 21, 0.9)";
        ctx.shadowBlur = 10;

        // Yellow body
        ctx.fillStyle = "#facc15";
        ctx.beginPath();
        ctx.ellipse(cx, cy + 1, 7, 8, 0, 0, Math.PI * 2);
        ctx.fill();

        // White wing feather highlight
        ctx.fillStyle = "#fef08a";
        ctx.beginPath();
        ctx.ellipse(cx - 3, cy + 1, 2.5, 5, -0.2, 0, Math.PI * 2);
        ctx.fill();

        // Rooster Comb on top (Crimson Red)
        ctx.fillStyle = "#ef4444";
        ctx.beginPath();
        ctx.arc(cx - 2.5, cy - 7, 2, 0, Math.PI * 2);
        ctx.arc(cx + 0.5, cy - 8, 2.2, 0, Math.PI * 2);
        ctx.arc(cx + 3.5, cy - 6, 1.8, 0, Math.PI * 2);
        ctx.fill();

        // Pointed Beak facing UP toward Goal (Vivid Orange)
        ctx.fillStyle = "#f97316";
        ctx.beginPath();
        ctx.moveTo(cx - 2.5, cy - 4.5);
        ctx.lineTo(cx + 2.5, cy - 4.5);
        ctx.lineTo(cx, cy - 9);
        ctx.closePath();
        ctx.fill();

        // Eyes (Crisp black with white glint)
        ctx.shadowBlur = 0;
        ctx.fillStyle = "#0f172a";
        ctx.beginPath();
        ctx.arc(cx - 2.5, cy - 2.5, 1.2, 0, Math.PI * 2);
        ctx.arc(cx + 2.5, cy - 2.5, 1.2, 0, Math.PI * 2);
        ctx.fill();

        ctx.fillStyle = "#ffffff";
        ctx.beginPath();
        ctx.arc(cx - 2.8, cy - 2.8, 0.5, 0, Math.PI * 2);
        ctx.arc(cx + 2.2, cy - 2.8, 0.5, 0, Math.PI * 2);
        ctx.fill();

      } else if (cell === 2) {
        // === TRAFFIC MOVING LEFT (←) ===
        // Electric Cyan / Sky Blue Sports Car
        ctx.shadowColor = "rgba(6, 182, 212, 0.85)";
        ctx.shadowBlur = 8;

        // Chassis (Aerodynamic nose on the left)
        ctx.fillStyle = "#06b6d4";
        ctx.beginPath();
        ctx.roundRect(x + 2, y + 4, CELL - 4, CELL - 8, [5, 2, 2, 5]);
        ctx.fill();

        // Windshield (Dark blue glass)
        ctx.fillStyle = "#082f49";
        ctx.beginPath();
        ctx.roundRect(x + 7, y + 6, CELL - 13, CELL - 12, 2);
        ctx.fill();

        // Roof highlight
        ctx.fillStyle = "#38bdf8";
        ctx.fillRect(x + 9, y + 7, CELL - 17, CELL - 14);

        // Headlights on LEFT (White / Diamond Cyan)
        ctx.shadowColor = "rgba(255, 255, 255, 0.95)";
        ctx.shadowBlur = 4;
        ctx.fillStyle = "#ffffff";
        ctx.fillRect(x + 2, y + 5, 2, 3);
        ctx.fillRect(x + 2, y + CELL - 8, 2, 3);

        // Taillights on RIGHT (Vivid Red)
        ctx.shadowColor = "rgba(239, 68, 68, 0.85)";
        ctx.shadowBlur = 3;
        ctx.fillStyle = "#ef4444";
        ctx.fillRect(x + CELL - 3.5, y + 5, 1.5, 3);
        ctx.fillRect(x + CELL - 3.5, y + CELL - 8, 1.5, 3);
        ctx.shadowBlur = 0;

      } else if (cell === 3) {
        // === TRAFFIC MOVING RIGHT (→) ===
        // Neon Fuchsia / Amethyst Magenta Sports Car
        ctx.shadowColor = "rgba(217, 70, 239, 0.85)";
        ctx.shadowBlur = 8;

        // Chassis (Aerodynamic nose on the right)
        ctx.fillStyle = "#d946ef";
        ctx.beginPath();
        ctx.roundRect(x + 2, y + 4, CELL - 4, CELL - 8, [2, 5, 5, 2]);
        ctx.fill();

        // Windshield (Dark violet glass)
        ctx.fillStyle = "#3b0764";
        ctx.beginPath();
        ctx.roundRect(x + 6, y + 6, CELL - 13, CELL - 12, 2);
        ctx.fill();

        // Roof highlight
        ctx.fillStyle = "#f0abfc";
        ctx.fillRect(x + 8, y + 7, CELL - 17, CELL - 14);

        // Headlights on RIGHT (White / Amber LED)
        ctx.shadowColor = "rgba(255, 255, 255, 0.95)";
        ctx.shadowBlur = 4;
        ctx.fillStyle = "#ffffff";
        ctx.fillRect(x + CELL - 4, y + 5, 2, 3);
        ctx.fillRect(x + CELL - 4, y + CELL - 8, 2, 3);

        // Taillights on LEFT (Vivid Red)
        ctx.shadowColor = "rgba(239, 68, 68, 0.85)";
        ctx.shadowBlur = 3;
        ctx.fillStyle = "#ef4444";
        ctx.fillRect(x + 2, y + 5, 1.5, 3);
        ctx.fillRect(x + 2, y + CELL - 8, 1.5, 3);
        ctx.shadowBlur = 0;

      } else if (cell === 5) {
        // === COLLISION / CRASH IMPACT ===
        const cx = x + CELL / 2;
        const cy = y + CELL / 2;

        ctx.shadowColor = "rgba(239, 68, 68, 0.95)";
        ctx.shadowBlur = 14;

        // Red outer starburst
        ctx.fillStyle = "#ef4444";
        ctx.beginPath();
        for (let i = 0; i < 8; i++) {
          const angle = (i * Math.PI) / 4;
          const r1 = CELL / 2 - 1;
          const r2 = CELL / 4;
          ctx.lineTo(cx + Math.cos(angle) * r1, cy + Math.sin(angle) * r1);
          ctx.lineTo(cx + Math.cos(angle + Math.PI / 8) * r2, cy + Math.sin(angle + Math.PI / 8) * r2);
        }
        ctx.closePath();
        ctx.fill();

        // Yellow inner star
        ctx.fillStyle = "#fde047";
        ctx.beginPath();
        ctx.arc(cx, cy, CELL / 3.5, 0, Math.PI * 2);
        ctx.fill();

        // White core
        ctx.fillStyle = "#ffffff";
        ctx.beginPath();
        ctx.arc(cx, cy, CELL / 6, 0, Math.PI * 2);
        ctx.fill();
        ctx.shadowBlur = 0;
      }
    }
  }
}

function headingFromAction(action?: string): number | undefined {
  switch (action) {
    case "RIGHT":
      return 0;
    case "DOWN":
      return 1;
    case "LEFT":
      return 2;
    case "UP":
      return 3;
    default:
      return undefined;
  }
}

function drawMinAtar(ctx: CanvasRenderingContext2D, grid: number[], envId: string, shipDir: number = 0) {
  const lower = envId.toLowerCase();
  const isFreeway = lower.includes("freeway");
  const isSeaquest = lower.includes("seaquest");
  const isSpaceInvaders = lower.includes("space");
  const isAsterix = lower.includes("asterix");
  const isAsteroids = lower.includes("asteroid") && !isAsterix;
  const isPacMan = lower.includes("pacman") || lower.includes("pac-man");

  if (isFreeway) {
    drawFreeway(ctx, grid);
    return;
  }

  if (isSeaquest) {
    ctx.fillStyle = "#091428";
  } else if (isSpaceInvaders) {
    ctx.fillStyle = "#080b18";
  } else if (isAsterix) {
    ctx.fillStyle = "#0c1210";
  } else if (isPacMan) {
    ctx.fillStyle = "#020617";
  } else if (isAsteroids) {
    ctx.fillStyle = "#050811";
  } else {
    ctx.fillStyle = "#0f172a";
  }
  ctx.fillRect(0, 0, W, H);

  // Subtle grid lines
  if (isSeaquest) {
    ctx.strokeStyle = "rgba(56, 189, 248, 0.08)";
  } else if (isSpaceInvaders) {
    ctx.strokeStyle = "rgba(192, 132, 252, 0.07)";
  } else if (isAsterix) {
    ctx.strokeStyle = "rgba(250, 204, 21, 0.06)";
  } else if (isPacMan) {
    ctx.strokeStyle = "rgba(20, 184, 166, 0.08)";
  } else if (isAsteroids) {
    ctx.strokeStyle = "rgba(56, 189, 248, 0.05)";
  } else {
    ctx.strokeStyle = "rgba(20, 184, 166, 0.06)";
  }
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

      if (isSeaquest) {
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
      } else if (isSpaceInvaders) {
        if (cell === 1) {
          // Cannon (Player)
          ctx.fillStyle = "#4ade80";
          ctx.shadowColor = "rgba(74, 222, 128, 0.8)";
          ctx.shadowBlur = 6;
          ctx.fillRect(x + 2, y + 8, CELL - 4, CELL - 11);
          ctx.fillRect(x + CELL / 2 - 1.5, y + 2, 3, 6);
          ctx.shadowBlur = 0;
        } else if (cell === 2) {
          // Alien Invader (Magenta / Violet neon)
          ctx.fillStyle = "#e879f9";
          ctx.shadowColor = "rgba(232, 121, 249, 0.8)";
          ctx.shadowBlur = 6;
          ctx.beginPath();
          ctx.roundRect(x + 3, y + 5, CELL - 6, CELL - 10, 3);
          ctx.fill();
          // Antennae & Claws
          ctx.fillRect(x + 4, y + 2, 2, 3);
          ctx.fillRect(x + CELL - 6, y + 2, 2, 3);
          ctx.fillRect(x + 4, y + CELL - 5, 2, 3);
          ctx.fillRect(x + CELL - 6, y + CELL - 5, 2, 3);
          // Eyes
          ctx.shadowBlur = 0;
          ctx.fillStyle = "#080b18";
          ctx.fillRect(x + 6, y + 8, 2, 2);
          ctx.fillRect(x + CELL - 8, y + 8, 2, 2);
        } else if (cell === 3) {
          // Alien Bomb (Dropping red energy bolt)
          ctx.fillStyle = "#f43f5e";
          ctx.shadowColor = "rgba(244, 63, 94, 0.9)";
          ctx.shadowBlur = 8;
          ctx.fillRect(x + CELL / 2 - 1.5, y + 2, 3, CELL - 4);
          ctx.shadowBlur = 0;
        } else if (cell === 4) {
          // Player Laser (Cyan laser bolt)
          ctx.fillStyle = "#38bdf8";
          ctx.shadowColor = "rgba(56, 189, 248, 0.9)";
          ctx.shadowBlur = 8;
          ctx.fillRect(x + CELL / 2 - 1, y + 1, 2, CELL - 2);
          ctx.shadowBlur = 0;
        } else if (cell === 5) {
          // Bunker Shield (Teal barrier)
          ctx.fillStyle = "#0d9488";
          ctx.strokeStyle = "#2dd4bf";
          ctx.lineWidth = 1;
          ctx.beginPath();
          ctx.roundRect(x + 2, y + 3, CELL - 4, CELL - 6, 2);
          ctx.fill();
          ctx.stroke();
        }
      } else if (isAsterix) {
        if (cell === 1) {
          ctx.fillStyle = "#2dd4bf";
          ctx.shadowColor = "rgba(45, 212, 191, 0.8)";
          ctx.shadowBlur = 8;
          ctx.beginPath();
          ctx.moveTo(x + CELL / 2, y + 3);
          ctx.lineTo(x + CELL - 4, y + CELL / 2);
          ctx.lineTo(x + CELL / 2, y + CELL - 3);
          ctx.lineTo(x + 4, y + CELL / 2);
          ctx.closePath();
          ctx.fill();
          ctx.shadowBlur = 0;
        } else if (cell === 2) {
          ctx.fillStyle = "#f43f5e";
          ctx.shadowColor = "rgba(244, 63, 94, 0.7)";
          ctx.shadowBlur = 6;
          ctx.beginPath();
          ctx.arc(x + CELL / 2, y + CELL / 2, CELL / 3.2, 0, Math.PI * 2);
          ctx.fill();
          ctx.shadowBlur = 0;
        } else if (cell === 3) {
          ctx.fillStyle = "#facc15";
          ctx.shadowColor = "rgba(250, 204, 21, 0.85)";
          ctx.shadowBlur = 8;
          ctx.beginPath();
          ctx.arc(x + CELL / 2, y + CELL / 2, CELL / 4, 0, Math.PI * 2);
          ctx.fill();
          ctx.shadowBlur = 0;
        } else if (cell === 4) {
          ctx.fillStyle = "rgba(56, 189, 248, 0.45)";
          ctx.fillRect(x + 8, y + 8, CELL - 16, CELL - 16);
        }
      } else if (isPacMan) {
        if (cell === 1) {
          // Mouth wedge is drawn facing right; rotate by heading
          // (0 RIGHT, 1 DOWN, 2 LEFT, 3 UP).
          ctx.save();
          ctx.translate(x + CELL / 2, y + CELL / 2);
          ctx.rotate((shipDir * Math.PI) / 2);
          ctx.fillStyle = "#facc15";
          ctx.shadowColor = "rgba(250, 204, 21, 0.7)";
          ctx.shadowBlur = 8;
          ctx.beginPath();
          ctx.arc(0, 0, CELL / 2.4, 0.35, Math.PI * 2 - 0.35);
          ctx.lineTo(0, 0);
          ctx.closePath();
          ctx.fill();
          ctx.restore();
        } else if (cell === 2) {
          ctx.fillStyle = "#1e3a5f";
          ctx.fillRect(x + 1, y + 1, CELL - 2, CELL - 2);
          ctx.strokeStyle = "#38bdf8";
          ctx.lineWidth = 1;
          ctx.strokeRect(x + 2, y + 2, CELL - 4, CELL - 4);
        } else if (cell === 3) {
          ctx.fillStyle = "#e2e8f0";
          ctx.beginPath();
          ctx.arc(x + CELL / 2, y + CELL / 2, 2.2, 0, Math.PI * 2);
          ctx.fill();
        } else if (cell === 4) {
          ctx.fillStyle = "#f472b6";
          ctx.shadowColor = "rgba(244, 114, 182, 0.7)";
          ctx.shadowBlur = 6;
          ctx.beginPath();
          ctx.arc(x + CELL / 2, y + CELL / 2, 4.5, 0, Math.PI * 2);
          ctx.fill();
          ctx.shadowBlur = 0;
        } else if (cell === 5) {
          ctx.fillStyle = "#f43f5e";
          ctx.beginPath();
          ctx.arc(x + CELL / 2, y + CELL / 2 - 2, CELL / 3, Math.PI, 0);
          ctx.lineTo(x + CELL - 4, y + CELL - 4);
          ctx.lineTo(x + 4, y + CELL - 4);
          ctx.closePath();
          ctx.fill();
        } else if (cell === 6) {
          ctx.fillStyle = "#38bdf8";
          ctx.beginPath();
          ctx.arc(x + CELL / 2, y + CELL / 2 - 2, CELL / 3, Math.PI, 0);
          ctx.lineTo(x + CELL - 4, y + CELL - 4);
          ctx.lineTo(x + 4, y + CELL - 4);
          ctx.closePath();
          ctx.fill();
        }
      } else if (isAsteroids) {
        if (cell === 1) {
          // Spaceship (Cyan arrowhead wedge) rotated to heading (0: UP, 1: RIGHT, 2: DOWN, 3: LEFT)
          ctx.save();
          ctx.translate(x + CELL / 2, y + CELL / 2);
          ctx.rotate((shipDir * Math.PI) / 2);
          ctx.fillStyle = "#38bdf8";
          ctx.shadowColor = "rgba(56, 189, 248, 0.85)";
          ctx.shadowBlur = 8;
          ctx.beginPath();
          ctx.moveTo(0, -CELL / 2 + 3);
          ctx.lineTo(CELL / 2 - 4, CELL / 2 - 4);
          ctx.lineTo(0, CELL / 2 - 7);
          ctx.lineTo(-CELL / 2 + 4, CELL / 2 - 4);
          ctx.closePath();
          ctx.fill();
          ctx.restore();
        } else if (cell === 2) {
          // Asteroid (Slate rock with crater)
          ctx.fillStyle = "#cbd5e1";
          ctx.shadowColor = "rgba(203, 213, 225, 0.6)";
          ctx.shadowBlur = 6;
          ctx.beginPath();
          ctx.arc(x + CELL / 2, y + CELL / 2, CELL / 2.7, 0, Math.PI * 2);
          ctx.fill();
          // Crater detail
          ctx.fillStyle = "#64748b";
          ctx.beginPath();
          ctx.arc(x + CELL / 2 - 2, y + CELL / 2 - 2, 2.5, 0, Math.PI * 2);
          ctx.fill();
          ctx.shadowBlur = 0;
        } else if (cell === 4) {
          // Laser Bullet (Yellow plasma orb)
          ctx.fillStyle = "#facc15";
          ctx.shadowColor = "rgba(250, 204, 21, 0.9)";
          ctx.shadowBlur = 8;
          ctx.beginPath();
          ctx.arc(x + CELL / 2, y + CELL / 2, 2.5, 0, Math.PI * 2);
          ctx.fill();
          ctx.shadowBlur = 0;
        }
      } else {
        // Breakout
        if (cell === 1) {
          // Player (paddle)
          ctx.fillStyle = "#4ade80";
          ctx.shadowColor = "rgba(74, 222, 128, 0.5)";
          ctx.shadowBlur = 6;
          ctx.fillRect(x + 1, y + 6, CELL - 2, CELL - 12);
          ctx.shadowBlur = 0;
        } else if (cell === 2) {
          // Ball
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
}

function initialGrid(envId: string): number[] {
  const g = new Array(100).fill(0);
  const lower = envId.toLowerCase();
  const isFreeway = lower.includes("freeway");
  const isSeaquest = lower.includes("seaquest");
  const isSpaceInvaders = lower.includes("space");
  const isAsterix = lower.includes("asterix");
  const isAsteroids = lower.includes("asteroid") && !isAsterix;
  const isPacMan = lower.includes("pacman") || lower.includes("pac-man");

  if (isFreeway) {
    // Goal & Start Sidewalks
    for (let c = 0; c < 10; c++) {
      g[0 * 10 + c] = 4;
      g[9 * 10 + c] = 4;
    }
    // Chicken at row 9 col 4
    g[9 * 10 + 4] = 1;
    // Sample traffic cars (odd rows: 2 = moving left, even rows: 3 = moving right)
    g[1 * 10 + 2] = 2;
    g[1 * 10 + 7] = 2;
    g[2 * 10 + 4] = 3;
    g[3 * 10 + 1] = 2;
    g[3 * 10 + 6] = 2;
    g[4 * 10 + 3] = 3;
    g[4 * 10 + 8] = 3;
    g[5 * 10 + 5] = 2;
    g[6 * 10 + 2] = 3;
    g[6 * 10 + 7] = 3;
    g[7 * 10 + 4] = 2;
    g[8 * 10 + 1] = 3;
    g[8 * 10 + 6] = 3;
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

  if (isSpaceInvaders) {
    // Marching alien armada across rows 1, 2, 3
    for (let c = 1; c <= 8; c += 2) {
      g[1 * 10 + c] = 2;
      g[2 * 10 + (c === 7 ? 6 : c + 1)] = 2;
      g[3 * 10 + c] = 2;
    }
    // Bunker shields at row 7
    g[7 * 10 + 2] = 5;
    g[7 * 10 + 4] = 5;
    g[7 * 10 + 7] = 5;
    // Alien bomb dropping
    g[4 * 10 + 3] = 3;
    // Player cannon at row 9 col 4
    g[9 * 10 + 4] = 1;
    // Player laser firing upward
    g[6 * 10 + 4] = 4;
    return g;
  }

  if (isAsterix) {
    g[5 * 10 + 5] = 1;
    g[2 * 10 + 0] = 2;
    g[2 * 10 + 1] = 4;
    g[6 * 10 + 9] = 3;
    g[6 * 10 + 8] = 4;
    g[4 * 10 + 2] = 2;
    return g;
  }

  if (isPacMan) {
    for (let c = 0; c < 10; c++) {
      g[c] = 2;
      g[9 * 10 + c] = 2;
      g[c * 10] = 2;
      g[c * 10 + 9] = 2;
    }
    g[2 * 10 + 2] = 2;
    g[2 * 10 + 3] = 2;
    g[2 * 10 + 6] = 2;
    g[2 * 10 + 7] = 2;
    g[1 * 10 + 1] = 4;
    g[1 * 10 + 8] = 4;
    g[3 * 10 + 4] = 3;
    g[3 * 10 + 5] = 3;
    g[7 * 10 + 4] = 1;
    g[1 * 10 + 4] = 5;
    g[1 * 10 + 5] = 5;
    return g;
  }

  if (isAsteroids) {
    // Spaceship in center
    g[5 * 10 + 5] = 1;
    // Floating asteroids of various positions
    g[1 * 10 + 2] = 2;
    g[2 * 10 + 8] = 2;
    g[4 * 10 + 1] = 2;
    g[7 * 10 + 2] = 2;
    g[8 * 10 + 7] = 2;
    g[6 * 10 + 8] = 2;
    // Plasma bullet shot
    g[3 * 10 + 5] = 4;
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
  if (lower.includes("freeway")) return "MinAtar Freeway";
  if (lower.includes("seaquest")) return "MinAtar Seaquest";
  if (lower.includes("space")) return "MinAtar Space Invaders";
  if (lower.includes("asterix")) return "MinAtar Asterix";
  if (lower.includes("pacman") || lower.includes("pac-man")) return "Grid Pac-Man";
  if (lower.includes("asteroid")) return "MinAtar Asteroids";
  return "MinAtar Breakout";
}

interface Props {
  missionId?: string;
  modelId?: string;
  envId?: string;
}

export function MinAtarPlayer({ missionId, modelId, envId = "MinAtar-Breakout-v0" }: Props) {
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
  const shipDirRef = useRef<number>(0);

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
      policyPlayUrl({ modelId, missionId }, envId, speed)
    );
    wsRef.current = ws;

    ws.onopen = () => {
      setPlaying(true);
      setLoading(false);
    };

    ws.onmessage = (e) => {
      const frame: Frame = JSON.parse(e.data as string);
      if (frame.type === "frame" && frame.grid) {
        const actionHeading = headingFromAction(frame.selected_action);
        if (frame.player_dir !== undefined) {
          shipDirRef.current = frame.player_dir;
        } else if (frame.ship_dir !== undefined) {
          shipDirRef.current = frame.ship_dir;
        } else if (actionHeading !== undefined) {
          shipDirRef.current = actionHeading;
        }
        const ctx = canvasRef.current?.getContext("2d");
        if (ctx) {
          drawMinAtar(
            ctx,
            frame.grid,
            envId,
            frame.player_dir ?? frame.ship_dir ?? actionHeading ?? shipDirRef.current,
          );
        }

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
        } else if (frame.gold_collected !== undefined) {
          setStatA({ label: "Gold", value: frame.gold_collected });
        } else if (frame.ghosts_eaten !== undefined) {
          setStatA({ label: "Ghosts", value: frame.ghosts_eaten });
          setStatB({ label: "Pellets", value: frame.pellets_left ?? 0 });
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
  }, [missionId, modelId, envId, speed, stop]);

  // Initial draw & stats initialization
  useEffect(() => {
    const lower = envId.toLowerCase();
    if (lower.includes("freeway")) {
      setStatA({ label: "Crossings", value: 0 });
      setStatB({ label: "Collisions", value: 0 });
    } else if (lower.includes("seaquest")) {
      setStatA({ label: "Divers", value: 0 });
      setStatB({ label: "Oxygen", value: 200 });
    } else if (lower.includes("space")) {
      setStatA({ label: "Aliens", value: 0 });
      setStatB(null);
    } else if (lower.includes("asterix")) {
      setStatA({ label: "Gold", value: 0 });
      setStatB(null);
    } else if (lower.includes("pacman") || lower.includes("pac-man")) {
      setStatA({ label: "Ghosts", value: 0 });
      setStatB({ label: "Pellets", value: 0 });
    } else if (lower.includes("asteroid")) {
      setStatA({ label: "Asteroids", value: 0 });
      setStatB(null);
    } else {
      setStatA({ label: "Bricks", value: 0 });
      setStatB(null);
    }
    const ctx = canvasRef.current?.getContext("2d");
    if (ctx) drawMinAtar(ctx, initialGrid(envId), envId);
  }, [envId]);

  useEffect(() => () => { wsRef.current?.close(); }, []);

  const lower = envId.toLowerCase();
  const isFreeway = lower.includes("freeway");
  const isSeaquest = lower.includes("seaquest");
  const isSpaceInvaders = lower.includes("space");
  const isAsterix = lower.includes("asterix");
  const isAsteroids = lower.includes("asteroid") && !isAsterix;
  const isPacMan = lower.includes("pacman") || lower.includes("pac-man");
  const isBreakout = !isFreeway && !isSeaquest && !isSpaceInvaders && !isAsteroids && !isAsterix && !isPacMan;

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
            {loading ? "Connecting…" : playing ? "■ Stop" : "▶ Play"}
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
            {isSpaceInvaders && (
              <div className="flex justify-between">
                <span>Armada:</span>
                <span className="font-mono text-fuchsia-400">Marching</span>
              </div>
            )}
            {isAsteroids && (
              <>
                <div className="flex justify-between">
                  <span>Wrap:</span>
                  <span className="font-mono text-sky-400">Toroidal</span>
                </div>
                <div className="flex justify-between">
                  <span>Motion:</span>
                  <span className="font-mono text-teal-400">Grid Step</span>
                </div>
              </>
            )}
            {isAsterix && (
              <div className="flex justify-between">
                <span>Spawn:</span>
                <span className="font-mono text-amber-400">Sides</span>
              </div>
            )}
            {isPacMan && (
              <div className="flex justify-between">
                <span>Ghosts:</span>
                <span className="font-mono text-rose-400">4 chase</span>
              </div>
            )}
            {isBreakout && (
              <div className="flex justify-between">
                <span>Wall:</span>
                <span className="font-mono text-amber-400">30 bricks</span>
              </div>
            )}
          </div>

          {/* Color Legend */}
          {isFreeway && (
            <div className="text-[10px] bg-[#0f172a]/70 p-2 rounded border border-[rgba(255,255,255,0.05)] space-y-1 font-mono">
              <div className="text-[9px] uppercase tracking-wider text-[#64748b] font-semibold border-b border-[rgba(255,255,255,0.05)] pb-0.5 mb-1">
                Color Key
              </div>
              <div className="flex items-center justify-between">
                <span className="flex items-center gap-1.5">
                  <span className="w-2.5 h-2.5 rounded-full bg-yellow-400 border border-yellow-300 shadow-[0_0_6px_rgba(250,204,21,0.8)] inline-block"></span>
                  <span className="text-[#cbd5e1]">Chicken</span>
                </span>
                <span className="text-yellow-400 font-semibold">Player</span>
              </div>
              <div className="flex items-center justify-between">
                <span className="flex items-center gap-1.5">
                  <span className="w-2.5 h-2 rounded bg-cyan-400 border border-cyan-300 shadow-[0_0_6px_rgba(6,182,212,0.8)] inline-block"></span>
                  <span className="text-[#cbd5e1]">Traffic (◄)</span>
                </span>
                <span className="text-cyan-400 font-semibold">Cyan</span>
              </div>
              <div className="flex items-center justify-between">
                <span className="flex items-center gap-1.5">
                  <span className="w-2.5 h-2 rounded bg-fuchsia-500 border border-fuchsia-400 shadow-[0_0_6px_rgba(217,70,239,0.8)] inline-block"></span>
                  <span className="text-[#cbd5e1]">Traffic (►)</span>
                </span>
                <span className="text-fuchsia-400 font-semibold">Magenta</span>
              </div>
              <div className="flex items-center justify-between">
                <span className="flex items-center gap-1.5">
                  <span className="w-2.5 h-2 rounded bg-emerald-600 border border-emerald-400 inline-block"></span>
                  <span className="text-[#cbd5e1]">Goal Row 0</span>
                </span>
                <span className="text-emerald-400 font-semibold">+1 Pt</span>
              </div>
              <div className="flex items-center justify-between">
                <span className="flex items-center gap-1.5">
                  <span className="w-2.5 h-2 rounded bg-slate-600 border border-slate-400 inline-block"></span>
                  <span className="text-[#cbd5e1]">Start Row 9</span>
                </span>
                <span className="text-slate-400">Spawn</span>
              </div>
              <div className="flex items-center justify-between">
                <span className="flex items-center gap-1.5">
                  <span className="w-2.5 h-2.5 rounded bg-rose-500 border border-rose-300 shadow-[0_0_6px_rgba(244,63,94,0.9)] inline-block"></span>
                  <span className="text-[#cbd5e1]">Collision</span>
                </span>
                <span className="text-rose-400">Crash</span>
              </div>
            </div>
          )}

          {isSeaquest && (
            <div className="text-[10px] bg-[#0f172a]/70 p-2 rounded border border-[rgba(255,255,255,0.05)] space-y-1 font-mono">
              <div className="text-[9px] uppercase tracking-wider text-[#64748b] font-semibold border-b border-[rgba(255,255,255,0.05)] pb-0.5 mb-1">
                Color Key
              </div>
              <div className="flex items-center justify-between">
                <span className="flex items-center gap-1.5">
                  <span className="w-2.5 h-2 rounded bg-teal-400 border border-teal-300 inline-block"></span>
                  <span className="text-[#cbd5e1]">Submarine</span>
                </span>
                <span className="text-teal-400 font-semibold">Player</span>
              </div>
              <div className="flex items-center justify-between">
                <span className="flex items-center gap-1.5">
                  <span className="w-2.5 h-2 rounded bg-rose-500 border border-rose-400 inline-block"></span>
                  <span className="text-[#cbd5e1]">Enemies</span>
                </span>
                <span className="text-rose-400">Threat</span>
              </div>
              <div className="flex items-center justify-between">
                <span className="flex items-center gap-1.5">
                  <span className="w-2.5 h-2.5 rounded-full bg-yellow-400 border border-yellow-300 inline-block"></span>
                  <span className="text-[#cbd5e1]">Divers</span>
                </span>
                <span className="text-yellow-400">Rescue</span>
              </div>
              <div className="flex items-center justify-between">
                <span className="flex items-center gap-1.5">
                  <span className="w-2.5 h-2 rounded bg-sky-400 border border-sky-300 inline-block"></span>
                  <span className="text-[#cbd5e1]">Torpedoes</span>
                </span>
                <span className="text-sky-400">Fire</span>
              </div>
              <div className="flex items-center justify-between">
                <span className="flex items-center gap-1.5">
                  <span className="w-2.5 h-2 rounded bg-sky-600/60 border border-sky-500 inline-block"></span>
                  <span className="text-[#cbd5e1]">Surface</span>
                </span>
                <span className="text-sky-300">O₂ Refill</span>
              </div>
            </div>
          )}

          {isSpaceInvaders && (
            <div className="text-[10px] bg-[#0f172a]/70 p-2 rounded border border-[rgba(255,255,255,0.05)] space-y-1 font-mono">
              <div className="text-[9px] uppercase tracking-wider text-[#64748b] font-semibold border-b border-[rgba(255,255,255,0.05)] pb-0.5 mb-1">
                Color Key
              </div>
              <div className="flex items-center justify-between">
                <span className="flex items-center gap-1.5">
                  <span className="w-2.5 h-2 rounded bg-emerald-400 border border-emerald-300 inline-block"></span>
                  <span className="text-[#cbd5e1]">Cannon</span>
                </span>
                <span className="text-emerald-400 font-semibold">Player</span>
              </div>
              <div className="flex items-center justify-between">
                <span className="flex items-center gap-1.5">
                  <span className="w-2.5 h-2 rounded bg-fuchsia-400 border border-fuchsia-300 inline-block"></span>
                  <span className="text-[#cbd5e1]">Invaders</span>
                </span>
                <span className="text-fuchsia-400 font-semibold">Armada</span>
              </div>
              <div className="flex items-center justify-between">
                <span className="flex items-center gap-1.5">
                  <span className="w-2.5 h-2 rounded bg-rose-500 border border-rose-400 inline-block"></span>
                  <span className="text-[#cbd5e1]">Bomb</span>
                </span>
                <span className="text-rose-400">Drop</span>
              </div>
              <div className="flex items-center justify-between">
                <span className="flex items-center gap-1.5">
                  <span className="w-2.5 h-2 rounded bg-sky-400 border border-sky-300 inline-block"></span>
                  <span className="text-[#cbd5e1]">Laser</span>
                </span>
                <span className="text-sky-400">Beam</span>
              </div>
              <div className="flex items-center justify-between">
                <span className="flex items-center gap-1.5">
                  <span className="w-2.5 h-2 rounded bg-teal-500 border border-teal-400 inline-block"></span>
                  <span className="text-[#cbd5e1]">Shields</span>
                </span>
                <span className="text-teal-400">Bunker</span>
              </div>
            </div>
          )}

          {isAsteroids && (
            <div className="text-[10px] bg-[#0f172a]/70 p-2 rounded border border-[rgba(255,255,255,0.05)] space-y-1 font-mono">
              <div className="text-[9px] uppercase tracking-wider text-[#64748b] font-semibold border-b border-[rgba(255,255,255,0.05)] pb-0.5 mb-1">
                Color Key
              </div>
              <div className="flex items-center justify-between">
                <span className="flex items-center gap-1.5">
                  <svg className="w-2.5 h-2.5 text-sky-400 drop-shadow-[0_0_4px_rgba(56,189,248,0.8)] inline-block" viewBox="0 0 10 10" fill="currentColor">
                    <polygon points="5,1 9,9 5,7 1,9" />
                  </svg>
                  <span className="text-[#cbd5e1]">Spaceship</span>
                </span>
                <span className="text-sky-400 font-semibold">Player</span>
              </div>
              <div className="flex items-center justify-between">
                <span className="flex items-center gap-1.5">
                  <span className="w-2.5 h-2.5 rounded-full bg-slate-300 border border-slate-200 inline-block"></span>
                  <span className="text-[#cbd5e1]">Asteroid</span>
                </span>
                <span className="text-slate-300 font-semibold">Target</span>
              </div>
              <div className="flex items-center justify-between">
                <span className="flex items-center gap-1.5">
                  <span className="w-2.5 h-2.5 rounded-full bg-yellow-400 border border-yellow-300 shadow-[0_0_6px_rgba(250,204,21,0.8)] inline-block"></span>
                  <span className="text-[#cbd5e1]">Plasma</span>
                </span>
                <span className="text-yellow-400">Fire</span>
              </div>
            </div>
          )}

          {isAsterix && (
            <div className="text-[10px] bg-[#0f172a]/70 p-2 rounded border border-[rgba(255,255,255,0.05)] space-y-1 font-mono">
              <div className="text-[9px] uppercase tracking-wider text-[#64748b] font-semibold border-b border-[rgba(255,255,255,0.05)] pb-0.5 mb-1">
                Color Key
              </div>
              <div className="flex items-center justify-between">
                <span className="flex items-center gap-1.5">
                  <span className="w-2.5 h-2.5 bg-teal-400 rotate-45 inline-block"></span>
                  <span className="text-[#cbd5e1]">Hero</span>
                </span>
                <span className="text-teal-400 font-semibold">Player</span>
              </div>
              <div className="flex items-center justify-between">
                <span className="flex items-center gap-1.5">
                  <span className="w-2.5 h-2.5 rounded-full bg-rose-500 inline-block"></span>
                  <span className="text-[#cbd5e1]">Enemy</span>
                </span>
                <span className="text-rose-400">Fatal</span>
              </div>
              <div className="flex items-center justify-between">
                <span className="flex items-center gap-1.5">
                  <span className="w-2.5 h-2.5 rounded-full bg-yellow-400 inline-block"></span>
                  <span className="text-[#cbd5e1]">Gold</span>
                </span>
                <span className="text-yellow-400">+1</span>
              </div>
            </div>
          )}

          {isPacMan && (
            <div className="text-[10px] bg-[#0f172a]/70 p-2 rounded border border-[rgba(255,255,255,0.05)] space-y-1 font-mono">
              <div className="text-[9px] uppercase tracking-wider text-[#64748b] font-semibold border-b border-[rgba(255,255,255,0.05)] pb-0.5 mb-1">
                Color Key
              </div>
              <div className="flex items-center justify-between">
                <span className="flex items-center gap-1.5">
                  <span className="w-2.5 h-2.5 rounded-full bg-yellow-400 inline-block"></span>
                  <span className="text-[#cbd5e1]">Pac-Man</span>
                </span>
                <span className="text-yellow-400 font-semibold">Player</span>
              </div>
              <div className="flex items-center justify-between">
                <span className="flex items-center gap-1.5">
                  <span className="w-2.5 h-2.5 rounded-full bg-rose-500 inline-block"></span>
                  <span className="text-[#cbd5e1]">Ghost</span>
                </span>
                <span className="text-rose-400">Chase</span>
              </div>
              <div className="flex items-center justify-between">
                <span className="flex items-center gap-1.5">
                  <span className="w-2.5 h-2.5 rounded-full bg-sky-400 inline-block"></span>
                  <span className="text-[#cbd5e1]">Frightened</span>
                </span>
                <span className="text-sky-400">Edible</span>
              </div>
            </div>
          )}

          {isBreakout && (
            <div className="text-[10px] bg-[#0f172a]/70 p-2 rounded border border-[rgba(255,255,255,0.05)] space-y-1 font-mono">
              <div className="text-[9px] uppercase tracking-wider text-[#64748b] font-semibold border-b border-[rgba(255,255,255,0.05)] pb-0.5 mb-1">
                Color Key
              </div>
              <div className="flex items-center justify-between">
                <span className="flex items-center gap-1.5">
                  <span className="w-2.5 h-2 rounded bg-emerald-400 border border-emerald-300 inline-block"></span>
                  <span className="text-[#cbd5e1]">Paddle</span>
                </span>
                <span className="text-emerald-400 font-semibold">Player</span>
              </div>
              <div className="flex items-center justify-between">
                <span className="flex items-center gap-1.5">
                  <span className="w-2.5 h-2.5 rounded-full bg-white border border-slate-300 inline-block"></span>
                  <span className="text-[#cbd5e1]">Ball</span>
                </span>
                <span className="text-white font-semibold">Active</span>
              </div>
              <div className="flex items-center justify-between">
                <span className="flex items-center gap-1.5">
                  <span className="w-2.5 h-2 rounded bg-rose-400 border border-rose-300 inline-block"></span>
                  <span className="text-[#cbd5e1]">Bricks</span>
                </span>
                <span className="text-rose-400">Wall</span>
              </div>
            </div>
          )}
        </div>
      </div>

      <PolicyInspector telemetry={policyTelemetry} />
    </div>
  );
}
