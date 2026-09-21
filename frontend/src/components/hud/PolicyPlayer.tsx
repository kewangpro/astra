"use client";

import { SnakePlayer } from "./SnakePlayer";
import { TetrisPlayer } from "./TetrisPlayer";
import { Game2048Player } from "./Game2048Player";
import { MinAtarPlayer } from "./MinAtarPlayer";
import { envIdFromDomain, isPlayableEnv } from "@/lib/playWs";

type Props = {
  modelId?: string;
  missionId?: string;
  envId: string;
};

export function PolicyPlayer({ modelId, missionId, envId }: Props) {
  const resolved = envIdFromDomain(envId);
  if (!isPlayableEnv(resolved)) {
    return (
      <div className="border border-dashed border-[#334155] rounded-lg p-8 text-center text-xs text-[#64748b]">
        No live viewer for {envId}.
      </div>
    );
  }
  const lower = resolved.toLowerCase();
  if (lower.includes("tetris")) {
    return <TetrisPlayer modelId={modelId} missionId={missionId} envId="Tetris-v0" />;
  }
  if (lower.includes("2048")) {
    return <Game2048Player modelId={modelId} missionId={missionId} envId="Game2048-v0" />;
  }
  if (
    lower.includes("minatar") ||
    lower.includes("breakout") ||
    lower.includes("seaquest") ||
    lower.includes("freeway") ||
    lower.includes("asterix") ||
    lower.includes("asteroid") ||
    lower.includes("pacman") ||
    lower.includes("pac-man") ||
    lower.includes("space")
  ) {
    return <MinAtarPlayer modelId={modelId} missionId={missionId} envId={resolved} />;
  }
  return <SnakePlayer modelId={modelId} missionId={missionId} envId="Snake-v0" />;
}
