const WS_BASE =
  typeof window !== "undefined"
    ? `ws://${window.location.hostname}:8200`
    : "ws://localhost:8200";

export type PlaySource = {
  modelId?: string;
  missionId?: string;
};

export function policyPlayUrl(
  source: PlaySource,
  envId: string,
  fps: number,
): string {
  const q = `env_id=${encodeURIComponent(envId)}&fps=${fps}`;
  if (source.modelId) {
    return `${WS_BASE}/ws/models/${source.modelId}/play?${q}`;
  }
  if (!source.missionId) {
    throw new Error("policyPlayUrl requires modelId or missionId");
  }
  return `${WS_BASE}/ws/missions/${source.missionId}/play?${q}`;
}

export function envIdFromDomain(domain: string): string {
  const d = domain.toLowerCase();
  if (d.includes("seaquest")) return "MinAtar-Seaquest-v0";
  if (d.includes("freeway")) return "MinAtar-Freeway-v0";
  if (d.includes("asteroid")) return "MinAtar-Asteroids-v0";
  if (d.includes("space")) return "MinAtar-SpaceInvaders-v0";
  if (d.includes("breakout") || d.includes("minatar")) return "MinAtar-Breakout-v0";
  if (d.includes("tetris")) return "Tetris-v0";
  if (d.includes("2048")) return "Game2048-v0";
  if (d.includes("snake")) return "Snake-v0";
  return domain;
}

export function isPlayableEnv(envId: string): boolean {
  const d = envId.toLowerCase();
  return (
    d.includes("snake") ||
    d.includes("tetris") ||
    d.includes("2048") ||
    d.includes("minatar") ||
    d.includes("breakout") ||
    d.includes("seaquest") ||
    d.includes("freeway") ||
    d.includes("asteroid") ||
    d.includes("space")
  );
}
