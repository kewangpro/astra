// The API serialises timestamps as naive ISO (no trailing "Z" or offset) but
// stores them in UTC. `new Date("2026-09-03T01:34:26")` would be read as *local*
// time, so every displayed timestamp would be off by the viewer's UTC offset.
// Normalise here: append "Z" unless the string already carries a zone marker.
export function parseTs(iso: string): Date {
  return new Date(/[zZ]|[+-]\d\d:?\d\d$/.test(iso) ? iso : iso + "Z");
}

export function fmtTs(iso: string): string {
  return parseTs(iso).toLocaleString([], {
    month: "numeric",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
}

export function formatRelativeTime(iso: string): string {
  try {
    const d = parseTs(iso);
    const now = Date.now();
    const diffSec = Math.floor((now - d.getTime()) / 1000);
    if (diffSec < 60) return "just now";
    const diffMin = Math.floor(diffSec / 60);
    if (diffMin < 60) return `${diffMin}m ago`;
    const diffHour = Math.floor(diffMin / 60);
    if (diffHour < 24) return `${diffHour}h ago`;
    const diffDay = Math.floor(diffHour / 24);
    if (diffDay < 7) return `${diffDay}d ago`;
    return d.toLocaleDateString([], { month: "short", day: "numeric" });
  } catch {
    return iso;
  }
}

export function formatDuration(startIso: string, endIso?: string | null): string {
  if (!startIso || !endIso) return "—";
  try {
    const start = parseTs(startIso).getTime();
    const end = parseTs(endIso).getTime();
    const diffSec = Math.max(0, Math.floor((end - start) / 1000));
    if (diffSec < 60) return `${diffSec}s`;
    const diffMin = Math.floor(diffSec / 60);
    const remSec = diffSec % 60;
    if (diffMin < 60) return `${diffMin}m ${remSec}s`;
    const diffHour = Math.floor(diffMin / 60);
    const remMin = diffMin % 60;
    return `${diffHour}h ${remMin}m`;
  } catch {
    return "—";
  }
}

