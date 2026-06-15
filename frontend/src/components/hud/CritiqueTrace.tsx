"use client";

import type { TelemetryEvent } from "@/lib/api";

interface Props {
  events: TelemetryEvent[];
}

interface CritiqueData {
  overall_score: number;
  approved: boolean;
  concerns: string[];
  feedback: string;
  rubric_scores: Record<string, number>;
  revision: number;
}

function ScorePip({ score }: { score: number }) {
  const color =
    score >= 8 ? "#4ade80" : score >= 6 ? "#fbbf24" : "#f87171";
  return (
    <span
      className="inline-block w-2 h-2 rounded-full mr-1"
      style={{ background: color, boxShadow: `0 0 4px ${color}` }}
    />
  );
}

function CritiqueCard({ event }: { event: TelemetryEvent }) {
  const data = (event as TelemetryEvent & { critique?: CritiqueData }).critique;
  if (!data) return null;

  const approved = data.approved;
  const borderColor = approved ? "rgba(74,222,128,0.2)" : "rgba(251,191,36,0.2)";
  const labelColor = approved ? "#4ade80" : "#fbbf24";
  const label = approved ? "APPROVED" : data.revision > 0 ? `REVISION ${data.revision}` : "FLAGGED";

  return (
    <div
      className="rounded p-3 space-y-2"
      style={{ background: "#1e293b", border: `1px solid ${borderColor}` }}
    >
      {/* Header row */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <ScorePip score={data.overall_score} />
          <span className="text-[11px] font-medium" style={{ color: labelColor }}>
            {label}
          </span>
        </div>
        <span className="text-[11px] font-mono" style={{ color: labelColor }}>
          {data.overall_score.toFixed(1)} / 10
        </span>
      </div>

      {/* Rubric scores */}
      {data.rubric_scores && (
        <div className="flex gap-3">
          {Object.entries(data.rubric_scores).map(([dim, score]) => (
            <div key={dim} className="flex flex-col items-center gap-0.5">
              <span className="text-[9px] text-[#64748b] uppercase tracking-wider">
                {dim.replace(/_/g, " ")}
              </span>
              <span className="text-[11px] font-mono" style={{ color: score >= 8 ? "#4ade80" : score >= 6 ? "#fbbf24" : "#f87171" }}>
                {(score as number).toFixed(0)}
              </span>
            </div>
          ))}
        </div>
      )}

      {/* Concerns */}
      {data.concerns.length > 0 && (
        <ul className="space-y-0.5">
          {data.concerns.map((c, i) => (
            <li key={i} className="text-[10px] text-[#94a3b8] flex gap-1.5">
              <span className="text-[#fbbf24] shrink-0">▸</span>
              {c}
            </li>
          ))}
        </ul>
      )}

      {/* Feedback */}
      {data.feedback && (
        <p className="text-[10px] text-[#64748b] italic leading-relaxed">
          {data.feedback}
        </p>
      )}
    </div>
  );
}

export function CritiqueTrace({ events }: Props) {
  const critiques = events.filter((e) => e.type === "critique");
  if (critiques.length === 0) return null;

  return (
    <div
      className="rounded-lg flex flex-col"
      style={{
        background: "#0f172a",
        border: "1px solid rgba(167,139,250,0.15)",
      }}
    >
      {/* Header */}
      <div className="px-4 py-2.5 border-b border-[rgba(167,139,250,0.08)] shrink-0 flex items-center gap-2">
        <span className="text-[10px] text-[#64748b] tracking-widest uppercase">
          critic trace
        </span>
        <span className="text-[10px] text-[#64748b]">({critiques.length})</span>
      </div>

      {/* Cards */}
      <div className="flex-1 overflow-y-auto p-3 space-y-2">
        {critiques.map((e, i) => (
          <CritiqueCard key={i} event={e} />
        ))}
      </div>
    </div>
  );
}
