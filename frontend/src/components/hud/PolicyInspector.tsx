"use client";

import React, { useState } from "react";
import { ChevronDown, ChevronUp, Cpu, Activity, Zap } from "lucide-react";

export interface PolicyTelemetry {
  q_values?: Record<string, number>;
  action_probs?: Record<string, number>;
  entropy?: number | null;
  selected_action?: string;
}

interface Props {
  telemetry?: PolicyTelemetry | null;
}

export function PolicyInspector({ telemetry }: Props) {
  const [expanded, setExpanded] = useState(true);

  if (!telemetry || (!telemetry.q_values && !telemetry.action_probs && !telemetry.selected_action)) {
    return null;
  }

  const { q_values, action_probs, entropy, selected_action } = telemetry;
  const actions = Object.keys(action_probs || q_values || {});

  // Determine entropy level
  let entropyBadge = "Balanced";
  let entropyColor = "text-teal-400 bg-teal-950/60 border-teal-800/60";
  if (entropy !== undefined && entropy !== null) {
    if (entropy > 1.1) {
      entropyBadge = "High Exploration";
      entropyColor = "text-amber-400 bg-amber-950/60 border-amber-800/60";
    } else if (entropy < 0.4) {
      entropyBadge = "High Certainty";
      entropyColor = "text-emerald-400 bg-emerald-950/60 border-emerald-800/60";
    }
  }

  return (
    <div className="mt-3 border border-[rgba(20,184,166,0.15)] bg-[#0f172a]/80 rounded-md p-3 text-xs">
      <div
        className="flex items-center justify-between cursor-pointer select-none"
        onClick={() => setExpanded(!expanded)}
      >
        <div className="flex items-center gap-2">
          <Cpu className="w-3.5 h-3.5 text-teal-400" />
          <span className="font-semibold text-slate-200 uppercase tracking-wider text-[10px]">
            Policy Inspector & Explainability
          </span>
          {selected_action && (
            <span className="font-mono px-1.5 py-0.5 rounded bg-teal-500/20 text-teal-300 border border-teal-500/40 text-[10px]">
              Action: {selected_action}
            </span>
          )}
        </div>
        <div className="flex items-center gap-2">
          {entropy !== undefined && entropy !== null && (
            <span
              className={`px-2 py-0.5 rounded-full border text-[10px] font-mono flex items-center gap-1 ${entropyColor}`}
            >
              <Activity className="w-2.5 h-2.5" />
              Entropy: {entropy.toFixed(2)} ({entropyBadge})
            </span>
          )}
          {expanded ? (
            <ChevronUp className="w-3.5 h-3.5 text-slate-400" />
          ) : (
            <ChevronDown className="w-3.5 h-3.5 text-slate-400" />
          )}
        </div>
      </div>

      {expanded && actions.length > 0 && (
        <div className="mt-3 space-y-1.5 pt-2 border-t border-slate-800">
          <div className="text-[10px] text-slate-400 flex justify-between font-mono pb-1">
            <span>Candidate Action</span>
            <span>{action_probs ? "Confidence / Prob" : "Q-Value / Score"}</span>
          </div>
          {actions.map((act) => {
            const isSelected = act === selected_action;
            const prob = action_probs ? action_probs[act] ?? 0 : null;
            const qVal = q_values ? q_values[act] ?? 0 : null;
            const percentage = prob !== null ? Math.round(prob * 100) : 50;

            return (
              <div key={act} className="space-y-0.5">
                <div className="flex justify-between items-center text-[11px] font-mono">
                  <span
                    className={`flex items-center gap-1.5 ${
                      isSelected ? "text-emerald-400 font-bold" : "text-slate-300"
                    }`}
                  >
                    {isSelected && <Zap className="w-3 h-3 text-emerald-400 fill-emerald-400" />}
                    {act}
                  </span>
                  <span className={isSelected ? "text-emerald-300" : "text-slate-400"}>
                    {prob !== null ? `${percentage}% (${prob.toFixed(3)})` : ""}
                    {qVal !== null && (prob !== null ? " · " : "")}
                    {qVal !== null ? `Q: ${qVal.toFixed(2)}` : ""}
                  </span>
                </div>
                <div className="w-full bg-slate-800 rounded-full h-1.5 overflow-hidden">
                  <div
                    className={`h-full rounded-full transition-all duration-150 ${
                      isSelected
                        ? "bg-gradient-to-r from-teal-500 to-emerald-400"
                        : "bg-slate-600/50"
                    }`}
                    style={{ width: `${Math.max(4, Math.min(100, percentage))}%` }}
                  />
                </div>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}
