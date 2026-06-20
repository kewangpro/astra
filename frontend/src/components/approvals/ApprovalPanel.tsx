"use client";

import { useState, useEffect, useRef } from "react";
import { usePendingApprovals, useResolveApproval, useAutoApprove } from "@/lib/hooks/useMissions";
import type { AutoApproveResult } from "@/lib/api";

interface Props {
  missionId: string;
  autoApproveMode?: boolean;
  onAutoApproveModeChange?: (on: boolean) => void;
}

const GATE_LABELS: Record<string, string> = {
  execute_code: "Execute Code",
  resource_allocation: "Resource Allocation",
  deploy_model: "Deploy Model",
};

export function ApprovalPanel({ missionId, autoApproveMode = false, onAutoApproveModeChange }: Props) {
  const { data: approvals } = usePendingApprovals(missionId);
  const resolve = useResolveApproval(missionId);
  const autoApprove = useAutoApprove(missionId);
  const [verdicts, setVerdicts] = useState<Record<string, AutoApproveResult>>({});
  const [classifyingGates, setClassifyingGates] = useState<Set<string>>(new Set());
  // Gates that were just auto-approved — keep visible briefly so user can see the result
  const [approvedGates, setApprovedGates] = useState<Map<string, AutoApproveResult>>(new Map());
  const triggeredRef = useRef<Set<string>>(new Set());

  const runAutoApprove = (gateId: string) => {
    setClassifyingGates((s) => new Set(s).add(gateId));
    autoApprove.mutateAsync(gateId).then((result) => {
      if (result.action === "blocked") {
        setVerdicts((v) => ({ ...v, [gateId]: result }));
      } else {
        // Show "Auto-approved" flash for 2s before letting the panel disappear
        setApprovedGates((m) => new Map(m).set(gateId, result));
        setTimeout(() => {
          setApprovedGates((m) => { const n = new Map(m); n.delete(gateId); return n; });
        }, 2000);
      }
    }).catch(() => {}).finally(() => {
      setClassifyingGates((s) => { const n = new Set(s); n.delete(gateId); return n; });
    });
  };

  // Auto-trigger classification for execute_code gates when mode is on
  useEffect(() => {
    if (!autoApproveMode || !approvals?.length) return;
    for (const gate of approvals) {
      if (gate.gate_type === "execute_code" && !triggeredRef.current.has(gate.id)) {
        triggeredRef.current.add(gate.id);
        runAutoApprove(gate.id);
      }
    }
  }, [autoApproveMode, approvals]);

  const pendingGates = approvals ?? [];
  if (!pendingGates.length && !approvedGates.size) return null;

  const handleAutoApprove = (gateId: string) => {
    onAutoApproveModeChange?.(true);
    runAutoApprove(gateId);
  };

  return (
    <div className="bg-[#1e293b] border border-[#fbbf24]/30 rounded-lg overflow-hidden animate-slide-in">
      <div className="flex items-center gap-2 px-4 py-2.5 bg-[#fbbf24]/10 border-b border-[#fbbf24]/20">
        <span className="text-[#fbbf24] text-sm">▲</span>
        <span className="text-[#fbbf24] text-xs font-semibold tracking-widest uppercase">
          {pendingGates.length > 0
            ? `${pendingGates.length} Approval${pendingGates.length !== 1 ? "s" : ""} Required`
            : "Approval Gate"}
        </span>
      </div>

      {/* Recently auto-approved — flash green briefly */}
      {Array.from(approvedGates.entries()).map(([gateId, result]) => (
        <div key={gateId} className="px-4 py-3 flex items-center gap-2 bg-[#4ade80]/5 border-b border-[#4ade80]/20">
          <span className="text-[#4ade80] text-xs">✓</span>
          <span className="text-[#4ade80] text-xs font-semibold">Auto-approved</span>
          <span className="text-[10px] text-[#64748b] ml-1">via {result.classifier}</span>
          <span className="text-[10px] text-[#94a3b8] ml-auto">#{gateId.slice(0, 8)}</span>
        </div>
      ))}

      <div className="divide-y divide-[rgba(20,184,166,0.08)]">
        {pendingGates.map((gate) => {
          const code =
            typeof gate.payload?.code === "string" ? gate.payload.code : null;
          const resources =
            typeof gate.payload?.resources === "object" && gate.payload.resources
              ? gate.payload.resources
              : null;
          const verdict = verdicts[gate.id];
          const isAutoLoading = classifyingGates.has(gate.id);

          return (
            <div key={gate.id} className="p-4">
              <div className="flex items-center justify-between mb-3">
                <span className="text-xs text-[#fbbf24] tracking-widest uppercase">
                  {GATE_LABELS[gate.gate_type] ?? gate.gate_type}
                </span>
                <span className="text-[10px] text-[#94a3b8]">#{gate.id}</span>
              </div>

              {code && (
                <pre className="bg-[#263347] border border-[rgba(20,184,166,0.1)] rounded p-3
                                text-[11px] text-[#94a3b8] overflow-x-auto max-h-48 mb-3 whitespace-pre-wrap">
                  {code}
                </pre>
              )}

              {resources && !code && (
                <div className="bg-[#263347] rounded p-3 mb-3 text-[11px] text-[#94a3b8] space-y-1">
                  {Object.entries(resources as Record<string, unknown>).map(([k, v]) => (
                    <div key={k} className="flex justify-between">
                      <span className="text-[#94a3b8]">{k}</span>
                      <span>{String(v)}</span>
                    </div>
                  ))}
                </div>
              )}

              {verdict && (
                <div className="mb-3 px-3 py-2 rounded bg-[#f87171]/10 border border-[#f87171]/20">
                  <div className="text-[10px] text-[#f87171] font-semibold mb-0.5">
                    ⚠ Unsafe — manual review required
                  </div>
                  <div className="text-[10px] text-[#94a3b8]">{verdict.reason}</div>
                  <div className="text-[9px] text-[#64748b] mt-0.5">classifier: {verdict.classifier}</div>
                </div>
              )}

              <div className="flex gap-2">
                <button
                  onClick={() =>
                    resolve.mutate({ approvalId: gate.id, decision: "approved" })
                  }
                  disabled={resolve.isPending || isAutoLoading}
                  aria-label={`Approve gate #${gate.id}`}
                  className="flex-1 py-2 rounded border border-[#4ade80]/30 text-[#4ade80] text-xs
                             hover:bg-[#4ade80]/10 disabled:opacity-40 transition-colors"
                >
                  Approve
                </button>
                {gate.gate_type === "execute_code" && (
                  <button
                    onClick={() => handleAutoApprove(gate.id)}
                    disabled={resolve.isPending || isAutoLoading}
                    aria-label={`Auto-approve gate #${gate.id}`}
                    className="flex-1 py-2 rounded border border-[#38bdf8]/30 text-[#38bdf8] text-xs
                               hover:bg-[#38bdf8]/10 disabled:opacity-40 transition-colors"
                  >
                    {isAutoLoading ? "Classifying…" : "Auto-Approve"}
                  </button>
                )}
                <button
                  onClick={() =>
                    resolve.mutate({ approvalId: gate.id, decision: "rejected" })
                  }
                  disabled={resolve.isPending || isAutoLoading}
                  aria-label={`Reject gate #${gate.id}`}
                  className="flex-1 py-2 rounded border border-[#f87171]/30 text-[#f87171] text-xs
                             hover:bg-[#f87171]/10 disabled:opacity-40 transition-colors"
                >
                  Reject
                </button>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
