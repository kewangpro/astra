"use client";

import { useState } from "react";
import { usePendingApprovals, useResolveApproval, useAutoApprove } from "@/lib/hooks/useMissions";
import type { AutoApproveResult } from "@/lib/api";

interface Props {
  missionId: string;
}

const GATE_LABELS: Record<string, string> = {
  execute_code: "Execute Code",
  resource_allocation: "Resource Allocation",
  deploy_model: "Deploy Model",
};

export function ApprovalPanel({ missionId }: Props) {
  const { data: approvals } = usePendingApprovals(missionId);
  const resolve = useResolveApproval(missionId);
  const autoApprove = useAutoApprove(missionId);
  const [verdicts, setVerdicts] = useState<Record<string, AutoApproveResult>>({});

  if (!approvals?.length) return null;

  const handleAutoApprove = async (gateId: string) => {
    const result = await autoApprove.mutateAsync(gateId);
    if (result.action === "blocked") {
      setVerdicts((v) => ({ ...v, [gateId]: result }));
    }
  };

  return (
    <div className="bg-[#1e293b] border border-[#fbbf24]/30 rounded-lg overflow-hidden animate-slide-in">
      <div className="flex items-center gap-2 px-4 py-2.5 bg-[#fbbf24]/10 border-b border-[#fbbf24]/20">
        <span className="text-[#fbbf24] text-sm">▲</span>
        <span className="text-[#fbbf24] text-xs font-semibold tracking-widest uppercase">
          {approvals.length} Approval{approvals.length !== 1 ? "s" : ""} Required
        </span>
      </div>

      <div className="divide-y divide-[rgba(20,184,166,0.08)]">
        {approvals.map((gate) => {
          const code =
            typeof gate.payload?.code === "string" ? gate.payload.code : null;
          const resources =
            typeof gate.payload?.resources === "object" && gate.payload.resources
              ? gate.payload.resources
              : null;
          const verdict = verdicts[gate.id];
          const isAutoLoading = autoApprove.isPending;

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
