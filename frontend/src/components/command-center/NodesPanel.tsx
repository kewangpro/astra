"use client";

import { useNodes } from "@/lib/hooks/useMissions";
import type { NodeStatus } from "@/lib/api";

function NodeCard({ node }: { node: NodeStatus }) {
  const color = node.alive ? "#4ade80" : "#f87171";

  return (
    <div
      className="rounded-lg p-3 flex items-center justify-between gap-4"
      style={{ background: "#1e293b", border: "1px solid rgba(255,255,255,0.05)" }}
    >
      <div className="flex items-center gap-2 min-w-0">
        <span
          className="inline-block w-1.5 h-1.5 rounded-full shrink-0"
          style={{ background: color }}
          title={node.alive ? "reachable" : "unreachable"}
        />
        <span className="text-[12px] text-[#e2e8f0] truncate">{node.host}</span>
        {node.is_local && (
          <span className="text-[9px] text-[#64748b] uppercase tracking-widest shrink-0">
            local
          </span>
        )}
      </div>
      <div className="flex items-center gap-4 shrink-0 text-[11px] text-[#94a3b8]">
        <span>
          {node.missions.length} mission{node.missions.length === 1 ? "" : "s"}
        </span>
        <span>
          {node.real_available_gb !== null
            ? `${node.real_available_gb.toFixed(1)} GB free`
            : "memory unknown"}
        </span>
      </div>
    </div>
  );
}

export function NodesPanel() {
  const { data: nodes, isLoading, error } = useNodes();

  // Fails quiet, not loud — this panel is a convenience view, not the
  // mission grid itself; a backend hiccup here shouldn't compete for
  // attention with the actual BACKEND_UNREACHABLE error MissionsGrid shows.
  if (isLoading || error || !nodes?.length) return null;

  return (
    <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
      {nodes.map((n) => (
        <NodeCard key={n.host} node={n} />
      ))}
    </div>
  );
}
