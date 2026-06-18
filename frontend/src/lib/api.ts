const BASE = "/api";

export interface Mission {
  id: string;
  goal: string;
  task_type: string;
  status: string;
  current_iteration: number;
  best_metric_value: string | null;
  best_metric_iteration: number | null;
  current_metric_value: string | null;
  target_metric: Record<string, number> | null;
  created_at: string;
  updated_at: string;
}

export interface ApprovalGate {
  id: string;
  mission_id: string;
  gate_type: string;
  status: string;
  payload: Record<string, unknown> | null;
  created_at: string;
}

export interface AutoApproveResult {
  gate_id: string;
  safe: boolean;
  reason: string;
  classifier: string;
  action: "approved" | "blocked";
}

export interface TelemetryEvent {
  type: string;           // "metric" | "backfill" | "backfill_complete" | "pivot"
  mission_id?: string;
  name?: string;
  value?: number;
  step?: number;
  iteration?: number;
  recorded_at?: string;
  reason?: string;
}

async function req<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...init,
  });
  if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
  return res.json() as Promise<T>;
}

export const api = {
  getMissions: () => req<Mission[]>("/missions"),
  getMission: (id: string) => req<Mission>(`/missions/${id}`),
  createMission: (goal: string, taskType?: string) =>
    req<Mission>("/missions", {
      method: "POST",
      body: JSON.stringify({ goal, task_type: taskType ?? "rl" }),
    }),
  runMission: (id: string) =>
    req<{ status: string }>(`/agent/missions/${id}/run`, { method: "POST" }),
  getPendingApprovals: (missionId: string) =>
    req<ApprovalGate[]>(`/approvals?pending_only=true`).then((gates) =>
      gates.filter((g) => g.mission_id === missionId)
    ),
  resolveApproval: (approvalId: string, decision: "approved" | "rejected") =>
    req<ApprovalGate>(
      `/approvals/${approvalId}/${decision === "approved" ? "approve" : "reject"}`,
      { method: "POST", body: JSON.stringify({}) }
    ),
  autoApprove: (approvalId: string) =>
    req<AutoApproveResult>(`/approvals/${approvalId}/auto-approve`, { method: "POST" }),
};
