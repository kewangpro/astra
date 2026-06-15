const BASE = "/api";

export interface Mission {
  id: number;
  goal: string;
  task_type: string;
  status: string;
  current_iteration: number;
  best_metric_value: string | null;
  created_at: string;
  updated_at: string;
}

export interface Metric {
  iteration: number;
  metric_name: string;
  metric_value: number;
  timestamp: string;
}

export interface ApprovalGate {
  id: number;
  mission_id: number;
  gate_type: string;
  status: string;
  payload: Record<string, unknown>;
  created_at: string;
}

export interface TelemetryEvent {
  ts: string;
  event: string;
  data: Record<string, unknown>;
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
  getMetrics: (id: string) => req<Metric[]>(`/missions/${id}/metrics`),
  getPendingApprovals: (id: string) =>
    req<ApprovalGate[]>(`/approvals/missions/${id}/pending`),
  resolveApproval: (approvalId: number, decision: "approved" | "rejected") =>
    req<ApprovalGate>(`/approvals/${approvalId}`, {
      method: "PATCH",
      body: JSON.stringify({ status: decision }),
    }),
};
