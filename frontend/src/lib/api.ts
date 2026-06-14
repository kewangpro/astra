const BASE = "/api";

export interface Mission {
  id: number;
  goal: string;
  status: string;
  iteration: number;
  best_metric: number | null;
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
  getMission: (id: number) => req<Mission>(`/missions/${id}`),
  createMission: (goal: string, domain?: string) =>
    req<Mission>("/missions", {
      method: "POST",
      body: JSON.stringify({ goal, domain: domain ?? "general" }),
    }),
  runMission: (id: number) =>
    req<{ status: string }>(`/agent/missions/${id}/run`, { method: "POST" }),
  getMetrics: (id: number) => req<Metric[]>(`/missions/${id}/metrics`),
  getPendingApprovals: (id: number) =>
    req<ApprovalGate[]>(`/approvals/missions/${id}/pending`),
  resolveApproval: (approvalId: number, decision: "approved" | "rejected") =>
    req<ApprovalGate>(`/approvals/${approvalId}`, {
      method: "PATCH",
      body: JSON.stringify({ status: decision }),
    }),
};
