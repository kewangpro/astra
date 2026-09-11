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
  host: string | null;    // which node this mission's sandbox is/was running on
  last_checkpoint_path?: string | null;
  autonomy_mode?: string;
  created_at: string;
  updated_at: string;
  completed_at: string | null;  // stamped when the mission reaches COMPLETED/FAILED/STALLED
}

export interface NodeMission {
  mission_id: string;
  sandbox_id: string | null;
}

export interface NodeStatus {
  host: string;
  is_local: boolean;
  alive: boolean;
  real_available_gb: number | null;
  missions: NodeMission[];
}

export interface ApprovalGate {
  id: string;
  mission_id: string;
  gate_type: string;
  status: string;
  payload: Record<string, unknown> | null;
  reviewer_note: string | null;
  created_at: string;
  resolved_at: string | null;
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

export interface ModelRecord {
  id: string;
  name: string;
  domain: string;
  framework?: string | null;
  architecture?: string | null;
  weights_path?: string | null;
  checkpoint_path?: string | null;
  best_metric_name?: string | null;
  best_metric_value?: number | null;
  is_champion: boolean;
  extra_metadata: Record<string, unknown>;
  experiment_id?: string | null;
  created_at: string;
  updated_at: string;
}

export interface TournamentEntry {
  model_id: string;
  name: string;
  checkpoint_path: string;
  mean_score: number;
  std_score: number;
  min_score: number;
  max_score: number;
  win_rate: number;
  scores: number[];
  rank: number;
}

export interface TournamentResponse {
  env_id: string;
  episodes: number;
  leaderboard: TournamentEntry[];
  champion_id: string | null;
}

export interface Recipe {
  name: string;
  filename: string;
  domain?: string | null;
  version?: string | null;
  description?: string | null;
  created_at?: string | null;
  content: Record<string, unknown>;
}

export interface RecipeRecord {
  id: string;
  name: string;
  version: string;
  domain: string;
  task_type: string;
  description?: string | null;
  hyperparameters: Record<string, unknown>;
  curriculum?: Record<string, unknown> | null;
  reward_shaping?: Record<string, unknown> | null;
  full_content: Record<string, unknown>;
  mission_id?: string | null;
  parent_recipe_id?: string | null;
  score?: number | null;
  target_metric?: Record<string, number> | null;
  generation: number;
  consecutive_wins: number;
  is_golden: boolean;
  created_at: string;
  updated_at: string;
}

export interface RecipeDispatchResponse {
  mission_id: string;
  status: string;
  recipe: string;
  task_type: string;
  goal: string;
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
  getNodes: () => req<NodeStatus[]>("/nodes"),
  getMission: (id: string) => req<Mission>(`/missions/${id}`),
  createMission: (goal: string, taskType?: string) =>
    req<Mission>("/missions", {
      method: "POST",
      body: JSON.stringify({ goal, ...(taskType ? { task_type: taskType } : {}) }),
    }),
  runMission: (id: string) =>
    req<{ status: string }>(`/agent/missions/${id}/run`, { method: "POST" }),
  cancelMission: (id: string) =>
    req<{ status: string }>(`/agent/missions/${id}/cancel`, { method: "POST" }),
  getPendingApprovals: (missionId: string) =>
    req<ApprovalGate[]>(`/approvals?pending_only=true&mission_id=${missionId}`),
  getApprovalHistory: (missionId: string) =>
    req<ApprovalGate[]>(`/approvals?pending_only=false&mission_id=${missionId}`),
  resolveApproval: (approvalId: string, decision: "approved" | "rejected") =>
    req<ApprovalGate>(
      `/approvals/${approvalId}/${decision === "approved" ? "approve" : "reject"}`,
      { method: "POST", body: JSON.stringify({}) }
    ),
  autoApprove: (approvalId: string) =>
    req<AutoApproveResult>(`/approvals/${approvalId}/auto-approve`, { method: "POST" }),

  // Model Registry & Tournament
  getModels: (domain?: string, championOnly?: boolean) => {
    const params = new URLSearchParams();
    if (domain) params.set("domain", domain);
    if (championOnly) params.set("champion_only", "true");
    const qs = params.toString();
    return req<ModelRecord[]>(`/registry/models${qs ? `?${qs}` : ""}`);
  },
  getModel: (id: string) => req<ModelRecord>(`/registry/models/${id}`),
  updateModel: (id: string, payload: Partial<ModelRecord>) =>
    req<ModelRecord>(`/registry/models/${id}`, {
      method: "PATCH",
      body: JSON.stringify(payload),
    }),
  deleteModel: (id: string) =>
    req<void>(`/registry/models/${id}`, { method: "DELETE" }),
  runTournament: (envId: string, modelIds?: string[], nEpisodes = 5, updateChampion = true) =>
    req<TournamentResponse>("/registry/tournament", {
      method: "POST",
      body: JSON.stringify({
        env_id: envId,
        model_ids: modelIds && modelIds.length > 0 ? modelIds : null,
        n_episodes: nEpisodes,
        update_champion: updateChampion,
      }),
    }),

  // Recipes & Lineage
  getRecipes: () => req<Recipe[]>("/recipes"),
  getDbRecipes: (domain?: string, goldenOnly?: boolean) => {
    const params = new URLSearchParams();
    if (domain) params.set("domain", domain);
    if (goldenOnly) params.set("golden_only", "true");
    const qs = params.toString();
    return req<RecipeRecord[]>(`/recipes/db${qs ? `?${qs}` : ""}`);
  },
  getRecipe: (name: string) => req<Recipe>(`/recipes/${name}`),
  getRecipeLineage: (id: string) => req<RecipeRecord[]>(`/recipes/${id}/lineage`),
  evolveRecipe: (id: string) =>
    req<{ child: RecipeRecord; parent_id: string }>(`/recipes/${id}/evolve`, {
      method: "POST",
    }),
  dispatchRecipe: (recipeName: string) =>
    req<RecipeDispatchResponse>(`/recipes/${recipeName}/dispatch`, {
      method: "POST",
    }),
};

