import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { api } from "@/lib/api";

export function useMissions() {
  return useQuery({ queryKey: ["missions"], queryFn: api.getMissions });
}

export function useMission(id: number) {
  return useQuery({
    queryKey: ["missions", id],
    queryFn: () => api.getMission(id),
    enabled: !!id,
  });
}

export function useMetrics(id: number) {
  return useQuery({
    queryKey: ["metrics", id],
    queryFn: () => api.getMetrics(id),
    enabled: !!id,
    refetchInterval: 3000,
  });
}

export function usePendingApprovals(id: number) {
  return useQuery({
    queryKey: ["approvals", id],
    queryFn: () => api.getPendingApprovals(id),
    enabled: !!id,
    refetchInterval: 3000,
  });
}

export function useCreateMission() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: ({ goal, taskType }: { goal: string; taskType?: string }) =>
      api.createMission(goal, taskType),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["missions"] }),
  });
}

export function useRunMission() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (id: number) => api.runMission(id),
    onSuccess: (_data, id) => {
      qc.invalidateQueries({ queryKey: ["missions", id] });
      qc.invalidateQueries({ queryKey: ["missions"] });
    },
  });
}

export function useResolveApproval(missionId: number) {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: ({
      approvalId,
      decision,
    }: {
      approvalId: number;
      decision: "approved" | "rejected";
    }) => api.resolveApproval(approvalId, decision),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["approvals", missionId] });
      qc.invalidateQueries({ queryKey: ["missions", missionId] });
    },
  });
}
