import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { api } from "@/lib/api";

export function useMissions() {
  return useQuery({ queryKey: ["missions"], queryFn: api.getMissions });
}

export function useMission(id: string) {
  return useQuery({
    queryKey: ["missions", id],
    queryFn: () => api.getMission(id),
    enabled: !!id,
  });
}

export function usePendingApprovals(id: string) {
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
    mutationFn: (id: string) => api.runMission(id),
    onSuccess: (_data, id) => {
      qc.invalidateQueries({ queryKey: ["missions", id] });
      qc.invalidateQueries({ queryKey: ["missions"] });
    },
  });
}

export function useCancelMission() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (id: string) => api.cancelMission(id),
    onSuccess: (_data, id) => {
      qc.invalidateQueries({ queryKey: ["missions", id] });
      qc.invalidateQueries({ queryKey: ["missions"] });
    },
  });
}

export function useAutoApprove(missionId: string) {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (approvalId: string) => api.autoApprove(approvalId),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["approvals", missionId] });
      qc.invalidateQueries({ queryKey: ["missions", missionId] });
    },
  });
}

export function useResolveApproval(missionId: string) {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: ({
      approvalId,
      decision,
    }: {
      approvalId: string;
      decision: "approved" | "rejected";
    }) => api.resolveApproval(approvalId, decision),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["approvals", missionId] });
      qc.invalidateQueries({ queryKey: ["missions", missionId] });
    },
  });
}
