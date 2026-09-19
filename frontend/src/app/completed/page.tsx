"use client";

import { useState, useMemo, useEffect } from "react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import {
  CheckCircle2,
  Target,
  TrendingUp,
  Clock,
  Server,
  ArrowUpRight,
  FileText,
  Search,
  X,
  ShieldCheck,
  Award,
  Sparkles,
  Layers,
  ChevronRight,
  ExternalLink,
} from "lucide-react";
import { useMissions, useMissionManifest } from "@/lib/hooks/useMissions";
import { api, Mission, ManifestRecord } from "@/lib/api";
import { parseTs, fmtTs, formatRelativeTime, formatDuration } from "@/lib/date";

const DOMAIN_TABS = [
  { id: "all", label: "All Completed" },
  { id: "minatar", label: "MinAtar" },
  { id: "snake", label: "Snake" },
  { id: "tetris", label: "Tetris" },
  { id: "2048", label: "2048" },
  { id: "llm", label: "LLM & Reasoning" },
  { id: "agentgym", label: "AgentGym" },
  { id: "classic", label: "Classic Control & ML" },
];

function isLowerBetterMetric(key: string): boolean {
  const k = key.toLowerCase();
  return k.includes("loss") || k.includes("perplexity") || k.includes("error") || k === "cost";
}

function getTargetProgress(m: Mission): { targetKey: string; targetVal: number; progressPct: number; isPassed: boolean } | null {
  if (!m.target_metric) return null;
  const entries = Object.entries(m.target_metric);
  if (entries.length === 0) return null;
  const [key, targetVal] = entries[0];
  if (typeof targetVal !== "number" || targetVal <= 0) return null;

  const currentNum = m.best_metric_value ? parseFloat(m.best_metric_value) : null;
  if (currentNum === null || Number.isNaN(currentNum)) return null;

  let pct: number;
  let isPassed = false;
  if (isLowerBetterMetric(key)) {
    if (currentNum <= targetVal) {
      pct = 100;
      isPassed = true;
    } else if (currentNum <= 0) {
      pct = 100;
      isPassed = true;
    } else {
      pct = Math.min(100, Math.max(0, (targetVal / currentNum) * 100));
      isPassed = currentNum <= targetVal;
    }
  } else {
    pct = (currentNum / targetVal) * 100;
    isPassed = currentNum >= targetVal;
  }
  return { targetKey: key, targetVal, progressPct: Math.round(pct), isPassed };
}

function ManifestModal({
  mission,
  onClose,
}: {
  mission: Mission;
  onClose: () => void;
}) {
  const router = useRouter();
  const [manifest, setManifest] = useState<ManifestRecord | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [activeView, setActiveView] = useState<"requirements" | "config">("requirements");

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    api
      .getManifest(mission.id)
      .then((data) => {
        if (!cancelled) setManifest(data);
      })
      .catch(() => {
        if (!cancelled) setManifest(null);
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [mission.id]);

  const targetProgress = getTargetProgress(mission);
  const durationStr = formatDuration(mission.created_at, mission.completed_at);

  return (
    <div className="fixed inset-0 z-50 bg-black/80 backdrop-blur-sm flex items-center justify-center p-4">
      <div className="bg-[#1e293b] border border-[#334155] rounded-xl max-w-2xl w-full max-h-[90vh] flex flex-col shadow-2xl overflow-hidden">
        {/* Modal Header */}
        <div className="p-4 sm:p-5 border-b border-[#334155] flex items-start justify-between gap-4 bg-[#0f172a]/60">
          <div>
            <div className="flex items-center gap-2 mb-1.5 flex-wrap">
              <span className="text-[11px] font-mono text-[#64748b] bg-[#0f172a] border border-[#334155] px-2 py-0.5 rounded">
                #{mission.id.slice(0, 8)}
              </span>
              <span className="text-[10px] px-2 py-0.5 rounded bg-[#4ade80]/15 text-[#4ade80] border border-[#4ade80]/30 font-medium uppercase tracking-wider">
                COMPLETED
              </span>
              {mission.task_type && (
                <span className="text-[10px] px-2 py-0.5 rounded bg-[#334155] text-[#cbd5e1] font-mono uppercase">
                  {mission.task_type}
                </span>
              )}
              {mission.host && (
                <span className="text-[10px] text-[#94a3b8] flex items-center gap-1 font-mono">
                  <Server className="w-3 h-3 text-[#64748b]" />
                  {mission.host}
                </span>
              )}
            </div>
            <h3 className="text-sm sm:text-base font-semibold text-[#e2e8f0] leading-snug">
              {mission.goal}
            </h3>
          </div>
          <button
            onClick={onClose}
            className="text-xs text-[#94a3b8] hover:text-[#e2e8f0] p-1.5 rounded-lg hover:bg-[#334155] transition-colors"
          >
            ✕
          </button>
        </div>

        {/* View Switcher Tabs */}
        <div className="px-5 pt-3 border-b border-[#334155] flex gap-4 bg-[#0f172a]/30 text-xs">
          <button
            onClick={() => setActiveView("requirements")}
            className={`pb-2.5 font-medium transition-colors border-b-2 ${
              activeView === "requirements"
                ? "border-[#14b8a6] text-[#14b8a6]"
                : "border-transparent text-[#94a3b8] hover:text-[#e2e8f0]"
            }`}
          >
            Requirement Manifest & Checks
          </button>
          <button
            onClick={() => setActiveView("config")}
            className={`pb-2.5 font-medium transition-colors border-b-2 ${
              activeView === "config"
                ? "border-[#14b8a6] text-[#14b8a6]"
                : "border-transparent text-[#94a3b8] hover:text-[#e2e8f0]"
            }`}
          >
            Raw Config & Metadata
          </button>
        </div>

        {/* Modal Body */}
        <div className="p-5 overflow-y-auto flex-1 space-y-4">
          {/* Quick Metrics Grid */}
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
            <div className="bg-[#0f172a] border border-[#334155] rounded-lg p-3">
              <span className="text-[10px] text-[#64748b] uppercase tracking-wider block mb-1">
                Achieved Best
              </span>
              <span className="text-sm font-semibold font-mono text-[#4ade80]">
                {mission.best_metric_value ?? "—"}
              </span>
              {mission.best_metric_iteration !== null && (
                <span className="text-[10px] text-[#64748b] block mt-0.5">
                  @ iter {mission.best_metric_iteration}
                </span>
              )}
            </div>

            <div className="bg-[#0f172a] border border-[#334155] rounded-lg p-3">
              <span className="text-[10px] text-[#64748b] uppercase tracking-wider block mb-1">
                Target Metric
              </span>
              <span className="text-sm font-semibold font-mono text-[#14b8a6]">
                {targetProgress ? `${targetProgress.targetVal} (${targetProgress.targetKey})` : "—"}
              </span>
              {targetProgress && (
                <span className="text-[10px] text-[#4ade80] block mt-0.5">
                  {targetProgress.progressPct}% of goal
                </span>
              )}
            </div>

            <div className="bg-[#0f172a] border border-[#334155] rounded-lg p-3">
              <span className="text-[10px] text-[#64748b] uppercase tracking-wider block mb-1">
                Total Duration
              </span>
              <span className="text-sm font-semibold text-[#e2e8f0]">
                {durationStr}
              </span>
              <span className="text-[10px] text-[#64748b] block mt-0.5">
                {mission.current_iteration} total iterations
              </span>
            </div>

            <div className="bg-[#0f172a] border border-[#334155] rounded-lg p-3">
              <span className="text-[10px] text-[#64748b] uppercase tracking-wider block mb-1">
                Completed
              </span>
              <span className="text-xs font-medium text-[#cbd5e1]">
                {formatRelativeTime(mission.completed_at || mission.created_at)}
              </span>
              <span className="text-[9px] text-[#64748b] block mt-0.5">
                {fmtTs(mission.completed_at || mission.created_at)}
              </span>
            </div>
          </div>

          {activeView === "requirements" ? (
            <div className="space-y-3">
              <div className="flex items-center justify-between">
                <h4 className="text-xs font-semibold text-[#cbd5e1] uppercase tracking-wider flex items-center gap-1.5">
                  <ShieldCheck className="w-3.5 h-3.5 text-[#14b8a6]" />
                  Manifest Requirements Verification
                </h4>
                {manifest && (
                  <span className="text-[10px] text-[#64748b] font-mono">
                    Manifest v{manifest.version}
                  </span>
                )}
              </div>

              {loading ? (
                <div className="py-8 text-center text-xs text-[#64748b]">
                  Loading manifest requirements...
                </div>
              ) : manifest && manifest.requirements.length > 0 ? (
                <div className="space-y-2">
                  {manifest.requirements.map((req) => (
                    <div
                      key={req.id}
                      className="bg-[#0f172a] border border-[#334155] rounded-lg p-3 flex items-start justify-between gap-3"
                    >
                      <div className="space-y-1">
                        <div className="flex items-center gap-2">
                          <span
                            className={`w-4 h-4 rounded-full flex items-center justify-center text-[10px] font-bold ${
                              req.passed
                                ? "bg-[#4ade80]/20 text-[#4ade80] border border-[#4ade80]/40"
                                : "bg-[#f87171]/20 text-[#f87171] border border-[#f87171]/40"
                            }`}
                          >
                            {req.passed ? "✓" : "✕"}
                          </span>
                          <span className="text-xs font-medium text-[#e2e8f0]">
                            {req.description}
                          </span>
                        </div>
                        {req.evidence && (
                          <div className="pl-6 text-[11px] text-[#94a3b8] font-mono">
                            Evidence: <span className="text-[#14b8a6]">{req.evidence}</span>
                          </div>
                        )}
                      </div>
                      <span className="text-[10px] px-2 py-0.5 rounded bg-[#1e293b] border border-[#334155] text-[#94a3b8] font-mono shrink-0">
                        {req.category}
                      </span>
                    </div>
                  ))}
                </div>
              ) : (
                <div className="bg-[#0f172a] border border-[#334155] rounded-lg p-4 text-xs text-[#94a3b8] space-y-2">
                  <div className="flex items-center gap-2 text-[#4ade80]">
                    <CheckCircle2 className="w-4 h-4" />
                    <span className="font-semibold">Mission Completed Successfully</span>
                  </div>
                  <p className="text-[11px] text-[#64748b]">
                    The training loop satisfied the stopping criteria and completed without unhandled errors. Target metric: {JSON.stringify(mission.target_metric)}.
                  </p>
                </div>
              )}
            </div>
          ) : (
            <div className="space-y-2">
              <h4 className="text-xs font-semibold text-[#cbd5e1] uppercase tracking-wider">
                Full Mission Record
              </h4>
              <pre className="bg-[#0f172a] border border-[#334155] rounded-lg p-4 font-mono text-xs text-[#cbd5e1] overflow-x-auto max-h-64">
                {JSON.stringify(mission, null, 2)}
              </pre>
            </div>
          )}
        </div>

        {/* Modal Footer */}
        <div className="p-4 border-t border-[#334155] flex items-center justify-between bg-[#0f172a]/60">
          <button
            onClick={onClose}
            className="px-4 py-1.5 rounded text-xs text-[#94a3b8] hover:text-[#e2e8f0] transition-colors"
          >
            Close
          </button>
          <button
            onClick={() => {
              onClose();
              router.push(`/missions/${mission.id}`);
            }}
            className="px-4 py-2 rounded text-xs font-semibold bg-[#14b8a6] hover:bg-[#0d9488] text-[#0f172a] flex items-center gap-1.5 transition-colors"
          >
            <span>Open Mission HUD</span>
            <ArrowUpRight className="w-3.5 h-3.5" />
          </button>
        </div>
      </div>
    </div>
  );
}

export default function CompletedMissionsPage() {
  const router = useRouter();
  const { data: allMissions, isLoading } = useMissions();
  const [searchQuery, setSearchQuery] = useState<string>("");
  const [activeTab, setActiveTab] = useState<string>("all");
  const [sortBy, setSortBy] = useState<"newest" | "score" | "duration">("newest");
  const [selectedMission, setSelectedMission] = useState<Mission | null>(null);

  // Filter only completed missions
  const completedMissions = useMemo(() => {
    if (!allMissions) return [];
    return allMissions.filter((m) => m.status === "completed");
  }, [allMissions]);

  // Tab & search filtering
  const filteredMissions = useMemo(() => {
    return completedMissions.filter((m) => {
      const q = searchQuery.toLowerCase().trim();
      const goalStr = m.goal.toLowerCase();
      const idStr = m.id.toLowerCase();
      const taskTypeStr = (m.task_type || "").toLowerCase();
      const hostStr = (m.host || "").toLowerCase();

      const matchesSearch =
        !q ||
        goalStr.includes(q) ||
        idStr.includes(q) ||
        taskTypeStr.includes(q) ||
        hostStr.includes(q);

      if (!matchesSearch) return false;

      if (activeTab === "all") return true;
      if (activeTab === "minatar") {
        return (
          goalStr.includes("minatar") ||
          goalStr.includes("seaquest") ||
          goalStr.includes("freeway") ||
          goalStr.includes("breakout") ||
          goalStr.includes("asterix") ||
          goalStr.includes("space_invaders") ||
          goalStr.includes("space invaders")
        );
      }
      if (activeTab === "snake") return goalStr.includes("snake");
      if (activeTab === "tetris") return goalStr.includes("tetris");
      if (activeTab === "2048") return goalStr.includes("2048");
      if (activeTab === "llm") {
        return (
          taskTypeStr === "sft" ||
          taskTypeStr === "dpo" ||
          taskTypeStr === "grpo" ||
          taskTypeStr === "star" ||
          taskTypeStr === "rft" ||
          taskTypeStr === "distill" ||
          taskTypeStr === "prompt" ||
          taskTypeStr === "mlx_lora" ||
          taskTypeStr === "post-training" ||
          goalStr.includes("fine-tuning") ||
          goalStr.includes("fine tuning") ||
          goalStr.includes("ensemble") ||
          goalStr.includes("reasoner") ||
          goalStr.includes("prompt") ||
          goalStr.includes("distill")
        );
      }
      if (activeTab === "agentgym") {
        return (
          goalStr.includes("agentgym") ||
          goalStr.includes("multiturn") ||
          goalStr.includes("webarena") ||
          goalStr.includes("alfworld")
        );
      }
      if (activeTab === "classic") {
        return (
          goalStr.includes("cartpole") ||
          goalStr.includes("lunarlander") ||
          goalStr.includes("scikit") ||
          goalStr.includes("classifier") ||
          taskTypeStr === "ml"
        );
      }

      return true;
    });
  }, [completedMissions, searchQuery, activeTab]);

  // Sorting
  const sortedMissions = useMemo(() => {
    return [...filteredMissions].sort((a, b) => {
      if (sortBy === "newest") {
        const tA = parseTs(a.completed_at || a.created_at).getTime();
        const tB = parseTs(b.completed_at || b.created_at).getTime();
        return tB - tA;
      }
      if (sortBy === "score") {
        const sA = a.best_metric_value ? parseFloat(a.best_metric_value) : -Infinity;
        const sB = b.best_metric_value ? parseFloat(b.best_metric_value) : -Infinity;
        return sB - sA;
      }
      if (sortBy === "duration") {
        const durA =
          a.completed_at && a.created_at
            ? parseTs(a.completed_at).getTime() - parseTs(a.created_at).getTime()
            : 0;
        const durB =
          b.completed_at && b.created_at
            ? parseTs(b.completed_at).getTime() - parseTs(b.created_at).getTime()
            : 0;
        return durA - durB;
      }
      return 0;
    });
  }, [filteredMissions, sortBy]);

  // High-level statistics
  const stats = useMemo(() => {
    const total = completedMissions.length;
    let targetMetCount = 0;
    let totalIterations = 0;

    for (const m of completedMissions) {
      totalIterations += m.current_iteration || 0;
      const prog = getTargetProgress(m);
      if (prog && prog.isPassed) {
        targetMetCount++;
      } else if (!prog && m.status === "completed") {
        targetMetCount++;
      }
    }

    const targetMetRate = total > 0 ? Math.round((targetMetCount / total) * 100) : 0;
    return { total, targetMetRate, totalIterations };
  }, [completedMissions]);

  return (
    <div className="max-w-7xl mx-auto px-6 py-8 space-y-8">
      {/* Header */}
      <div>
        <div className="flex items-center gap-2 mb-1">
          <span className="text-xl">🏆</span>
          <h1 className="text-lg font-semibold text-[#e2e8f0] tracking-wide">
            Completed Missions Archive
          </h1>
        </div>
        <p className="text-xs text-[#94a3b8]">
          Browse converged training missions, verify requirement manifests, review achieved benchmarks, and replay runs.
        </p>
      </div>

      {/* Summary KPI Cards */}
      <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
        <div className="bg-[#1e293b]/70 border border-[#334155] rounded-lg p-4 flex items-center justify-between">
          <div>
            <span className="text-[11px] text-[#94a3b8] tracking-wider uppercase block">
              Completed Runs
            </span>
            <span className="text-2xl font-bold font-mono text-[#4ade80]">
              {stats.total}
            </span>
          </div>
          <div className="w-10 h-10 rounded-full bg-[#4ade80]/10 border border-[#4ade80]/20 flex items-center justify-center text-[#4ade80]">
            <CheckCircle2 className="w-5 h-5" />
          </div>
        </div>

        <div className="bg-[#1e293b]/70 border border-[#334155] rounded-lg p-4 flex items-center justify-between">
          <div>
            <span className="text-[11px] text-[#94a3b8] tracking-wider uppercase block">
              Target Pass Rate
            </span>
            <span className="text-2xl font-bold font-mono text-[#14b8a6]">
              {stats.targetMetRate}%
            </span>
          </div>
          <div className="w-10 h-10 rounded-full bg-[#14b8a6]/10 border border-[#14b8a6]/20 flex items-center justify-center text-[#14b8a6]">
            <Target className="w-5 h-5" />
          </div>
        </div>

        <div className="bg-[#1e293b]/70 border border-[#334155] rounded-lg p-4 flex items-center justify-between">
          <div>
            <span className="text-[11px] text-[#94a3b8] tracking-wider uppercase block">
              Total Iterations Executed
            </span>
            <span className="text-2xl font-bold font-mono text-[#a855f7]">
              {stats.totalIterations}
            </span>
          </div>
          <div className="w-10 h-10 rounded-full bg-[#a855f7]/10 border border-[#a855f7]/20 flex items-center justify-center text-[#a855f7]">
            <Layers className="w-5 h-5" />
          </div>
        </div>
      </div>

      {/* Filter Tabs & Search Bar (similar to recipes) */}
      <div className="flex flex-col md:flex-row items-stretch md:items-center justify-between gap-4 bg-[#1e293b]/70 border border-[#334155] rounded-lg p-4">
        {/* Domain Tabs */}
        <div className="flex items-center gap-1.5 overflow-x-auto pb-1 md:pb-0">
          {DOMAIN_TABS.map((tab) => (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              className={`px-3 py-1.5 text-xs rounded transition-colors whitespace-nowrap ${
                activeTab === tab.id
                  ? "bg-[#14b8a6] text-[#0f172a] font-semibold"
                  : "bg-[#0f172a] text-[#94a3b8] hover:text-[#e2e8f0] border border-[#334155]"
              }`}
            >
              {tab.label}
            </button>
          ))}
        </div>

        {/* Search & Sort Controls */}
        <div className="flex items-center gap-3">
          <div className="relative w-full md:w-64">
            <input
              type="text"
              placeholder="Search goals, IDs, environments..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="w-full bg-[#0f172a] border border-[#334155] rounded px-3 py-1.5 text-xs text-[#e2e8f0] placeholder-[#64748b] focus:outline-none focus:border-[#14b8a6]"
            />
            {searchQuery && (
              <button
                onClick={() => setSearchQuery("")}
                className="absolute right-2.5 top-1.5 text-xs text-[#64748b] hover:text-[#94a3b8]"
              >
                ✕
              </button>
            )}
          </div>

          <select
            value={sortBy}
            onChange={(e) => setSortBy(e.target.value as any)}
            className="bg-[#0f172a] border border-[#334155] text-xs text-[#94a3b8] rounded px-2.5 py-1.5 focus:outline-none focus:border-[#14b8a6]"
          >
            <option value="newest">Newest First</option>
            <option value="score">Highest Metric</option>
            <option value="duration">Fastest Duration</option>
          </select>
        </div>
      </div>

      {/* Grid of Completed Missions */}
      {isLoading ? (
        <div className="py-16 text-center text-xs text-[#64748b]">
          Loading completed missions...
        </div>
      ) : sortedMissions.length === 0 ? (
        <div className="py-16 text-center border border-dashed border-[#334155] rounded-lg space-y-2">
          <span className="text-2xl">🔍</span>
          <p className="text-xs text-[#94a3b8]">No completed missions match your criteria.</p>
        </div>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-5">
          {sortedMissions.map((m) => {
            const targetProgress = getTargetProgress(m);
            const duration = formatDuration(m.created_at, m.completed_at);
            const bestValFormatted = m.best_metric_value
              ? (() => {
                  const num = parseFloat(m.best_metric_value);
                  return Number.isFinite(num) ? num.toFixed(2) : m.best_metric_value;
                })()
              : null;

            return (
              <div
                key={m.id}
                className="bg-[#1e293b]/70 border border-[#334155] rounded-lg p-5 flex flex-col justify-between hover:border-[#4ade80]/40 transition-all hover:shadow-lg hover:shadow-[#4ade80]/5"
              >
                <div>
                  {/* Top Bar: ID, Type, Host, Timestamp */}
                  <div className="flex items-start justify-between gap-2 mb-2.5">
                    <div className="flex items-center gap-1.5 flex-wrap">
                      <span className="text-[10px] text-[#64748b] tracking-widest font-mono">
                        #{m.id.slice(0, 8)}
                      </span>
                      {m.task_type && (
                        <span className="text-[9px] px-1.5 py-0.5 rounded bg-[#0f172a] border border-[#334155] text-[#94a3b8] uppercase font-mono">
                          {m.task_type}
                        </span>
                      )}
                      {m.host && (
                        <span className="text-[9px] text-[#64748b] flex items-center gap-1">
                          <Server className="w-2.5 h-2.5" />
                          {m.host}
                        </span>
                      )}
                    </div>
                    <span
                      className="text-[10px] text-[#64748b]"
                      title={`Completed at ${fmtTs(m.completed_at || m.created_at)}`}
                    >
                      {formatRelativeTime(m.completed_at || m.created_at)}
                    </span>
                  </div>

                  {/* Goal Description */}
                  <h3
                    onClick={() => router.push(`/missions/${m.id}`)}
                    className="text-sm font-semibold text-[#e2e8f0] break-words line-clamp-2 mb-3 cursor-pointer hover:text-[#14b8a6] transition-colors leading-relaxed"
                  >
                    {m.goal}
                  </h3>

                  {/* Metric & Target Box */}
                  <div className="bg-[#0f172a] border border-[#334155] rounded-lg p-3 mb-4 space-y-2">
                    <div className="flex items-center justify-between text-xs">
                      <span className="text-[10px] text-[#64748b] flex items-center gap-1 uppercase tracking-wider">
                        <TrendingUp className="w-3 h-3 text-[#4ade80]" />
                        Achieved
                      </span>
                      <span className="font-semibold font-mono text-[#4ade80]">
                        {bestValFormatted ?? "—"}
                      </span>
                    </div>

                    {targetProgress ? (
                      <div className="space-y-1.5 pt-1.5 border-t border-[#334155]/60">
                        <div className="flex justify-between text-[10px] text-[#94a3b8]">
                          <span className="flex items-center gap-1">
                            <Target className="w-3 h-3 text-[#14b8a6]" />
                            target {targetProgress.targetVal} ({targetProgress.targetKey})
                          </span>
                          <span className="font-mono text-[#4ade80]">
                            {targetProgress.progressPct}% met
                          </span>
                        </div>
                        <div className="w-full rounded-full h-1 bg-[#1e293b] overflow-hidden">
                          <div
                            className="h-full rounded-full bg-[#4ade80] transition-all"
                            style={{
                              width: `${Math.min(100, targetProgress.progressPct)}%`,
                            }}
                          />
                        </div>
                      </div>
                    ) : (
                      <div className="text-[10px] text-[#64748b] pt-1 border-t border-[#334155]/60 flex items-center justify-between">
                        <span>Iter {m.current_iteration}</span>
                        {m.best_metric_iteration !== null && (
                          <span>best @ {m.best_metric_iteration}</span>
                        )}
                      </div>
                    )}
                  </div>

                  {/* Badges / Duration */}
                  <div className="flex items-center justify-between text-[11px] text-[#64748b] mb-2">
                    <span className="flex items-center gap-1">
                      <Clock className="w-3 h-3" />
                      {duration}
                    </span>
                    <span className="text-[10px] bg-[#4ade80]/10 text-[#4ade80] border border-[#4ade80]/20 px-2 py-0.5 rounded font-mono">
                      ✓ Converged
                    </span>
                  </div>
                </div>

                {/* Card Footer / Actions */}
                <div className="pt-4 border-t border-[#334155]/60 flex items-center justify-between gap-2">
                  <button
                    onClick={() => setSelectedMission(m)}
                    className="text-xs text-[#94a3b8] hover:text-[#e2e8f0] flex items-center gap-1 transition-colors"
                  >
                    <FileText className="w-3 h-3" />
                    Manifest
                  </button>

                  <Link
                    href={`/missions/${m.id}`}
                    className="px-3 py-1.5 rounded text-xs font-semibold bg-[#14b8a6]/10 hover:bg-[#14b8a6]/20 border border-[#14b8a6]/30 text-[#14b8a6] hover:text-[#2dd4bf] flex items-center gap-1 transition-colors"
                  >
                    <span>Mission HUD</span>
                    <ArrowUpRight className="w-3 h-3" />
                  </Link>
                </div>
              </div>
            );
          })}
        </div>
      )}

      {/* Manifest & Verification Modal */}
      {selectedMission && (
        <ManifestModal
          mission={selectedMission}
          onClose={() => setSelectedMission(null)}
        />
      )}
    </div>
  );
}
