"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { api, ModelRecord, TournamentResponse, TournamentEntry } from "@/lib/api";

const ENV_OPTIONS = [
  { id: "Snake-v0", label: "Snake-v0" },
  { id: "Tetris-v0", label: "Tetris-v0" },
  { id: "Game2048-v0", label: "2048 (Lookahead)" },
  { id: "MinAtar-Breakout-v0", label: "MinAtar Breakout" },
  { id: "MinAtar-SpaceInvaders-v0", label: "MinAtar Space Invaders" },
  { id: "MinAtar-Asteroids-v0", label: "MinAtar Asteroids" },
];

export default function ModelsPage() {
  const [models, setModels] = useState<ModelRecord[]>([]);
  const [selectedEnv, setSelectedEnv] = useState<string>("Snake-v0");
  const [filterDomain, setFilterDomain] = useState<string>("all");
  const [episodes, setEpisodes] = useState<number>(5);
  const [loading, setLoading] = useState<boolean>(true);
  const [tournamentRunning, setTournamentRunning] = useState<boolean>(false);
  const [tournamentResult, setTournamentResult] = useState<TournamentResponse | null>(null);
  const [tournamentError, setTournamentError] = useState<string | null>(null);
  const [selectedModelIds, setSelectedModelIds] = useState<string[]>([]);

  const fetchModels = async () => {
    try {
      setLoading(true);
      const data = await api.getModels();
      setModels(data);
    } catch (e) {
      console.error("Failed to load models:", e);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchModels();
  }, []);

  const handleRunTournament = async () => {
    try {
      setTournamentRunning(true);
      setTournamentError(null);
      const res = await api.runTournament(
        selectedEnv,
        selectedModelIds.length > 0 ? selectedModelIds : undefined,
        episodes,
        true
      );
      setTournamentResult(res);
      fetchModels();
    } catch (e: any) {
      setTournamentError(e.message || "Failed to execute tournament match");
    } finally {
      setTournamentRunning(false);
    }
  };

  const handleSetChampion = async (id: string) => {
    try {
      await api.updateModel(id, { is_champion: true });
      fetchModels();
    } catch (e) {
      console.error("Failed to set champion:", e);
    }
  };

  const handleDeleteModel = async (id: string) => {
    if (!confirm("Delete this model record from the registry?")) return;
    try {
      await api.deleteModel(id);
      fetchModels();
    } catch (e) {
      console.error("Failed to delete model:", e);
    }
  };

  const filteredModels = models.filter((m) => {
    if (filterDomain === "all") return true;
    return m.domain.toLowerCase().includes(filterDomain.toLowerCase());
  });

  return (
    <div className="max-w-7xl mx-auto px-6 py-8 space-y-8">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-end justify-between gap-4">
        <div>
          <div className="flex items-center gap-2 mb-1">
            <span className="text-xl">🏆</span>
            <h1 className="text-lg font-semibold text-[#e2e8f0] tracking-wide">
              Model Registry & Tournament Arena
            </h1>
          </div>
          <p className="text-xs text-[#94a3b8]">
            Benchmark policy checkpoints side-by-side across fixed seeds, track win rates, and crown champion models.
          </p>
        </div>
        <div className="flex items-center gap-3">
          <Link
            href="/"
            className="text-xs text-[#14b8a6] hover:text-[#2dd4bf] transition-colors border border-[#14b8a6]/20 bg-[#14b8a6]/5 px-3 py-1.5 rounded"
          >
            ← Command Center
          </Link>
          <Link
            href="/recipes"
            className="text-xs text-[#94a3b8] hover:text-[#e2e8f0] transition-colors border border-[#334155] bg-[#1e293b] px-3 py-1.5 rounded"
          >
            Recipe Library →
          </Link>
        </div>
      </div>

      {/* Tournament Arena Section */}
      <div className="bg-[#1e293b]/70 border border-[#334155] rounded-lg p-6 space-y-6">
        <div className="flex items-center justify-between">
          <div>
            <h2 className="text-sm font-semibold text-[#e2e8f0] flex items-center gap-2">
              <span className="inline-block w-2 h-2 rounded-full bg-[#14b8a6]" />
              Tournament Head-to-Head Arena
            </h2>
            <p className="text-xs text-[#94a3b8] mt-0.5">
              Simulate identical fixed environment seeds across candidate policies.
            </p>
          </div>
          <button
            onClick={handleRunTournament}
            disabled={tournamentRunning}
            className={`px-4 py-2 rounded text-xs font-semibold tracking-wider transition-all flex items-center gap-2 ${
              tournamentRunning
                ? "bg-[#334155] text-[#94a3b8] cursor-not-allowed"
                : "bg-[#14b8a6] hover:bg-[#0d9488] text-[#0f172a] shadow-lg shadow-[#14b8a6]/20"
            }`}
          >
            {tournamentRunning ? (
              <>
                <span className="animate-spin inline-block w-3.5 h-3.5 border-2 border-[#94a3b8] border-t-transparent rounded-full" />
                Simulating Matches...
              </>
            ) : (
              <>
                <span>⚔️</span> Run Tournament Match
              </>
            )}
          </button>
        </div>

        {/* Controls */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 pt-2 border-t border-[#334155]/50">
          <div>
            <label className="block text-[11px] uppercase tracking-wider text-[#94a3b8] mb-1.5">
              Target Environment
            </label>
            <select
              value={selectedEnv}
              onChange={(e) => setSelectedEnv(e.target.value)}
              className="w-full bg-[#0f172a] border border-[#334155] rounded px-3 py-2 text-xs text-[#e2e8f0] focus:outline-none focus:border-[#14b8a6]"
            >
              {ENV_OPTIONS.map((env) => (
                <option key={env.id} value={env.id}>
                  {env.label}
                </option>
              ))}
            </select>
          </div>

          <div>
            <label className="block text-[11px] uppercase tracking-wider text-[#94a3b8] mb-1.5">
              Evaluation Episodes (Fixed Seeds)
            </label>
            <div className="flex items-center gap-2">
              {[3, 5, 10, 20].map((num) => (
                <button
                  key={num}
                  type="button"
                  onClick={() => setEpisodes(num)}
                  className={`flex-1 py-1.5 text-xs rounded border transition-colors ${
                    episodes === num
                      ? "bg-[#14b8a6]/20 border-[#14b8a6] text-[#14b8a6] font-medium"
                      : "bg-[#0f172a] border-[#334155] text-[#94a3b8] hover:text-[#e2e8f0]"
                  }`}
                >
                  {num} eps
                </button>
              ))}
            </div>
          </div>

          <div>
            <label className="block text-[11px] uppercase tracking-wider text-[#94a3b8] mb-1.5">
              Candidate Selection
            </label>
            <p className="text-xs text-[#64748b] pt-1">
              Auto-discovers checkpoints in <code className="text-[#14b8a6] font-mono">runs/</code> and registry.
            </p>
          </div>
        </div>

        {/* Tournament Error Message */}
        {tournamentError && (
          <div className="bg-[#f87171]/10 border border-[#f87171]/30 rounded p-3 text-xs text-[#f87171]">
            ⚠️ {tournamentError}
          </div>
        )}

        {/* Tournament Results Leaderboard */}
        {tournamentResult && (
          <div className="space-y-4 pt-4 border-t border-[#334155]/60">
            <div className="flex items-center justify-between">
              <span className="text-xs uppercase tracking-widest text-[#94a3b8]">
                Leaderboard Results ({tournamentResult.env_id} · {tournamentResult.episodes} episodes)
              </span>
              {tournamentResult.champion_id && (
                <span className="text-xs text-[#fbbf24] bg-[#fbbf24]/10 border border-[#fbbf24]/30 px-2.5 py-0.5 rounded-full flex items-center gap-1.5">
                  <span>👑</span> Champion: {tournamentResult.leaderboard[0]?.name || tournamentResult.champion_id}
                </span>
              )}
            </div>

            <div className="grid grid-cols-1 gap-3">
              {tournamentResult.leaderboard.map((entry: TournamentEntry) => {
                const isFirst = entry.rank === 1;
                const isSecond = entry.rank === 2;
                const isThird = entry.rank === 3;
                return (
                  <div
                    key={entry.model_id}
                    className={`p-4 rounded-lg border transition-all ${
                      isFirst
                        ? "bg-[#14b8a6]/10 border-[#14b8a6]/40 shadow-md shadow-[#14b8a6]/5"
                        : "bg-[#0f172a]/60 border-[#334155]/60"
                    }`}
                  >
                    <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
                      <div className="flex items-center gap-3">
                        <span
                          className={`w-7 h-7 rounded-full flex items-center justify-center font-bold text-xs ${
                            isFirst
                              ? "bg-[#fbbf24] text-[#0f172a]"
                              : isSecond
                              ? "bg-[#94a3b8] text-[#0f172a]"
                              : isThird
                              ? "bg-[#b45309] text-[#e2e8f0]"
                              : "bg-[#334155] text-[#94a3b8]"
                          }`}
                        >
                          {entry.rank}
                        </span>
                        <div>
                          <div className="flex items-center gap-2">
                            <span className="text-sm font-medium text-[#e2e8f0]">
                              {entry.name}
                            </span>
                            {isFirst && (
                              <span className="text-[10px] bg-[#fbbf24]/20 text-[#fbbf24] px-2 py-0.5 rounded">
                                Winner
                              </span>
                            )}
                          </div>
                          <span className="text-[11px] text-[#64748b] font-mono">
                            {entry.checkpoint_path}
                          </span>
                        </div>
                      </div>

                      {/* Stats */}
                      <div className="flex items-center gap-6 text-xs">
                        <div>
                          <span className="text-[#64748b] text-[10px] block uppercase">Mean Score</span>
                          <span className="text-sm font-semibold text-[#14b8a6]">
                            {entry.mean_score.toFixed(1)} <span className="text-[10px] text-[#64748b]">±{entry.std_score.toFixed(1)}</span>
                          </span>
                        </div>
                        <div>
                          <span className="text-[#64748b] text-[10px] block uppercase">Min / Max</span>
                          <span className="text-xs text-[#e2e8f0] font-mono">
                            {entry.min_score.toFixed(0)} - {entry.max_score.toFixed(0)}
                          </span>
                        </div>
                        <div>
                          <span className="text-[#64748b] text-[10px] block uppercase">Win Rate</span>
                          <span className="text-xs font-semibold text-[#a855f7]">
                            {(entry.win_rate * 100).toFixed(0)}%
                          </span>
                        </div>
                      </div>
                    </div>

                    {/* Per-Episode Scores Bar */}
                    <div className="mt-3 pt-2 border-t border-[#334155]/40 flex items-center gap-2">
                      <span className="text-[10px] text-[#64748b] uppercase tracking-wider">Per Seed:</span>
                      <div className="flex items-center gap-1.5 flex-wrap">
                        {entry.scores.map((s, idx) => (
                          <span
                            key={idx}
                            className="text-[11px] font-mono bg-[#1e293b] border border-[#334155] px-1.5 py-0.5 rounded text-[#cbd5e1]"
                          >
                            ep{idx + 1}: {s.toFixed(0)}
                          </span>
                        ))}
                      </div>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        )}
      </div>

      {/* Model Records Table Section */}
      <div className="bg-[#1e293b]/70 border border-[#334155] rounded-lg p-6 space-y-4">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
          <div>
            <h2 className="text-sm font-semibold text-[#e2e8f0]">
              Registered Model Checkpoints
            </h2>
            <p className="text-xs text-[#94a3b8] mt-0.5">
              Production candidate models tracked in the SQLite Model Registry.
            </p>
          </div>

          <div className="flex items-center gap-2">
            <label className="text-xs text-[#64748b]">Domain:</label>
            <select
              value={filterDomain}
              onChange={(e) => setFilterDomain(e.target.value)}
              className="bg-[#0f172a] border border-[#334155] rounded px-2.5 py-1 text-xs text-[#e2e8f0] focus:outline-none"
            >
              <option value="all">All Domains</option>
              <option value="Snake">Snake</option>
              <option value="Tetris">Tetris</option>
              <option value="2048">2048</option>
              <option value="MinAtar">MinAtar</option>
            </select>
          </div>
        </div>

        {loading ? (
          <div className="py-8 text-center text-xs text-[#64748b]">
            Loading registry models...
          </div>
        ) : filteredModels.length === 0 ? (
          <div className="py-8 text-center border border-dashed border-[#334155] rounded-lg space-y-2">
            <span className="text-2xl">🤖</span>
            <p className="text-xs text-[#94a3b8]">No model records found in database.</p>
            <p className="text-[11px] text-[#64748b] max-w-md mx-auto">
              Completed missions automatically register champion checkpoints. You can also run the Tournament Arena above to evaluate any models saved in <code className="text-[#14b8a6]">runs/</code>.
            </p>
          </div>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-left text-xs text-[#e2e8f0]">
              <thead className="text-[11px] text-[#64748b] uppercase border-b border-[#334155]/60">
                <tr>
                  <th className="py-2.5 px-3">Name</th>
                  <th className="py-2.5 px-3">Domain</th>
                  <th className="py-2.5 px-3">Framework / Architecture</th>
                  <th className="py-2.5 px-3">Best Metric</th>
                  <th className="py-2.5 px-3">Status</th>
                  <th className="py-2.5 px-3 text-right">Actions</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-[#334155]/40">
                {filteredModels.map((m) => (
                  <tr key={m.id} className="hover:bg-[#0f172a]/40 transition-colors">
                    <td className="py-3 px-3 font-medium">
                      <div className="flex items-center gap-2">
                        <span>{m.name}</span>
                        {m.is_champion && (
                          <span className="text-[10px] bg-[#fbbf24]/20 text-[#fbbf24] border border-[#fbbf24]/30 px-1.5 py-0.5 rounded flex items-center gap-1">
                            <span>👑</span> Champion
                          </span>
                        )}
                      </div>
                      <div className="text-[10px] text-[#64748b] font-mono truncate max-w-xs">
                        {m.checkpoint_path || m.weights_path}
                      </div>
                    </td>
                    <td className="py-3 px-3">
                      <span className="bg-[#0f172a] border border-[#334155] px-2 py-0.5 rounded text-[11px] text-[#14b8a6]">
                        {m.domain}
                      </span>
                    </td>
                    <td className="py-3 px-3 text-[#94a3b8]">
                      {m.framework || "PyTorch"} / {m.architecture || "Custom"}
                    </td>
                    <td className="py-3 px-3">
                      {m.best_metric_name ? (
                        <span className="font-mono text-[#4ade80]">
                          {m.best_metric_name}={m.best_metric_value ?? "-"}
                        </span>
                      ) : (
                        <span className="text-[#64748b]">-</span>
                      )}
                    </td>
                    <td className="py-3 px-3">
                      {m.is_champion ? (
                        <span className="text-[#fbbf24] font-medium text-[11px]">Active Champ</span>
                      ) : (
                        <span className="text-[#64748b] text-[11px]">Contender</span>
                      )}
                    </td>
                    <td className="py-3 px-3 text-right space-x-2">
                      {!m.is_champion && (
                        <button
                          onClick={() => handleSetChampion(m.id)}
                          className="text-[11px] text-[#fbbf24] hover:text-[#fde68a] transition-colors"
                        >
                          Crown Champ
                        </button>
                      )}
                      <button
                        onClick={() => handleDeleteModel(m.id)}
                        className="text-[11px] text-[#f87171] hover:text-[#fca5a5] transition-colors"
                      >
                        Delete
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
}
