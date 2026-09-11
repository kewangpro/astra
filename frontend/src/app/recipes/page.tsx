"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { api, Recipe, RecipeRecord } from "@/lib/api";

const DOMAIN_TABS = [
  { id: "all", label: "All Recipes" },
  { id: "snake", label: "Snake" },
  { id: "tetris", label: "Tetris" },
  { id: "2048", label: "2048" },
  { id: "minatar", label: "MinAtar" },
  { id: "llm", label: "LLM / Reasoning" },
];

export default function RecipesPage() {
  const router = useRouter();
  const [recipes, setRecipes] = useState<Recipe[]>([]);
  const [dbRecipes, setDbRecipes] = useState<RecipeRecord[]>([]);
  const [searchQuery, setSearchQuery] = useState<string>("");
  const [activeTab, setActiveTab] = useState<string>("all");
  const [goldenOnly, setGoldenOnly] = useState<boolean>(false);
  const [loading, setLoading] = useState<boolean>(true);
  const [dispatching, setDispatching] = useState<string | null>(null);

  // Modal states
  const [selectedRecipe, setSelectedRecipe] = useState<Recipe | null>(null);
  const [lineageRecipe, setLineageRecipe] = useState<RecipeRecord | null>(null);
  const [lineageChain, setLineageChain] = useState<RecipeRecord[]>([]);
  const [loadingLineage, setLoadingLineage] = useState<boolean>(false);

  const fetchRecipes = async () => {
    try {
      setLoading(true);
      const [allList, dbList] = await Promise.all([
        api.getRecipes().catch(() => []),
        api.getDbRecipes().catch(() => []),
      ]);
      setRecipes(allList);
      setDbRecipes(dbList);
    } catch (e) {
      console.error("Failed to load recipes:", e);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchRecipes();
  }, []);

  const handleDispatch = async (recipeName: string) => {
    try {
      setDispatching(recipeName);
      const res = await api.dispatchRecipe(recipeName);
      if (res.mission_id) {
        router.push(`/missions/${res.mission_id}`);
      }
    } catch (e: any) {
      alert(`Failed to dispatch recipe: ${e.message || e}`);
      setDispatching(null);
    }
  };

  const handleViewLineage = async (rec: RecipeRecord) => {
    setLineageRecipe(rec);
    setLoadingLineage(true);
    try {
      const chain = await api.getRecipeLineage(rec.id);
      setLineageChain(chain);
    } catch (e) {
      console.error("Failed to fetch lineage:", e);
      setLineageChain([rec]);
    } finally {
      setLoadingLineage(false);
    }
  };

  // Filter recipes
  const filteredRecipes = recipes.filter((r) => {
    const q = searchQuery.toLowerCase();
    const matchesSearch =
      !q ||
      r.name.toLowerCase().includes(q) ||
      (r.description && r.description.toLowerCase().includes(q)) ||
      (r.domain && r.domain.toLowerCase().includes(q));

    if (!matchesSearch) return false;

    if (activeTab === "all") return true;
    if (activeTab === "snake") return r.name.toLowerCase().includes("snake");
    if (activeTab === "tetris") return r.name.toLowerCase().includes("tetris");
    if (activeTab === "2048") return r.name.toLowerCase().includes("2048");
    if (activeTab === "minatar") return r.name.toLowerCase().includes("minatar");
    if (activeTab === "llm")
      return (
        r.name.toLowerCase().includes("distill") ||
        r.name.toLowerCase().includes("dpo") ||
        r.name.toLowerCase().includes("grpo") ||
        r.name.toLowerCase().includes("prompt") ||
        r.name.toLowerCase().includes("rft") ||
        r.name.toLowerCase().includes("sft") ||
        r.name.toLowerCase().includes("mlx")
      );

    return true;
  });

  return (
    <div className="max-w-7xl mx-auto px-6 py-8 space-y-8">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-end justify-between gap-4">
        <div>
          <div className="flex items-center gap-2 mb-1">
            <span className="text-xl">📚</span>
            <h1 className="text-lg font-semibold text-[#e2e8f0] tracking-wide">
              Recipe Library & Lineage Visualizer
            </h1>
          </div>
          <p className="text-xs text-[#94a3b8]">
            Browse canonical YAML training blueprints, inspect genetic mutation lineage, and dispatch 1-click missions.
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
            href="/models"
            className="text-xs text-[#94a3b8] hover:text-[#e2e8f0] transition-colors border border-[#334155] bg-[#1e293b] px-3 py-1.5 rounded"
          >
            Model Registry →
          </Link>
        </div>
      </div>

      {/* Filter Tabs & Search Bar */}
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

        {/* Search */}
        <div className="relative w-full md:w-72">
          <input
            type="text"
            placeholder="Search recipes, domains, algos..."
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
      </div>

      {/* Recipe Gallery Grid */}
      {loading ? (
        <div className="py-16 text-center text-xs text-[#64748b]">
          Loading recipe library...
        </div>
      ) : filteredRecipes.length === 0 ? (
        <div className="py-16 text-center border border-dashed border-[#334155] rounded-lg space-y-2">
          <span className="text-2xl">🔍</span>
          <p className="text-xs text-[#94a3b8]">No recipes match your filter.</p>
        </div>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-5">
          {filteredRecipes.map((r) => {
            const dbMatch = dbRecipes.find((dbR) => dbR.name === r.name);
            const targetMetric = (r.content as any)?.target_metric || dbMatch?.target_metric;
            const targetStr = targetMetric
              ? Object.entries(targetMetric)
                  .map(([k, v]) => `${k}: ${v}`)
                  .join(", ")
              : null;
            const isDispatchingThis = dispatching === r.name;

            return (
              <div
                key={r.name}
                className="bg-[#1e293b]/70 border border-[#334155] rounded-lg p-5 flex flex-col justify-between hover:border-[#14b8a6]/40 transition-all hover:shadow-lg hover:shadow-[#14b8a6]/5"
              >
                <div>
                  <div className="flex items-start justify-between gap-2 mb-2">
                    <h3 className="text-sm font-semibold text-[#e2e8f0] break-words">
                      {r.name}
                    </h3>
                    <span className="text-[10px] bg-[#0f172a] border border-[#334155] text-[#14b8a6] px-2 py-0.5 rounded font-mono">
                      {r.domain || (r.content as any)?.domain || "RL"}
                    </span>
                  </div>

                  <p className="text-xs text-[#94a3b8] line-clamp-2 mb-4 leading-relaxed">
                    {r.description || (r.content as any)?.description || "Standard reinforcement learning & optimization recipe."}
                  </p>

                  {/* Target metric pill */}
                  {targetStr && (
                    <div className="mb-4 inline-flex items-center gap-1.5 text-[11px] bg-[#0f172a] border border-[#14b8a6]/30 text-[#14b8a6] px-2.5 py-1 rounded">
                      <span className="text-xs">🎯</span>
                      <span className="font-mono">{targetStr}</span>
                    </div>
                  )}
                </div>

                {/* Card footer / Actions */}
                <div className="pt-4 border-t border-[#334155]/60 flex items-center justify-between gap-2">
                  <div className="flex items-center gap-2">
                    <button
                      onClick={() => setSelectedRecipe(r)}
                      className="text-xs text-[#94a3b8] hover:text-[#e2e8f0] transition-colors"
                    >
                      YAML
                    </button>
                    {dbMatch && (
                      <button
                        onClick={() => handleViewLineage(dbMatch)}
                        className="text-xs text-[#a855f7] hover:text-[#c084fc] transition-colors"
                      >
                        Lineage
                      </button>
                    )}
                  </div>

                  <button
                    onClick={() => handleDispatch(r.name)}
                    disabled={isDispatchingThis}
                    className={`px-3 py-1.5 rounded text-xs font-semibold tracking-wider transition-all flex items-center gap-1.5 ${
                      isDispatchingThis
                        ? "bg-[#334155] text-[#94a3b8] cursor-not-allowed"
                        : "bg-[#14b8a6] hover:bg-[#0d9488] text-[#0f172a]"
                    }`}
                  >
                    {isDispatchingThis ? (
                      <>
                        <span className="animate-spin inline-block w-3 h-3 border-2 border-[#94a3b8] border-t-transparent rounded-full" />
                        Launching...
                      </>
                    ) : (
                      <>
                        <span>🚀</span> Dispatch
                      </>
                    )}
                  </button>
                </div>
              </div>
            );
          })}
        </div>
      )}

      {/* YAML / Details Modal */}
      {selectedRecipe && (
        <div className="fixed inset-0 z-50 bg-black/75 backdrop-blur-sm flex items-center justify-center p-4">
          <div className="bg-[#1e293b] border border-[#334155] rounded-xl max-w-2xl w-full max-h-[85vh] flex flex-col shadow-2xl">
            <div className="p-4 border-b border-[#334155] flex items-center justify-between">
              <div>
                <h3 className="text-sm font-semibold text-[#e2e8f0]">
                  {selectedRecipe.name}
                </h3>
                <span className="text-[11px] text-[#64748b]">
                  {selectedRecipe.filename}
                </span>
              </div>
              <button
                onClick={() => setSelectedRecipe(null)}
                className="text-xs text-[#94a3b8] hover:text-[#e2e8f0] p-1"
              >
                ✕
              </button>
            </div>

            <div className="p-4 overflow-y-auto flex-1 font-mono text-xs text-[#cbd5e1] bg-[#0f172a]/60">
              <pre className="whitespace-pre-wrap">
                {JSON.stringify(selectedRecipe.content, null, 2)}
              </pre>
            </div>

            <div className="p-4 border-t border-[#334155] flex items-center justify-between">
              <button
                onClick={() => setSelectedRecipe(null)}
                className="text-xs text-[#94a3b8] hover:text-[#e2e8f0]"
              >
                Close
              </button>
              <button
                onClick={() => {
                  const name = selectedRecipe.name;
                  setSelectedRecipe(null);
                  handleDispatch(name);
                }}
                className="px-4 py-2 rounded text-xs font-semibold bg-[#14b8a6] hover:bg-[#0d9488] text-[#0f172a] flex items-center gap-1.5"
              >
                <span>🚀</span> Dispatch Mission Now
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Lineage Visualizer Modal */}
      {lineageRecipe && (
        <div className="fixed inset-0 z-50 bg-black/75 backdrop-blur-sm flex items-center justify-center p-4">
          <div className="bg-[#1e293b] border border-[#334155] rounded-xl max-w-2xl w-full max-h-[85vh] flex flex-col shadow-2xl">
            <div className="p-4 border-b border-[#334155] flex items-center justify-between">
              <div>
                <h3 className="text-sm font-semibold text-[#e2e8f0] flex items-center gap-2">
                  <span>🧬</span> Lineage Ancestry DAG: {lineageRecipe.name}
                </h3>
                <span className="text-[11px] text-[#64748b]">
                  Tracking genetic evolution, parameter mutations, and consecutive generation wins.
                </span>
              </div>
              <button
                onClick={() => setLineageRecipe(null)}
                className="text-xs text-[#94a3b8] hover:text-[#e2e8f0] p-1"
              >
                ✕
              </button>
            </div>

            <div className="p-6 overflow-y-auto flex-1 space-y-6">
              {loadingLineage ? (
                <div className="py-8 text-center text-xs text-[#64748b]">
                  Tracing lineage chain...
                </div>
              ) : (
                <div className="relative pl-6 space-y-6 before:absolute before:left-2 before:top-2 before:bottom-2 before:w-0.5 before:bg-[#334155]">
                  {lineageChain.map((node, idx) => (
                    <div key={node.id} className="relative group">
                      {/* Node dot */}
                      <span className="absolute -left-6 top-1.5 w-4 h-4 rounded-full bg-[#1e293b] border-2 border-[#a855f7] flex items-center justify-center text-[9px] text-[#a855f7] font-bold">
                        {idx + 1}
                      </span>

                      <div className="bg-[#0f172a] border border-[#334155] rounded-lg p-4 space-y-2">
                        <div className="flex items-center justify-between">
                          <span className="text-xs font-semibold text-[#e2e8f0]">
                            {node.name} <span className="text-[10px] text-[#a855f7]">v{node.version}</span>
                          </span>
                          <span className="text-[10px] bg-[#a855f7]/10 text-[#a855f7] border border-[#a855f7]/20 px-2 py-0.5 rounded">
                            Gen {node.generation}
                          </span>
                        </div>

                        {node.score !== null && (
                          <div className="text-xs text-[#4ade80]">
                            Score / Reward: <span className="font-mono">{node.score}</span>
                          </div>
                        )}

                        <div className="text-[11px] text-[#94a3b8] font-mono bg-[#1e293b]/60 p-2 rounded max-h-28 overflow-y-auto">
                          {JSON.stringify(node.hyperparameters, null, 2)}
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>

            <div className="p-4 border-t border-[#334155] flex justify-end">
              <button
                onClick={() => setLineageRecipe(null)}
                className="px-4 py-1.5 rounded text-xs text-[#e2e8f0] bg-[#334155] hover:bg-[#475569]"
              >
                Close
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
