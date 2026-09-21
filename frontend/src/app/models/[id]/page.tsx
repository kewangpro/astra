"use client";

import { use, useEffect, useState } from "react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { api, ModelRecord } from "@/lib/api";
import { PolicyPlayer } from "@/components/hud/PolicyPlayer";
import { envIdFromDomain, isPlayableEnv } from "@/lib/playWs";

function missionIdFromModel(m: ModelRecord): string | null {
  const mid = m.extra_metadata?.mission_id;
  if (typeof mid === "string" && mid) return mid;
  const path = m.checkpoint_path || m.weights_path || "";
  const parts = path.split(/[/\\]/);
  const i = parts.indexOf("missions");
  if (i >= 0 && parts[i + 1]) return parts[i + 1];
  return null;
}

export default function ModelDetailPage({
  params,
}: {
  params: Promise<{ id: string }>;
}) {
  const { id } = use(params);
  const router = useRouter();
  const [model, setModel] = useState<ModelRecord | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let cancelled = false;
    api
      .getModel(id)
      .then((m) => {
        if (!cancelled) setModel(m);
      })
      .catch((e) => {
        if (!cancelled) setError(e.message || "Model not found");
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [id]);

  const handleSetChampion = async () => {
    if (!model) return;
    const updated = await api.updateModel(model.id, { is_champion: true });
    setModel(updated);
  };

  const handleDelete = async () => {
    if (!model) return;
    if (!confirm("Delete this model record from the registry?")) return;
    await api.deleteModel(model.id);
    router.push("/models");
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center h-96 text-[#64748b] text-sm">
        Loading model…
      </div>
    );
  }

  if (error || !model) {
    return (
      <div className="flex flex-col items-center justify-center h-96 gap-4 text-[#f87171] text-sm">
        {error || "Model not found."}
        <Link href="/models" className="text-[#14b8a6] text-xs hover:underline">
          ← Models
        </Link>
      </div>
    );
  }

  const envId = envIdFromDomain(model.domain);
  const missionId = missionIdFromModel(model);
  const ckpt = model.checkpoint_path || model.weights_path;
  const playable = isPlayableEnv(envId);

  return (
    <div className="max-w-7xl mx-auto px-6 py-8 space-y-6">
      <div className="flex items-start justify-between gap-4">
        <div className="min-w-0">
          <div className="flex items-center gap-3 mb-1">
            <Link
              href="/models"
              className="text-[#94a3b8] text-xs hover:text-[#14b8a6] transition-colors"
            >
              ← models
            </Link>
            <span className="text-[#64748b] text-xs truncate">/ {model.id.slice(0, 8)}</span>
          </div>
          <h1 className="text-lg font-semibold text-[#e2e8f0] tracking-wide flex items-center gap-2 flex-wrap">
            {model.name}
            {model.is_champion && (
              <span className="text-[10px] bg-[#fbbf24]/20 text-[#fbbf24] border border-[#fbbf24]/30 px-1.5 py-0.5 rounded">
                Champion
              </span>
            )}
          </h1>
          <p className="text-xs text-[#94a3b8] mt-1">
            Play / inference
          </p>
        </div>
        <div className="flex items-center gap-2 shrink-0">
          {!model.is_champion && (
            <button
              type="button"
              onClick={handleSetChampion}
              className="text-[11px] px-2.5 py-1 rounded border border-[#fbbf24]/40 text-[#fbbf24] hover:bg-[#fbbf24]/10"
            >
              Crown champ
            </button>
          )}
          <button
            type="button"
            onClick={handleDelete}
            className="text-[11px] px-2.5 py-1 rounded border border-[#f87171]/40 text-[#f87171] hover:bg-[#f87171]/10"
          >
            Delete
          </button>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
        <div className="lg:col-span-2">
          {playable ? (
            <PolicyPlayer modelId={model.id} envId={envId} />
          ) : (
            <div className="border border-dashed border-[#334155] rounded-lg p-10 text-center text-xs text-[#64748b]">
              No live canvas for {model.domain}.
            </div>
          )}
        </div>
        <aside className="space-y-3 bg-[#1e293b]/70 border border-[#334155] rounded-lg p-4 text-xs">
          <div>
            <div className="text-[10px] uppercase tracking-wider text-[#64748b] mb-1">Environment</div>
            <div className="text-[#14b8a6] font-medium">{model.domain}</div>
          </div>
          <div>
            <div className="text-[10px] uppercase tracking-wider text-[#64748b] mb-1">Architecture</div>
            <div className="text-[#e2e8f0]">
              {model.framework || "PyTorch"} / {model.architecture || "Custom"}
            </div>
          </div>
          <div>
            <div className="text-[10px] uppercase tracking-wider text-[#64748b] mb-1">Best metric</div>
            {model.best_metric_name ? (
              <div className="font-mono text-[#4ade80]">
                {model.best_metric_name}={model.best_metric_value ?? "—"}
              </div>
            ) : (
              <div className="text-[#64748b]">—</div>
            )}
          </div>
          {missionId && (
            <div>
              <div className="text-[10px] uppercase tracking-wider text-[#64748b] mb-1">Trained by</div>
              <Link
                href={`/missions/${missionId}`}
                className="text-[#14b8a6] hover:underline font-mono"
              >
                {missionId.slice(0, 8)}
              </Link>
            </div>
          )}
          {ckpt && (
            <div>
              <div className="text-[10px] uppercase tracking-wider text-[#64748b] mb-1">Checkpoint</div>
              <div className="font-mono text-[10px] text-[#94a3b8] break-all leading-relaxed">
                {ckpt}
              </div>
            </div>
          )}
        </aside>
      </div>
    </div>
  );
}
