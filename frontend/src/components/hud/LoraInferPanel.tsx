"use client";

import { useState } from "react";
import { api, ModelRecord } from "@/lib/api";

interface Props {
  model: ModelRecord;
}

export function LoraInferPanel({ model }: Props) {
  const [prompt, setPrompt] = useState("");
  const [completion, setCompletion] = useState<string | null>(null);
  const [promptUsed, setPromptUsed] = useState<string | null>(null);
  const [latencyMs, setLatencyMs] = useState<number | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  const handleGenerate = async () => {
    const text = prompt.trim();
    if (!text || loading) return;
    setLoading(true);
    setError(null);
    setCompletion(null);
    setPromptUsed(null);
    setLatencyMs(null);
    try {
      const res = await api.inferModel(model.id, { prompt: text });
      setCompletion(res.completion);
      setPromptUsed(res.prompt_used);
      setLatencyMs(res.latency_ms);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Inference failed");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="bg-[#1e293b] border border-[rgba(20,184,166,0.15)] rounded-lg p-5 space-y-4">
      <div className="flex items-center justify-between gap-3">
        <div>
          <div className="text-[10px] uppercase tracking-wider text-[#64748b]">
            Inference
          </div>
          <div className="text-sm text-[#e2e8f0] mt-0.5">
            {model.domain} · {model.framework || "mlx"}
          </div>
        </div>
        {latencyMs != null && (
          <div className="text-[10px] font-mono text-[#64748b]">
            {latencyMs} ms
          </div>
        )}
      </div>

      <div className="space-y-1.5">
        <label className="text-[10px] uppercase tracking-wider text-[#64748b]">
          Prompt
        </label>
        <textarea
          value={prompt}
          onChange={(e) => setPrompt(e.target.value)}
          rows={4}
          placeholder={
            model.domain.toLowerCase().includes("routing")
              ? "e.g. What is Apple’s stock price today?"
              : "Enter a prompt…"
          }
          className="w-full resize-y min-h-[96px] rounded-md bg-[#0f172a] border border-[#334155] px-3 py-2 text-sm text-[#e2e8f0] placeholder:text-[#475569] focus:outline-none focus:border-[#14b8a6]/50"
          disabled={loading}
          onKeyDown={(e) => {
            if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) {
              e.preventDefault();
              void handleGenerate();
            }
          }}
        />
      </div>

      <div className="flex items-center gap-2">
        <button
          type="button"
          onClick={() => void handleGenerate()}
          disabled={loading || !prompt.trim()}
          className="text-[11px] px-3 py-1.5 rounded border border-[#14b8a6]/40 text-[#14b8a6] hover:bg-[#14b8a6]/10 disabled:opacity-40 disabled:cursor-not-allowed"
        >
          {loading ? "Generating…" : "Generate"}
        </button>
        <span className="text-[10px] text-[#475569]">⌘↵ to run</span>
      </div>

      {error && (
        <div className="rounded-md border border-[#f87171]/30 bg-[#f87171]/10 px-3 py-2 text-xs text-[#f87171]">
          {error}
        </div>
      )}

      {(completion != null || loading) && (
        <div className="space-y-1.5">
          <div className="text-[10px] uppercase tracking-wider text-[#64748b]">
            Completion
          </div>
          <pre className="whitespace-pre-wrap break-words rounded-md bg-[#0f172a] border border-[#334155] px-3 py-2 text-xs text-[#e2e8f0] font-mono leading-relaxed min-h-[80px]">
            {loading ? "Waiting on Mini…" : completion}
          </pre>
        </div>
      )}

      {promptUsed && !loading && (
        <details className="text-[10px] text-[#64748b]">
          <summary className="cursor-pointer hover:text-[#94a3b8]">
            Prompt sent to model
          </summary>
          <pre className="mt-2 whitespace-pre-wrap break-words rounded-md bg-[#0f172a] border border-[#334155] px-3 py-2 text-[10px] text-[#94a3b8] font-mono max-h-48 overflow-y-auto">
            {promptUsed}
          </pre>
        </details>
      )}
    </div>
  );
}
