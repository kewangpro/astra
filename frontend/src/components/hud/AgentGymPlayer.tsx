"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { PolicyInspector, PolicyTelemetry } from "./PolicyInspector";
import { policyPlayUrl } from "@/lib/playWs";

interface HistoryTurn {
  role?: string;
  tool?: string;
  content?: string;
  status?: string;
  result?: string;
}

interface Frame {
  type: "frame" | "episode_end" | "error";
  step?: number;
  episode?: number;
  episode_reward?: number;
  total_reward?: number;
  done?: boolean;
  message?: string;
  query?: string;
  scenario_name?: string;
  scenario_id?: string;
  sequence_progress?: number;
  required_steps?: number;
  task_success?: number;
  history?: HistoryTurn[];
  last_tool_result?: string;
  selected_action?: string;
  q_values?: Record<string, number>;
  action_probs?: Record<string, number>;
  entropy?: number | null;
}

interface Props {
  missionId?: string;
  modelId?: string;
  envId?: string;
}

export function AgentGymPlayer({
  missionId,
  modelId,
  envId = "MultiTurnAgentGym-v0",
}: Props) {
  const wsRef = useRef<WebSocket | null>(null);
  const [playing, setPlaying] = useState(false);
  const [episode, setEpisode] = useState(0);
  const [step, setStep] = useState(0);
  const [episodeReward, setEpisodeReward] = useState(0);
  const [bestReward, setBestReward] = useState<number | null>(null);
  const [speed, setSpeed] = useState(4);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [policyTelemetry, setPolicyTelemetry] = useState<PolicyTelemetry | null>(null);
  const [query, setQuery] = useState("");
  const [scenarioName, setScenarioName] = useState("");
  const [progress, setProgress] = useState("0/0");
  const [taskSuccess, setTaskSuccess] = useState(0);
  const [history, setHistory] = useState<HistoryTurn[]>([]);
  const [lastResult, setLastResult] = useState("");
  const [selectedAction, setSelectedAction] = useState("");

  const stop = useCallback(() => {
    wsRef.current?.close();
    wsRef.current = null;
    setPlaying(false);
    setLoading(false);
  }, []);

  const start = useCallback(() => {
    if (wsRef.current) return;
    setError(null);
    setLoading(true);

    const ws = new WebSocket(policyPlayUrl({ modelId, missionId }, envId, speed));
    wsRef.current = ws;

    ws.onmessage = (ev) => {
      let msg: Frame;
      try {
        msg = JSON.parse(ev.data);
      } catch {
        return;
      }
      if (msg.type === "error") {
        setError(msg.message || "Play stream error");
        stop();
        return;
      }
      if (msg.type === "episode_end") {
        setEpisode((e) => e + 1);
        if (typeof msg.total_reward === "number") {
          setBestReward((b) => (b == null ? msg.total_reward! : Math.max(b, msg.total_reward!)));
        }
        return;
      }
      if (msg.type === "frame") {
        setLoading(false);
        setPlaying(true);
        setStep(msg.step ?? 0);
        setEpisodeReward(msg.episode_reward ?? 0);
        setQuery(msg.query || "");
        setScenarioName(msg.scenario_name || msg.scenario_id || "");
        setProgress(`${msg.sequence_progress ?? 0}/${msg.required_steps ?? 0}`);
        setTaskSuccess(msg.task_success ?? 0);
        setHistory(msg.history || []);
        setLastResult(msg.last_tool_result || "");
        setSelectedAction(msg.selected_action || "");
        setPolicyTelemetry({
          q_values: msg.q_values,
          action_probs: msg.action_probs,
          entropy: msg.entropy ?? null,
          selected_action: msg.selected_action,
        });
      }
    };

    ws.onerror = () => {
      setError("WebSocket error");
      stop();
    };
    ws.onclose = () => {
      wsRef.current = null;
      setPlaying(false);
      setLoading(false);
    };
  }, [envId, missionId, modelId, speed, stop]);

  useEffect(() => () => stop(), [stop]);

  return (
    <div className="space-y-3">
      <div className="flex flex-wrap items-center gap-3 text-xs text-[#94a3b8]">
        <button
          type="button"
          onClick={playing ? stop : start}
          className="rounded border border-[#334155] bg-[#0f172a] px-3 py-1.5 text-[#e2e8f0] hover:border-teal-500/60"
        >
          {loading ? "Loading…" : playing ? "Stop" : "Play"}
        </button>
        <label className="flex items-center gap-1">
          FPS
          <input
            type="number"
            min={1}
            max={20}
            value={speed}
            disabled={playing}
            onChange={(e) => setSpeed(Number(e.target.value) || 4)}
            className="w-14 rounded border border-[#334155] bg-[#0f172a] px-1 py-0.5"
          />
        </label>
        <span>Ep {episode}</span>
        <span>Step {step}</span>
        <span>R {episodeReward.toFixed(2)}</span>
        {bestReward != null && <span>Best {bestReward.toFixed(2)}</span>}
        <span>Progress {progress}</span>
        <span className={taskSuccess >= 1 ? "text-teal-400" : ""}>
          Success {taskSuccess >= 1 ? "yes" : "no"}
        </span>
      </div>

      {error && (
        <div className="rounded border border-red-500/40 bg-red-950/30 px-3 py-2 text-xs text-red-300">
          {error}
        </div>
      )}

      <div className="rounded-lg border border-[#1e293b] bg-[#0b1220] p-4 space-y-3 min-h-[220px]">
        <div>
          <div className="text-[10px] uppercase tracking-wide text-[#64748b]">Scenario</div>
          <div className="text-sm text-[#e2e8f0]">{scenarioName || "—"}</div>
        </div>
        <div>
          <div className="text-[10px] uppercase tracking-wide text-[#64748b]">User query</div>
          <div className="text-sm text-[#cbd5e1] whitespace-pre-wrap">{query || "—"}</div>
        </div>
        <div>
          <div className="text-[10px] uppercase tracking-wide text-[#64748b]">
            Selected action
          </div>
          <div className="font-mono text-sm text-teal-300">{selectedAction || "—"}</div>
        </div>
        <div>
          <div className="text-[10px] uppercase tracking-wide text-[#64748b]">Dialogue</div>
          <ul className="mt-1 max-h-48 overflow-y-auto space-y-1 text-xs">
            {history.length === 0 && (
              <li className="text-[#64748b]">No turns yet.</li>
            )}
            {history.map((h, i) => (
              <li key={i} className="rounded bg-[#111827] px-2 py-1 text-[#94a3b8]">
                <span className="text-[#64748b]">[{h.role || "?"}] </span>
                {h.tool || h.content || ""}
                {h.status ? (
                  <span className="text-[#475569]"> · {h.status}</span>
                ) : null}
              </li>
            ))}
          </ul>
        </div>
        {lastResult ? (
          <div>
            <div className="text-[10px] uppercase tracking-wide text-[#64748b]">
              Last tool result
            </div>
            <div className="text-xs text-[#94a3b8]">{lastResult}</div>
          </div>
        ) : null}
      </div>

      <PolicyInspector telemetry={policyTelemetry} />
    </div>
  );
}
