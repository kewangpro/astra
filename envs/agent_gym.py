"""
MultiTurnAgentGym-v0: A Gymnasium environment for multi-turn agent tool-use and trajectory RL.

Simulates interactive agent workflows where an agent processes user queries by orchestrating
sequences of tools (e.g., database lookup, calculation, knowledge base retrieval, email notification,
human escalation) across multiple turns before submitting a final solution.

Observation Space (obs_mode=\"vector\", default):
  Box(low=0.0, high=1.0, shape=(32,), dtype=np.float32)
  Normalized 32D dialogue state feature vector:
    [0]: Normalized turn count (current_step / max_steps)
    [1:9]: One-hot encoding of active scenario type (8 slots)
    [9:17]: One-hot encoding of last executed action (8 slots)
    [17]: Last tool execution status (1.0 = success, 0.0 = error/initial)
    [18]: Goal progression fraction (completed_steps / total_required_steps)
    [19:27]: Bitmask of tools executed so far in episode
    [27]: Repetition penalty flag (1.0 if agent called identical tool consecutively)
    [28:32]: Scenario context flags [requires_email, requires_calc, is_escalation, is_blocked]

Observation Space (obs_mode=\"text\"):
  Text string with query, last tool result, progress, and remaining step budget.
  Used by LLM trajectory RL (Agent GRPO); SB3 MLP policies use vector mode.

Action Space:
  Discrete(8)
    0: FINISH_TASK      (Submit final response and conclude task)
    1: QUERY_DATABASE   (Query order, inventory, or account records)
    2: SEND_EMAIL       (Send confirmation, receipt, or alert email)
    3: SEARCH_KB        (Search documentation, knowledge base, policies)
    4: CALCULATE        (Compute prices, taxes, discounts, or metrics)
    5: ESCALATE_HUMAN   (Escalate blocked or sensitive case to human operator)
    6: RETRY_TOOL       (Retry previous action with alternative parameters)
    7: CLARIFY_QUERY    (Request clarification on ambiguous parameters)

Reward Shaping:
  +step_reward (default 0.5): Correct tool invocation matching next scenario milestone
  -invalid_penalty (default 0.2): Redundant or irrelevant tool call
  -0.1: Extra penalty for repeating the same invalid tool consecutively
  -turn_penalty (default 0.05): Per-step cost (encourages concise execution)
  -1.0: Premature task completion before required milestones
  +completion_reward (default 2.0): Terminal task success bonus when all milestones verified
"""
from __future__ import annotations

import json
import string
from typing import Optional, Dict, Any, List, Tuple, Union
import numpy as np
import gymnasium as gym
from gymnasium import spaces

# Broad charset so tool-result text obs passes Gymnasium's Text space checker.
_TEXT_CHARSET = string.printable + "—–""''…€"

ACTION_FINISH = 0
ACTION_QUERY_DB = 1
ACTION_SEND_EMAIL = 2
ACTION_SEARCH_KB = 3
ACTION_CALCULATE = 4
ACTION_ESCALATE = 5
ACTION_RETRY = 6
ACTION_CLARIFY = 7

ACTION_NAMES = [
    "finish_task",
    "query_database",
    "send_email",
    "search_kb",
    "calculate",
    "escalate_human",
    "retry_tool",
    "clarify_query",
]

TOOL_RESULT_TEMPLATES = {
    ACTION_QUERY_DB: "Database result: records retrieved successfully for the active query.",
    ACTION_SEND_EMAIL: "Email sent successfully to the recipient.",
    ACTION_SEARCH_KB: "Knowledge base hit: relevant policy article returned.",
    ACTION_CALCULATE: "Calculation complete: numeric result computed.",
    ACTION_ESCALATE: "Escalation ticket opened for human operator.",
    ACTION_RETRY: "Retry succeeded after transient failure.",
    ACTION_CLARIFY: "Clarification request sent to the user.",
    ACTION_FINISH: "Task marked complete.",
}

SCENARIOS = [
    {
        "id": "order_delay_notify",
        "name": "Order Status & Delay Notification",
        "query": "Check status of order #8492 and notify the customer by email if it is delayed.",
        "required_sequence": [ACTION_QUERY_DB, ACTION_SEND_EMAIL, ACTION_FINISH],
        "context_flags": [1.0, 0.0, 0.0, 0.0],
    },
    {
        "id": "kb_refund_calc",
        "name": "Refund Policy Search & Calculation",
        "query": "Look up our return window policy for damaged items and compute the 80% prorated refund on $150.",
        "required_sequence": [ACTION_SEARCH_KB, ACTION_CALCULATE, ACTION_FINISH],
        "context_flags": [0.0, 1.0, 0.0, 0.0],
    },
    {
        "id": "troubleshoot_alert",
        "name": "System Triage & Team Alert",
        "query": "Search knowledge base for error code 503, query database for affected accounts, and email the ops team.",
        "required_sequence": [ACTION_SEARCH_KB, ACTION_QUERY_DB, ACTION_SEND_EMAIL, ACTION_FINISH],
        "context_flags": [1.0, 0.0, 0.0, 1.0],
    },
    {
        "id": "billing_tax_update",
        "name": "Billing Query & Tax Calculation",
        "query": "Fetch customer invoice #3301 and compute total after applying 8.5% sales tax.",
        "required_sequence": [ACTION_QUERY_DB, ACTION_CALCULATE, ACTION_FINISH],
        "context_flags": [0.0, 1.0, 0.0, 0.0],
    },
    {
        "id": "escalate_security",
        "name": "Security Breach Escalation",
        "query": "Search policy for unauthorized access attempts and escalate immediately to the human security lead.",
        "required_sequence": [ACTION_SEARCH_KB, ACTION_ESCALATE, ACTION_FINISH],
        "context_flags": [0.0, 0.0, 1.0, 1.0],
    },
    {
        "id": "order_clarify_query",
        "name": "Ambiguous Order Clarification",
        "query": "Customer says 'check my recent item' without providing order ID or email.",
        "required_sequence": [ACTION_CLARIFY, ACTION_FINISH],
        "context_flags": [0.0, 0.0, 0.0, 1.0],
    },
    {
        "id": "db_retry_resync",
        "name": "Database Timeout & Retry",
        "query": "Query inventory table with retry on initial transient network timeout.",
        "required_sequence": [ACTION_QUERY_DB, ACTION_RETRY, ACTION_FINISH],
        "context_flags": [0.0, 0.0, 0.0, 1.0],
    },
    {
        "id": "full_triage_flow",
        "name": "End-to-End Customer Support Triage",
        "query": "Look up return policy, query customer purchases, compute refund amount, and email return label.",
        "required_sequence": [ACTION_SEARCH_KB, ACTION_QUERY_DB, ACTION_CALCULATE, ACTION_SEND_EMAIL, ACTION_FINISH],
        "context_flags": [1.0, 1.0, 0.0, 0.0],
    },
]

ObsType = Union[np.ndarray, str]


class AgentToolGym(gym.Env):
    metadata = {"render_modes": ["text", "human"]}

    def __init__(
        self,
        max_steps: int = 8,
        render_mode: Optional[str] = None,
        turn_penalty: float = 0.05,
        step_reward: float = 0.5,
        completion_reward: float = 2.0,
        invalid_penalty: float = 0.2,
        scenario_idx: Optional[int] = None,
        obs_mode: str = "vector",
        tool_failure_prob: float = 0.0,
    ):
        super().__init__()
        self.max_steps = max_steps
        self.render_mode = render_mode
        self.turn_penalty = turn_penalty
        self.step_reward = step_reward
        self.completion_reward = completion_reward
        self.invalid_penalty = invalid_penalty
        self.fixed_scenario_idx = scenario_idx
        self.obs_mode = (obs_mode or "vector").lower()
        if self.obs_mode not in ("vector", "text"):
            raise ValueError(f"obs_mode must be 'vector' or 'text', got {obs_mode!r}")
        self.tool_failure_prob = float(tool_failure_prob)

        self.action_space = spaces.Discrete(len(ACTION_NAMES))
        if self.obs_mode == "text":
            self.observation_space = spaces.Text(
                max_length=4096, min_length=1, charset=_TEXT_CHARSET
            )
        else:
            self.observation_space = spaces.Box(
                low=0.0, high=1.0, shape=(32,), dtype=np.float32
            )

        self._step_count = 0
        self._current_scenario_idx = 0
        self._scenario: Dict[str, Any] = {}
        self._required_sequence: List[int] = []
        self._sequence_progress = 0
        self._last_action: Optional[int] = None
        self._last_action_success = 0.0
        self._executed_tools_mask = [0.0] * 8
        self._history: List[Dict[str, Any]] = []
        self._task_success = False
        self._last_tool_result = ""
        self._repetition_flag = 0.0

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> Tuple[ObsType, Dict[str, Any]]:
        super().reset(seed=seed)
        self._step_count = 0
        self._sequence_progress = 0
        self._last_action = None
        self._last_action_success = 0.0
        self._executed_tools_mask = [0.0] * 8
        self._task_success = False
        self._history = []
        self._last_tool_result = ""
        self._repetition_flag = 0.0

        if self.fixed_scenario_idx is not None:
            self._current_scenario_idx = self.fixed_scenario_idx % len(SCENARIOS)
        elif options and "scenario_idx" in options:
            self._current_scenario_idx = int(options["scenario_idx"]) % len(SCENARIOS)
        else:
            self._current_scenario_idx = int(self.np_random.integers(0, len(SCENARIOS)))

        self._scenario = SCENARIOS[self._current_scenario_idx]
        self._required_sequence = list(self._scenario["required_sequence"])

        self._history.append({
            "role": "user",
            "content": self._scenario["query"],
        })

        obs = self._get_obs()
        info = self._base_info()
        info.update({
            "required_steps": len(self._required_sequence),
            "task_success": 0.0,
            "dialogue": self.dialogue_state(),
        })
        return obs, info

    def step(self, action: int | Dict[str, Any] | str) -> Tuple[ObsType, float, bool, bool, Dict[str, Any]]:
        # Support text or dict actions for LLM / structured agents
        if isinstance(action, str):
            try:
                parsed = json.loads(action)
                action = self._parse_action_dict(parsed)
            except Exception:
                action = self._parse_action_name(action)
        elif isinstance(action, dict):
            action = self._parse_action_dict(action)

        action = int(action)
        self._step_count += 1
        reward = -self.turn_penalty
        terminated = False
        truncated = False
        self._repetition_flag = 0.0

        repetition = (self._last_action is not None and self._last_action == action)
        if 0 <= action < 8:
            self._executed_tools_mask[action] = 1.0

        expected_action = (
            self._required_sequence[self._sequence_progress]
            if self._sequence_progress < len(self._required_sequence)
            else None
        )

        if action == expected_action:
            # Optional stochastic tool failure (does not advance milestone)
            failed = (
                action != ACTION_FINISH
                and self.tool_failure_prob > 0.0
                and float(self.np_random.random()) < self.tool_failure_prob
            )
            if failed:
                self._last_action_success = 0.0
                self._last_tool_result = (
                    f"Tool {ACTION_NAMES[action]} failed: transient error. Retry or choose another tool."
                )
                reward -= self.invalid_penalty
                self._history.append({
                    "role": "assistant",
                    "tool": ACTION_NAMES[action],
                    "status": "tool_failure",
                    "result": self._last_tool_result,
                })
            else:
                self._sequence_progress += 1
                reward += self.step_reward
                self._last_action_success = 1.0
                self._last_tool_result = TOOL_RESULT_TEMPLATES.get(
                    action, f"Tool {ACTION_NAMES[action]} succeeded."
                )
                self._history.append({
                    "role": "assistant",
                    "tool": ACTION_NAMES[action],
                    "status": "success",
                    "result": self._last_tool_result,
                })

                if action == ACTION_FINISH:
                    if self._sequence_progress == len(self._required_sequence):
                        reward += self.completion_reward
                        self._task_success = True
                    terminated = True
        else:
            self._last_action_success = 0.0
            if action == ACTION_FINISH:
                reward -= 1.0
                terminated = True
                self._last_tool_result = "Premature finish: required milestones incomplete."
                self._history.append({
                    "role": "assistant",
                    "tool": "finish_task",
                    "status": "premature_failure",
                    "result": self._last_tool_result,
                })
            else:
                reward -= self.invalid_penalty
                if repetition:
                    reward -= 0.1
                    self._repetition_flag = 1.0
                tool_name = ACTION_NAMES[action] if 0 <= action < len(ACTION_NAMES) else f"unknown_{action}"
                self._last_tool_result = (
                    f"Tool {tool_name} was invalid or out of order for this scenario."
                )
                self._history.append({
                    "role": "assistant",
                    "tool": tool_name,
                    "status": "invalid_or_out_of_order",
                    "result": self._last_tool_result,
                })

        self._last_action = action

        if self._step_count >= self.max_steps and not terminated:
            truncated = True

        obs = self._get_obs()
        info = self._base_info()
        info.update({
            "step_count": self._step_count,
            "sequence_progress": self._sequence_progress,
            "required_steps": len(self._required_sequence),
            "task_success": 1.0 if self._task_success else 0.0,
            "last_action_name": ACTION_NAMES[action] if 0 <= action < len(ACTION_NAMES) else str(action),
            "last_tool_result": self._last_tool_result,
            "dialogue": self.dialogue_state(),
        })

        return obs, reward, terminated, truncated, info

    def dialogue_state(self) -> Dict[str, Any]:
        """Structured dialogue payload for play HUD / LM bridge."""
        return {
            "scenario_id": self._scenario.get("id", ""),
            "scenario_name": self._scenario.get("name", ""),
            "query": self._scenario.get("query", ""),
            "history": list(self._history),
            "sequence_progress": self._sequence_progress,
            "required_steps": len(self._required_sequence),
            "required_tools": [ACTION_NAMES[a] for a in self._required_sequence],
            "steps_remaining": max(0, self.max_steps - self._step_count),
            "task_success": bool(self._task_success),
            "last_tool_result": self._last_tool_result,
        }

    def _base_info(self) -> Dict[str, Any]:
        return {
            "scenario_id": self._scenario.get("id", ""),
            "scenario_name": self._scenario.get("name", ""),
            "query": self._scenario.get("query", ""),
            "obs_mode": self.obs_mode,
        }

    def _get_obs(self) -> ObsType:
        if self.obs_mode == "text":
            return self._get_text_obs()
        return self._get_vector_obs()

    def _get_text_obs(self) -> str:
        progress = f"{self._sequence_progress}/{max(1, len(self._required_sequence))}"
        remaining = max(0, self.max_steps - self._step_count)
        tools_used = [
            ACTION_NAMES[i] for i, flag in enumerate(self._executed_tools_mask) if flag
        ]
        lines = [
            f"Scenario: {self._scenario.get('name', '')}",
            f"User query: {self._scenario.get('query', '')}",
            f"Progress: {progress} milestones | Steps remaining: {remaining}",
            f"Tools used: {', '.join(tools_used) if tools_used else '(none)'}",
        ]
        if self._last_tool_result:
            lines.append(f"Last tool result: {self._last_tool_result}")
        else:
            lines.append("Last tool result: (none yet)")
        lines.append(
            "Respond with a single JSON object: "
            '{"tool": "<one of: ' + ", ".join(ACTION_NAMES) + '>"}'
        )
        return "\n".join(lines)

    def _get_vector_obs(self) -> np.ndarray:
        obs = np.zeros(32, dtype=np.float32)
        obs[0] = min(1.0, self._step_count / max(1, self.max_steps))

        if 0 <= self._current_scenario_idx < 8:
            obs[1 + self._current_scenario_idx] = 1.0

        if self._last_action is not None and 0 <= self._last_action < 8:
            obs[9 + self._last_action] = 1.0

        obs[17] = self._last_action_success

        total_req = max(1, len(self._required_sequence))
        obs[18] = min(1.0, self._sequence_progress / total_req)

        obs[19:27] = self._executed_tools_mask
        obs[27] = self._repetition_flag

        context_flags = self._scenario.get("context_flags", [0.0, 0.0, 0.0, 0.0])
        obs[28:32] = context_flags[:4]

        return obs

    def _parse_action_dict(self, d: dict) -> int:
        name = d.get("name") or d.get("tool") or d.get("action") or ""
        return self._parse_action_name(name)

    def _parse_action_name(self, name: str) -> int:
        cleaned = str(name).strip().lower().replace("-", "_")
        for idx, act in enumerate(ACTION_NAMES):
            if act in cleaned or cleaned in act:
                return idx
        return ACTION_FINISH

    def render(self) -> str:
        lines = [f"=== Scenario: {self._scenario.get('name', 'Unknown')} ==="]
        lines.append(f"Query: {self._scenario.get('query', '')}")
        lines.append("Dialogue Steps:")
        for h in self._history:
            lines.append(
                f"  [{h.get('role')}] {h.get('tool', h.get('content', ''))} "
                f"(status: {h.get('status', 'ok')})"
            )
        lines.append(
            f"Progress: {self._sequence_progress}/{len(self._required_sequence)} "
            f"| Success: {self._task_success}"
        )
        output = "\n".join(lines)
        if self.render_mode == "human":
            print(output)
        return output


def register():
    """Register MultiTurnAgentGym-v0, AgentGym-v0, and text observation alias."""
    if "MultiTurnAgentGym-v0" not in gym.envs.registry:
        gym.register(
            id="MultiTurnAgentGym-v0",
            entry_point="envs.agent_gym:AgentToolGym",
            kwargs={"max_steps": 8, "obs_mode": "vector"},
        )
    if "AgentGym-v0" not in gym.envs.registry:
        gym.register(
            id="AgentGym-v0",
            entry_point="envs.agent_gym:AgentToolGym",
            kwargs={"max_steps": 8, "obs_mode": "vector"},
        )
    if "MultiTurnAgentGymText-v0" not in gym.envs.registry:
        gym.register(
            id="MultiTurnAgentGymText-v0",
            entry_point="envs.agent_gym:AgentToolGym",
            kwargs={"max_steps": 8, "obs_mode": "text"},
        )
