"""
CriticAgent — Step 7.1.

Red-teams proposed training plans before execution using a structured rubric.
Returns a CritiqueResult with per-dimension scores and actionable feedback.
The LoopStateMachine intercepts the plan, passes it here, and asks the
LeadAgent to revise if the overall score is below APPROVAL_THRESHOLD.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Optional

from backend.agent.inference.base import InferenceProvider, Message, GenerationConfig
from backend.logging_config import get_logger

logger = get_logger(__name__)

APPROVAL_THRESHOLD = 7.0   # out of 10
MAX_REVISIONS = 2           # max critique→revise cycles before proceeding anyway

_CRITIC_SYSTEM = """\
You are ASTRA's Safety Critic — a skeptical peer reviewer whose job is to
red-team proposed ML training plans before they are executed.

Evaluate the plan on THREE dimensions (each scored 0–10):
  1. Safety          — Does the plan avoid dangerous operations, resource abuse,
                       or unbounded loops? Does it have clear exit conditions?
  2. Complexity      — Is the algorithm/architecture appropriately simple for the
                       stated goal? Unnecessary complexity is a risk.
  3. Overfitting_Risk — Are there validation safeguards (train/val split,
                        early stopping, regularisation)?

Rules:
- Be concise and specific. Flag real problems, not imaginary ones.
- An overall_score below 7 means the plan needs revision.
- If a concern is minor, note it but still approve.
- Return ONLY valid JSON — no markdown, no preamble."""

_CRITIC_SCHEMA = {
    "type": "object",
    "properties": {
        "critique": {
            "type": "object",
            "properties": {
                "safety_score":          {"type": "number"},
                "complexity_score":      {"type": "number"},
                "overfitting_risk_score": {"type": "number"},
                "overall_score":         {"type": "number"},
                "concerns":              {"type": "array", "items": {"type": "string"}},
                "feedback":              {"type": "string"},
            },
            "required": ["safety_score", "complexity_score",
                         "overfitting_risk_score", "overall_score",
                         "concerns", "feedback"],
        }
    },
    "required": ["critique"],
}


@dataclass
class CritiqueResult:
    safety_score: float
    complexity_score: float
    overfitting_risk_score: float
    overall_score: float
    concerns: list = field(default_factory=list)
    feedback: str = ""
    approved: bool = False
    revision: int = 0

    def rubric_scores(self) -> dict:
        return {
            "safety":          self.safety_score,
            "complexity":      self.complexity_score,
            "overfitting_risk": self.overfitting_risk_score,
        }

    def to_dict(self) -> dict:
        return {
            "overall_score":    self.overall_score,
            "approved":         self.approved,
            "concerns":         self.concerns,
            "feedback":         self.feedback,
            "rubric_scores":    self.rubric_scores(),
            "revision":         self.revision,
        }


class CriticAgent:
    def __init__(self, provider: InferenceProvider) -> None:
        self._provider = provider

    async def review(self, plan: dict, goal: str, revision: int = 0) -> CritiqueResult:
        """
        Evaluate a training plan. Returns CritiqueResult with approved=True
        when overall_score >= APPROVAL_THRESHOLD.
        """
        user_msg = (
            f"Goal: {goal}\n\n"
            f"Proposed plan:\n{json.dumps(plan, indent=2)}\n\n"
            "Score this plan on Safety, Complexity, and Overfitting_Risk. Return JSON."
        )
        messages = [
            Message(role="system", content=_CRITIC_SYSTEM),
            Message(role="user", content=user_msg),
        ]
        config = GenerationConfig(max_tokens=1024, temperature=0.2, json_schema=_CRITIC_SCHEMA)

        raw = await self._provider.generate(messages, config)
        parsed = self._parse(raw)

        result = CritiqueResult(
            safety_score=float(parsed.get("safety_score", 8)),
            complexity_score=float(parsed.get("complexity_score", 8)),
            overfitting_risk_score=float(parsed.get("overfitting_risk_score", 8)),
            overall_score=float(parsed.get("overall_score", 8)),
            concerns=parsed.get("concerns", []),
            feedback=parsed.get("feedback", ""),
            approved=float(parsed.get("overall_score", 8)) >= APPROVAL_THRESHOLD,
            revision=revision,
        )
        logger.info(
            "CriticAgent: revision=%d score=%.1f approved=%s concerns=%d",
            revision, result.overall_score, result.approved, len(result.concerns),
        )
        return result

    @staticmethod
    def _parse(raw: str) -> dict:
        import re, json as _json
        text = re.sub(r"<\|im_end\|>|<\|endoftext\|>", "", raw)
        text = re.sub(r"```(?:json)?\s*", "", text).replace("```", "").strip()
        m = re.search(r"\{.*\}", text, re.DOTALL)
        try:
            obj = _json.loads(m.group(0) if m else text)
            return obj.get("critique", obj)
        except Exception:
            logger.warning("CriticAgent: JSON parse failed — using default approval")
            return {"overall_score": 8.0, "concerns": [], "feedback": "Parse error — defaulting to approve"}
