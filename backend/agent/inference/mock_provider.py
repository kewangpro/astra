"""
MockProvider — deterministic responses for testing and development.
No model weights required; returns scripted JSON or code based on the last
user message keywords.
"""
from __future__ import annotations

import json
from typing import Optional

from backend.agent.inference.base import InferenceProvider, Message, GenerationConfig
from backend.logging_config import get_logger

logger = get_logger(__name__)


class MockProvider(InferenceProvider):
    def __init__(self) -> None:
        self._loaded = True

    def is_loaded(self) -> bool:
        return self._loaded

    def unload(self) -> None:
        self._loaded = False

    @property
    def model_id(self) -> str:
        return "mock"

    async def generate(self, messages: list[Message], config: Optional[GenerationConfig] = None) -> str:
        last = messages[-1].content.lower() if messages else ""

        if config and config.json_schema:
            return self._mock_json(last)
        return self._mock_text(last)

    @staticmethod
    def _mock_json(prompt: str) -> str:
        if "plan" in prompt or "goal" in prompt:
            return json.dumps({
                "plan": {
                    "task_type": "rl",
                    "algorithm": "PPO",
                    "hyperparameters": {
                        "learning_rate": 3e-4,
                        "gamma": 0.99,
                        "batch_size": 64,
                        "total_timesteps": 300000,
                    },
                    "curriculum_phases": [
                        {"name": "phase_1", "success_threshold": {"mean_reward": 20}},
                        {"name": "phase_2", "success_threshold": {"mean_reward": 80}},
                    ],
                    "estimated_iterations": 5,
                }
            })
        if "pivot" in prompt or "adjust" in prompt:
            return json.dumps({
                "pivot": {
                    "reason": "plateau_detected",
                    "adjustments": {"learning_rate": 1e-4, "entropy_coef": 0.05},
                }
            })
        return json.dumps({"result": "ok"})

    @staticmethod
    def _mock_text(prompt: str) -> str:
        if "fix" in prompt or "error" in prompt:
            return "The error is caused by a missing import. Add `import numpy as np` at the top."
        if "script" in prompt or "code" in prompt:
            return (
                "import gymnasium as gym\n"
                "env = gym.make('CartPole-v1')\n"
                "obs, _ = env.reset()\n"
                "print('Mock training script generated')\n"
            )
        return "Mock response: task acknowledged."
