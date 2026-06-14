from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Message:
    role: str   # "system" | "user" | "assistant"
    content: str


@dataclass
class GenerationConfig:
    max_tokens: int = 2048
    temperature: float = 0.2
    top_p: float = 0.95
    # When set, the provider must return valid JSON matching this schema.
    # Implementations may use grammar-based sampling or retry-on-parse-error.
    json_schema: Optional[dict] = None


class InferenceProvider(ABC):
    """Unified interface for LLM backends (MLX, vLLM, Mock)."""

    @abstractmethod
    async def generate(self, messages: list[Message], config: Optional[GenerationConfig] = None) -> str:
        """Return the assistant's response text."""

    @abstractmethod
    def is_loaded(self) -> bool:
        """True if a model is currently resident in memory."""

    @abstractmethod
    def unload(self) -> None:
        """Release model weights from memory (for ModelManager GC)."""

    @property
    @abstractmethod
    def model_id(self) -> str:
        """Human-readable identifier of the loaded model."""
