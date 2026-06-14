"""
Smart KV Cache — eviction policy for the Lead Agent's context window.

Strategy: preserve system prompt and code context; evict the oldest
conversation history turns when the context approaches the token budget.

This wraps the message list that gets passed to InferenceProvider.generate()
and ensures we never exceed the model's context window while retaining
the highest-value context.
"""
from __future__ import annotations

from collections import deque
from typing import Optional

from backend.agent.inference.base import Message

# Rough token estimates (characters / 4)
SYSTEM_PRESERVED_TOKENS = 2048    # always keep system prompt
CODE_PRESERVED_TOKENS = 4096      # always keep generated code context
MAX_HISTORY_TOKENS = 4096         # sliding window for conversation history


def _estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


class SmartKVCache:
    """
    Maintains three message buckets with different retention policies:
      - system:  pinned, never evicted
      - code:    pinned until a new training iteration starts
      - history: sliding window, oldest turns evicted first
    """

    def __init__(
        self,
        max_history_tokens: int = MAX_HISTORY_TOKENS,
        max_code_tokens: int = CODE_PRESERVED_TOKENS,
    ) -> None:
        self._system: list[Message] = []
        self._code: deque[Message] = deque()
        self._history: deque[Message] = deque()
        self._max_history_tokens = max_history_tokens
        self._max_code_tokens = max_code_tokens
        self._history_token_count = 0
        self._code_token_count = 0

    def set_system_prompt(self, content: str) -> None:
        self._system = [Message(role="system", content=content)]

    def add_code_context(self, content: str) -> None:
        """Add generated code or log snippets to the pinned code context."""
        msg = Message(role="user", content=content)
        tokens = _estimate_tokens(content)
        self._code.append(msg)
        self._code_token_count += tokens
        # Evict oldest code context if over budget
        while self._code_token_count > self._max_code_tokens and self._code:
            evicted = self._code.popleft()
            self._code_token_count -= _estimate_tokens(evicted.content)

    def add_turn(self, role: str, content: str) -> None:
        """Add a conversation turn to the history sliding window."""
        msg = Message(role=role, content=content)
        tokens = _estimate_tokens(content)
        self._history.append(msg)
        self._history_token_count += tokens
        # Evict oldest history turns if over budget
        while self._history_token_count > self._max_history_tokens and self._history:
            evicted = self._history.popleft()
            self._history_token_count -= _estimate_tokens(evicted.content)

    def flush_code_context(self) -> None:
        """Clear code context at the start of a new training iteration."""
        self._code.clear()
        self._code_token_count = 0

    def get_messages(self, query: Optional[str] = None) -> list[Message]:
        """
        Build the final message list for the LLM call:
        system → code context → history → (optional) new query.
        """
        messages = list(self._system) + list(self._code) + list(self._history)
        if query:
            messages.append(Message(role="user", content=query))
        return messages

    def token_budget_used(self) -> int:
        system_tokens = sum(_estimate_tokens(m.content) for m in self._system)
        return system_tokens + self._code_token_count + self._history_token_count

    def clear_history(self) -> None:
        self._history.clear()
        self._history_token_count = 0
