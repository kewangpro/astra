"""Unit tests for SmartKVCache in agent/kv_cache.py."""
from __future__ import annotations

import pytest

from backend.agent.kv_cache import SmartKVCache, _estimate_tokens, MAX_HISTORY_TOKENS, CODE_PRESERVED_TOKENS


# ── _estimate_tokens ──────────────────────────────────────────────────────────

class TestEstimateTokens:
    def test_empty_string_returns_one(self):
        assert _estimate_tokens("") == 1

    def test_four_chars_is_one_token(self):
        assert _estimate_tokens("abcd") == 1

    def test_eight_chars_is_two_tokens(self):
        assert _estimate_tokens("abcdefgh") == 2

    def test_long_text(self):
        text = "x" * 400
        assert _estimate_tokens(text) == 100


# ── SmartKVCache ──────────────────────────────────────────────────────────────

class TestSmartKVCache:
    def setup_method(self):
        self.cache = SmartKVCache(max_history_tokens=100, max_code_tokens=50)

    def test_empty_cache_returns_no_messages(self):
        assert self.cache.get_messages() == []

    def test_system_prompt_included(self):
        self.cache.set_system_prompt("You are a helpful assistant.")
        msgs = self.cache.get_messages()
        assert len(msgs) == 1
        assert msgs[0].role == "system"

    def test_system_prompt_replaced_on_second_call(self):
        self.cache.set_system_prompt("first")
        self.cache.set_system_prompt("second")
        msgs = self.cache.get_messages()
        assert len(msgs) == 1
        assert msgs[0].content == "second"

    def test_add_turn_appears_in_messages(self):
        self.cache.add_turn("user", "hello")
        self.cache.add_turn("assistant", "hi")
        msgs = self.cache.get_messages()
        assert len(msgs) == 2
        assert msgs[0].role == "user"
        assert msgs[1].role == "assistant"

    def test_query_appended_as_last_message(self):
        self.cache.add_turn("user", "old message")
        msgs = self.cache.get_messages(query="new query")
        assert msgs[-1].content == "new query"
        assert msgs[-1].role == "user"

    def test_history_eviction_when_over_budget(self):
        # Each message = 40 chars = 10 tokens; budget = 100 tokens
        # After 11 messages the oldest should be evicted
        self.cache = SmartKVCache(max_history_tokens=100, max_code_tokens=50)
        msg = "a" * 40  # 10 tokens each
        for i in range(11):
            self.cache.add_turn("user", msg)
        msgs = self.cache.get_messages()
        # Should have evicted at least 1
        assert len(msgs) <= 10

    def test_token_budget_does_not_exceed_max(self):
        self.cache = SmartKVCache(max_history_tokens=100, max_code_tokens=50)
        for _ in range(20):
            self.cache.add_turn("user", "a" * 40)  # 10 tokens each
        assert self.cache._history_token_count <= 100

    def test_add_code_context_appears_between_system_and_history(self):
        self.cache.set_system_prompt("sys")
        self.cache.add_code_context("code snippet")
        self.cache.add_turn("user", "question")
        msgs = self.cache.get_messages()
        roles = [m.role for m in msgs]
        # system first, then user (code), then user (history)
        assert roles[0] == "system"
        assert roles[1] == "user"   # code context role is user
        assert roles[2] == "user"   # history

    def test_code_context_eviction_when_over_budget(self):
        self.cache = SmartKVCache(max_history_tokens=100, max_code_tokens=40)
        chunk = "c" * 80  # 20 tokens
        self.cache.add_code_context(chunk)
        self.cache.add_code_context(chunk)
        self.cache.add_code_context(chunk)
        assert self.cache._code_token_count <= 40

    def test_flush_code_context_clears_code(self):
        self.cache.add_code_context("some code")
        self.cache.flush_code_context()
        assert self.cache._code_token_count == 0
        assert len(list(self.cache._code)) == 0

    def test_flush_does_not_clear_history(self):
        self.cache.add_turn("user", "preserved")
        self.cache.flush_code_context()
        msgs = self.cache.get_messages()
        assert any(m.content == "preserved" for m in msgs)

    def test_clear_history_empties_history(self):
        self.cache.add_turn("user", "msg1")
        self.cache.add_turn("assistant", "msg2")
        self.cache.clear_history()
        assert self.cache._history_token_count == 0
        assert self.cache.get_messages() == []

    def test_token_budget_used_accounts_all_buckets(self):
        self.cache.set_system_prompt("s" * 40)   # ~10 tokens
        self.cache.add_code_context("c" * 40)    # ~10 tokens
        self.cache.add_turn("user", "h" * 40)    # ~10 tokens
        used = self.cache.token_budget_used()
        assert used >= 30

    def test_token_budget_used_starts_at_zero(self):
        fresh = SmartKVCache()
        assert fresh.token_budget_used() == 0

    def test_message_order_system_code_history_query(self):
        self.cache.set_system_prompt("SYS")
        self.cache.add_code_context("CODE")
        self.cache.add_turn("user", "HIST")
        msgs = self.cache.get_messages(query="QUERY")
        contents = [m.content for m in msgs]
        assert contents == ["SYS", "CODE", "HIST", "QUERY"]
