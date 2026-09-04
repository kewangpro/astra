"""Unit tests for LeadAgent.propose_pivot()'s escalation-level-4 (deep plateau) prompt."""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from backend.agent.lead_agent import LeadAgent


def _agent() -> LeadAgent:
    provider = MagicMock()
    agent = LeadAgent(provider)
    agent._generate_structured = AsyncMock(return_value={"reason": "x", "adjustments": {}})
    return agent


class TestDeepPlateauEscalation:
    @pytest.mark.asyncio
    async def test_level_four_prompt_lists_tried_architectures(self):
        """Real incident: 111 pivots cycled the same 3 architectures because the
        LLM was never told what had already failed beyond a 5-item window.
        Level 4 must explicitly enumerate the full tried-architecture history."""
        agent = _agent()
        tried = [{"net_arch": [400, 300]}, {"net_arch": [256, 256, 128]}, {"net_arch": [256, 256]}]
        await agent.propose_pivot(
            {"food_eaten": 40.0}, [], escalation_level=4,
            current_algorithm="DQN", algorithm_locked=True,
            tried_architectures=tried,
        )
        query = agent._generate_structured.call_args.args[0][-1].content
        assert "DEEP PLATEAU" in query
        for arch in tried:
            assert str(arch["net_arch"]) in query or repr(arch) in query or "net_arch" in query

    @pytest.mark.asyncio
    async def test_level_four_without_tried_architectures_still_describes_deep_plateau(self):
        agent = _agent()
        await agent.propose_pivot(
            {"food_eaten": 40.0}, [], escalation_level=4,
            current_algorithm="DQN", algorithm_locked=True,
        )
        query = agent._generate_structured.call_args.args[0][-1].content
        assert "DEEP PLATEAU" in query

    @pytest.mark.asyncio
    async def test_level_three_does_not_mention_deep_plateau(self):
        agent = _agent()
        await agent.propose_pivot(
            {"food_eaten": 40.0}, [], escalation_level=3,
            current_algorithm="DQN", algorithm_locked=True,
        )
        query = agent._generate_structured.call_args.args[0][-1].content
        assert "DEEP PLATEAU" not in query

    @pytest.mark.asyncio
    async def test_level_four_not_algorithm_locked_suggests_algorithm_switch(self):
        agent = _agent()
        await agent.propose_pivot(
            {"food_eaten": 40.0}, [], escalation_level=4,
            current_algorithm="DQN", algorithm_locked=False,
        )
        query = agent._generate_structured.call_args.args[0][-1].content
        assert "switching algorithm" in query.lower()

    @pytest.mark.asyncio
    async def test_unrecognized_escalation_level_falls_back_to_level_zero(self):
        """Guards against a silent regression: if PivotEngine ever returns a level
        higher than what the local escalation_desc dict knows about, it must not
        crash or silently drop escalation guidance — falls back to level 0 text."""
        agent = _agent()
        await agent.propose_pivot(
            {"food_eaten": 40.0}, [], escalation_level=99,
            current_algorithm="DQN", algorithm_locked=True,
        )
        query = agent._generate_structured.call_args.args[0][-1].content
        assert "Level 0" in query


class TestDpoGrpoEscalation:
    """Real incident: a DPO mission's pivots were complete no-ops for 47+ consecutive
    iterations — the escalation guidance told the LLM to try RL concepts (reward shaping,
    algorithm switches) that don't apply and always get silently discarded. Sampling-diversity
    knobs (temp/k_collect/num_generations) are now the one lever that actually reaches
    training; the prompt must steer the LLM toward exactly that, at every level."""

    @pytest.mark.asyncio
    async def test_dpo_escalation_never_directs_toward_rl_actions(self):
        agent = _agent()
        for level in (0, 1, 2, 3, 4):
            await agent.propose_pivot(
                {"pass_rate": 0.75}, [], escalation_level=level,
                current_algorithm="DPO", algorithm_locked=True,
            )
            query = agent._generate_structured.call_args.args[0][-1].content
            # the old RL-flavored text (concrete action items, not just the words
            # "reward"/"architecture" in passing) must never appear for dpo/grpo
            assert "food_reward" not in query
            assert "net_arch" not in query
            assert "distance_weight" not in query
            assert "k_collect" in query

    @pytest.mark.asyncio
    async def test_grpo_escalation_mentions_num_generations_not_k_collect(self):
        agent = _agent()
        await agent.propose_pivot(
            {"pass_rate": 0.75}, [], escalation_level=4,
            current_algorithm="GRPO", algorithm_locked=True,
        )
        query = agent._generate_structured.call_args.args[0][-1].content
        assert "num_generations" in query
        assert "k_collect" not in query

    @pytest.mark.asyncio
    async def test_distill_escalation_mentions_iters_only(self):
        agent = _agent()
        await agent.propose_pivot(
            {"pass_rate": 0.75}, [], escalation_level=4,
            current_algorithm="Distill", algorithm_locked=True,
        )
        query = agent._generate_structured.call_args.args[0][-1].content
        assert "iters" in query
        assert "k_collect" not in query
        assert "num_generations" not in query
        assert "food_reward" not in query and "net_arch" not in query

    @pytest.mark.asyncio
    async def test_dpo_escalation_case_insensitive_algorithm(self):
        agent = _agent()
        await agent.propose_pivot(
            {"pass_rate": 0.75}, [], escalation_level=4,
            current_algorithm="dpo", algorithm_locked=True,
        )
        query = agent._generate_structured.call_args.args[0][-1].content
        assert "k_collect" in query
