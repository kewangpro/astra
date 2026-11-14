"""Unit tests for astrarl rollout backends + DAPO-lite helpers (no MLX weights)."""
from __future__ import annotations

import json
from unittest.mock import patch

from astrarl.grpo_math import compute_advantages
from astrarl.mlx_grpo_step import collect_group
from astrarl.rollout import (
    OpenAICompatRolloutBackend,
    asymmetric_clip_ratio,
    make_rollout_backend,
    resolve_clip_epsilons,
    rewards_are_flat,
)


class _DummyPlugin:
    def build_prompt(self, case: dict) -> str:
        return f"P:{case['id']}"

    def score(self, completion: str, case: dict) -> float:
        return float(completion) if completion else 0.0


class _StubRollout:
    def __init__(self, batches: list[list[str]]):
        self.batches = list(batches)
        self.calls = 0

    def generate(self, prompt, *, n, temperature, max_tokens):
        out = self.batches[min(self.calls, len(self.batches) - 1)]
        self.calls += 1
        return list(out)[:n]


def test_asymmetric_clip():
    assert asymmetric_clip_ratio(0.5, 0.2, 0.28) == 0.8
    assert asymmetric_clip_ratio(1.5, 0.2, 0.28) == 1.28
    assert asymmetric_clip_ratio(1.0, 0.2, 0.28) == 1.0


def test_resolve_clip_defaults():
    assert resolve_clip_epsilons(0.1) == (0.1, 0.1)
    assert resolve_clip_epsilons(0.1, 0.2, 0.28) == (0.2, 0.28)
    assert resolve_clip_epsilons(0.1, None, 0.28) == (0.1, 0.28)


def test_rewards_are_flat():
    assert rewards_are_flat([0.0, 0.0, 0.0])
    assert rewards_are_flat([1.0, 1.0])
    assert not rewards_are_flat([0.0, 1.0])
    assert rewards_are_flat([])


def test_compute_advantages():
    adv = compute_advantages([0.0, 1.0], baseline=0.5)
    assert len(adv) == 2
    assert adv[0] < 0 < adv[1]


def test_openai_completions_path():
    backend = OpenAICompatRolloutBackend("http://example.test/v1", model="m")

    def fake_urlopen(req, timeout=0):
        class Resp:
            def read(self):
                return json.dumps({
                    "choices": [{"text": "a"}, {"text": "b"}],
                }).encode()

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

        return Resp()

    with patch("astrarl.rollout.urllib.request.urlopen", fake_urlopen):
        out = backend.generate("prompt", n=2, temperature=1.0, max_tokens=16)
    assert out == ["a", "b"]


def test_factory_requires_url():
    try:
        make_rollout_backend("openai", base_url="")
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_collect_group_dynamic_resample_then_skip():
    rollout = _StubRollout([["0", "0"], ["0", "0"], ["0", "0"]])
    group, ema = collect_group(
        rollout=rollout,
        plugin=_DummyPlugin(),
        case={"id": "c1"},
        k=2,
        temperature=1.0,
        max_tokens=8,
        reward_ema=0.0,
        dynamic_sample_retries=2,
    )
    assert group.skipped
    assert group.skip_reason == "flat_rewards"
    assert group.dynamic_resamples == 2
    assert rollout.calls == 3  # initial + 2 retries


def test_collect_group_resamples_until_variance():
    rollout = _StubRollout([["1", "1"], ["0", "1"]])
    group, ema = collect_group(
        rollout=rollout,
        plugin=_DummyPlugin(),
        case={"id": "c1"},
        k=2,
        temperature=1.0,
        max_tokens=8,
        reward_ema=0.0,
        dynamic_sample_retries=3,
    )
    assert not group.skipped
    assert group.dynamic_resamples == 1
    assert len(group.advantages) == 2
