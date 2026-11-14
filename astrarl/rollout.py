"""Pluggable rollout backends for online LM RL (MLX local | OpenAI-compatible HTTP).

Platform-owned: Ensemble (and other use cases) import from ``astrarl.rollout``.
"""
from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any, Optional, Protocol, runtime_checkable


@runtime_checkable
class RolloutBackend(Protocol):
    def generate(
        self,
        prompt: str,
        *,
        n: int,
        temperature: float,
        max_tokens: int,
    ) -> list[str]:
        ...


class MLXRolloutBackend:
    """Wraps ``mlx_lm.generate`` (default GRPO path)."""

    def __init__(self, model: Any, tokenizer: Any) -> None:
        self.model = model
        self.tokenizer = tokenizer

    def generate(
        self,
        prompt: str,
        *,
        n: int,
        temperature: float,
        max_tokens: int,
    ) -> list[str]:
        import mlx.core as mx
        from mlx_lm.generate import generate as mlx_generate
        from mlx_lm.sample_utils import make_sampler

        sampler = make_sampler(temp=temperature)
        completions: list[str] = []
        for _ in range(n):
            try:
                text = mlx_generate(
                    self.model,
                    self.tokenizer,
                    prompt=prompt,
                    max_tokens=max_tokens,
                    sampler=sampler,
                    verbose=False,
                )
                completions.append(text.strip())
            except (IndexError, KeyError):
                completions.append("")
            mx.clear_cache()
        return completions


class OpenAICompatRolloutBackend:
    """HTTP OpenAI-compatible completions (``/v1/completions`` then chat fallback)."""

    def __init__(
        self,
        base_url: str,
        *,
        api_key: str = "",
        model: str = "default",
        timeout_s: float = 120.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key or "EMPTY"
        self.model = model
        self.timeout_s = timeout_s

    def _headers(self) -> dict[str, str]:
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

    def _post(self, path: str, body: dict[str, Any]) -> dict[str, Any]:
        url = f"{self.base_url}{path}"
        data = json.dumps(body).encode("utf-8")
        req = urllib.request.Request(
            url, data=data, headers=self._headers(), method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def generate(
        self,
        prompt: str,
        *,
        n: int,
        temperature: float,
        max_tokens: int,
    ) -> list[str]:
        try:
            out = self._post(
                "/v1/completions",
                {
                    "model": self.model,
                    "prompt": prompt,
                    "n": max(1, int(n)),
                    "temperature": float(temperature),
                    "max_tokens": int(max_tokens),
                },
            )
            choices = out.get("choices") or []
            texts = [(c.get("text") or "").strip() for c in choices]
            if texts:
                while len(texts) < n:
                    more = self._post(
                        "/v1/completions",
                        {
                            "model": self.model,
                            "prompt": prompt,
                            "n": 1,
                            "temperature": float(temperature),
                            "max_tokens": int(max_tokens),
                        },
                    )
                    texts.append(((more.get("choices") or [{}])[0].get("text") or "").strip())
                return texts[:n]
        except (urllib.error.HTTPError, urllib.error.URLError, KeyError, json.JSONDecodeError):
            pass

        texts = []
        for _ in range(n):
            try:
                out = self._post(
                    "/v1/chat/completions",
                    {
                        "model": self.model,
                        "messages": [{"role": "user", "content": prompt}],
                        "temperature": float(temperature),
                        "max_tokens": int(max_tokens),
                        "n": 1,
                    },
                )
                msg = ((out.get("choices") or [{}])[0].get("message") or {})
                texts.append((msg.get("content") or "").strip())
            except (urllib.error.HTTPError, urllib.error.URLError, KeyError, json.JSONDecodeError):
                texts.append("")
        return texts


def make_rollout_backend(
    kind: str,
    *,
    model: Any = None,
    tokenizer: Any = None,
    base_url: str = "",
    api_key: str = "",
    openai_model: str = "default",
) -> RolloutBackend:
    k = (kind or "mlx").lower().strip()
    if k in ("mlx", "local"):
        if model is None or tokenizer is None:
            raise ValueError("MLXRolloutBackend requires model and tokenizer")
        return MLXRolloutBackend(model, tokenizer)
    if k in ("openai", "openai_compat", "http", "vllm"):
        if not base_url:
            raise ValueError("OpenAICompatRolloutBackend requires base_url")
        return OpenAICompatRolloutBackend(
            base_url, api_key=api_key, model=openai_model,
        )
    raise ValueError(f"unknown rollout backend: {kind!r}")


# ── DAPO-lite helpers (pure Python; tested without MLX) ───────────────────────

def asymmetric_clip_ratio(ratio: float, eps_low: float, eps_high: float) -> float:
    """Clip importance ratio into [1-eps_low, 1+eps_high] (DAPO clip-higher)."""
    lo = 1.0 - float(eps_low)
    hi = 1.0 + float(eps_high)
    if ratio < lo:
        return lo
    if ratio > hi:
        return hi
    return ratio


def rewards_are_flat(rewards: list[float], atol: float = 1e-9) -> bool:
    """True when a group has no relative signal (all rewards equal)."""
    if not rewards:
        return True
    return max(rewards) - min(rewards) <= atol


def resolve_clip_epsilons(
    clip_epsilon: float,
    clip_epsilon_low: Optional[float] = None,
    clip_epsilon_high: Optional[float] = None,
) -> tuple[float, float]:
    """Default both sides to clip_epsilon when asymmetric flags are unset."""
    low = float(clip_epsilon if clip_epsilon_low is None else clip_epsilon_low)
    high = float(clip_epsilon if clip_epsilon_high is None else clip_epsilon_high)
    return low, high
