"""
Ollama inference provider.

Calls an Ollama HTTP server (local or remote) via its REST API.
Suitable for offloading the planning/reasoning model to a separate machine
(e.g. mac-mini.local) while the local MacBook runs MLX for code generation.

Structured output: uses Ollama's native `format: "json"` flag when a
json_schema is requested, then extracts the first JSON object from the reply.
"""
from __future__ import annotations

import json
import re
from typing import Optional

import httpx

from backend.agent.inference.base import InferenceProvider, Message, GenerationConfig
from backend.logging_config import get_logger

logger = get_logger(__name__)

_DEFAULT_TIMEOUT = 300  # seconds — planning calls can be slow on first token


class OllamaProvider(InferenceProvider):
    def __init__(
        self,
        model_id: str = "llama3.1",
        base_url: str = "http://localhost:11434",
    ) -> None:
        self._model_id = model_id
        self._base_url = base_url.rstrip("/")
        self._available: Optional[bool] = None  # lazily checked

    # ── InferenceProvider interface ───────────────────────────────────────────

    @property
    def model_id(self) -> str:
        return self._model_id

    def is_loaded(self) -> bool:
        # Ollama manages model loading server-side; treat as always ready
        return True

    async def unload(self) -> None:
        # Ollama manages its own memory; nothing to do locally
        pass

    async def generate(self, messages: list[Message], config: Optional[GenerationConfig] = None) -> str:
        cfg = config or GenerationConfig()

        payload: dict = {
            "model": self._model_id,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "stream": False,
            "options": {
                "temperature": cfg.temperature,
                "top_p": cfg.top_p,
                "num_predict": cfg.max_tokens,
            },
        }

        if cfg.json_schema:
            payload["format"] = "json"
            # Inject schema hint into the last user message
            schema_hint = (
                f"\nRespond ONLY with valid JSON matching this schema:\n"
                f"{json.dumps(cfg.json_schema, indent=2)}"
            )
            if payload["messages"] and payload["messages"][-1]["role"] == "user":
                payload["messages"][-1]["content"] += schema_hint
            else:
                payload["messages"].append({"role": "user", "content": schema_hint})

        url = f"{self._base_url}/api/chat"
        logger.debug("OllamaProvider: POST %s model=%s", url, self._model_id)

        async with httpx.AsyncClient(timeout=_DEFAULT_TIMEOUT) as client:
            response = await client.post(url, json=payload)
            response.raise_for_status()

        data = response.json()
        text = data.get("message", {}).get("content", "")

        if cfg.json_schema:
            return self._extract_json(text)
        return text

    @staticmethod
    def _extract_json(text: str) -> str:
        clean = re.sub(r"```(?:json)?\s*(.*?)\s*```", r"\1", text, flags=re.DOTALL).strip()
        match = re.search(r"\{.*\}", clean, re.DOTALL)
        return match.group(0) if match else clean
