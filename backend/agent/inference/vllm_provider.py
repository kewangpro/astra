"""
vLLM (Metal) inference provider — optional, for 64GB+ unified memory environments.

Provides PagedAttention for large log contexts and continuous batching for
simultaneous specialist reasoning. Not recommended on 24GB (pre-allocation
overhead leaves insufficient memory for training sandboxes).

Install: pip install vllm  (requires vLLM with Metal backend)
"""
from __future__ import annotations

from typing import Optional

from backend.agent.inference.base import InferenceProvider, Message, GenerationConfig
from backend.logging_config import get_logger

logger = get_logger(__name__)

_VLLM_AVAILABLE = False
try:
    from vllm import LLM, SamplingParams
    _VLLM_AVAILABLE = True
except ImportError:
    pass


class VLLMProvider(InferenceProvider):
    def __init__(
        self,
        model_id: str = "mistralai/Mistral-Nemo-Instruct-2407",
        tensor_parallel_size: int = 1,
    ) -> None:
        self._model_id = model_id
        self._tensor_parallel_size = tensor_parallel_size
        self._llm = None

        if not _VLLM_AVAILABLE:
            raise RuntimeError(
                "vllm is not installed. Install with: pip install vllm\n"
                "Note: vLLM Metal backend requires 64GB+ unified memory."
            )

    def load(self) -> None:
        if self._llm is not None:
            return
        logger.info("Loading vLLM model: %s", self._model_id)
        self._llm = LLM(
            model=self._model_id,
            tensor_parallel_size=self._tensor_parallel_size,
        )
        logger.info("vLLM model loaded: %s", self._model_id)

    async def unload(self) -> None:
        self._llm = None
        logger.info("vLLM model unloaded: %s", self._model_id)

    def is_loaded(self) -> bool:
        return self._llm is not None

    @property
    def model_id(self) -> str:
        return self._model_id

    async def generate(self, messages: list[Message], config: Optional[GenerationConfig] = None) -> str:
        if not self.is_loaded():
            self.load()

        cfg = config or GenerationConfig()
        prompt = "\n".join(f"{m.role}: {m.content}" for m in messages)

        params = SamplingParams(
            temperature=cfg.temperature,
            top_p=cfg.top_p,
            max_tokens=cfg.max_tokens,
        )
        outputs = self._llm.generate([prompt], params)
        return outputs[0].outputs[0].text
