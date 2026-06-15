"""
ErrorAnalyzer — Step 3.2 (Self-Healer).

Parses sandbox error output (stack traces, runtime exceptions) and uses the
Lead Agent's LLM to generate a corrected version of the training script.
"""
from __future__ import annotations

import os
import re

from backend.agent.inference.base import InferenceProvider, Message, GenerationConfig
from backend.logging_config import get_logger

logger = get_logger(__name__)

_SYSTEM_PROMPT = """\
You are ASTRA's Self-Healer — an expert Python debugger.
You are given a training script that failed and its error output.
Analyze the error and return the complete corrected Python script.
Rules:
- Fix ONLY what caused the error; do not restructure working code.
- If the error is an ImportError, add the missing import.
- If the error is a shape mismatch or type error, fix the data handling.
- If the error is a hyperparameter issue, adjust to a safe default.
- DO NOT use markdown code blocks (```python ... ```).
- Return ONLY the raw corrected Python script, no explanation, no preamble, no stop tokens."""

_MAX_TRACEBACK_LINES = 50   # truncate very long tracebacks


def _extract_traceback(error_output: str) -> str:
    """Pull the last N lines of the traceback for context efficiency."""
    lines = error_output.strip().splitlines()
    return "\n".join(lines[-_MAX_TRACEBACK_LINES:])


def _extract_error_type(error_output: str) -> str:
    """Return the exception class name from the traceback."""
    match = re.search(r"^(\w+(?:Error|Exception|Warning))", error_output, re.MULTILINE)
    return match.group(1) if match else "UnknownError"


class ErrorAnalyzer:
    def __init__(self, provider: InferenceProvider) -> None:
        self._provider = provider

    async def fix_script(
        self,
        script_path: str,
        error_output: str,
        iteration: int = 0,
    ) -> str:
        """
        Generate a fixed version of the failing script.
        Writes the fix to {script_path}.fixed_{iteration}.py and returns the path.
        """
        try:
            with open(script_path, "r") as f:
                original_code = f.read()
        except FileNotFoundError:
            logger.error("ErrorAnalyzer: script not found: %s", script_path)
            raise

        traceback = _extract_traceback(error_output)
        error_type = _extract_error_type(error_output)

        logger.warning("ErrorAnalyzer: fixing %s after %s (iteration %d)", script_path, error_type, iteration)

        user_prompt = (
            f"The following Python training script raised a {error_type}:\n\n"
            f"=== SCRIPT ===\n{original_code}\n\n"
            f"=== ERROR ===\n{traceback}\n\n"
            "Return the complete corrected script."
        )

        messages = [
            Message(role="system", content=_SYSTEM_PROMPT),
            Message(role="user", content=user_prompt),
        ]

        fixed_code = await self._provider.generate(
            messages, GenerationConfig(max_tokens=4096, temperature=0.05)
        )
        fixed_code = self._strip_fences(fixed_code)

        fixed_path = f"{script_path}.fixed_{iteration}.py"
        with open(fixed_path, "w") as f:
            f.write(fixed_code)

        logger.info("ErrorAnalyzer: fix written to %s", fixed_path)
        return fixed_path

    @staticmethod
    def _strip_fences(text: str) -> str:
        import re
        # Remove common LLM artifacts
        text = re.sub(r"<\|im_end\|>|<\|endoftext\|>", "", text)
        # Remove lines that are purely markdown fences or artifacts
        lines = []
        for line in text.splitlines():
            clean_line = line.strip()
            if clean_line.startswith("```"):
                continue
            lines.append(line)
        return "\n".join(lines).strip()
