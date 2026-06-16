"""
ErrorAnalyzer — Step 3.2 (Self-Healer).

Parses sandbox error output (stack traces, runtime exceptions) and uses the
Lead Agent's LLM to generate a corrected version of the training script.
"""
from __future__ import annotations

import os
import re
from typing import Optional

from backend.agent.inference.base import InferenceProvider, Message, GenerationConfig
from backend.logging_config import get_logger

logger = get_logger(__name__)

_SYSTEM_PROMPT = """\
You are ASTRA's Self-Healer — an expert Python debugger.
You are given a training script that failed and its error output.
Analyze the error and return the complete corrected Python script.
Rules:
- Scan the ENTIRE script for ALL instances of this error class and fix them all in one pass — do not fix just the one line that appeared in the traceback.
- If the error is an ImportError, ModuleNotFoundError, or NameError, add ALL missing imports at the top and check the full script for any other missing names at the same time. NameError almost always means a missing import.
- For RL scripts using stable-baselines3, the script MUST include ALL of these imports — add any that are missing:
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback
  Never omit these. If PPO or BaseCallback appear in the script body, they MUST be imported.
- If the error is a TypeError about unexpected keyword arguments, remove ALL invalid kwargs from every model constructor call in the script, not just the first one.
- Valid SB3 PPO kwargs: learning_rate, n_steps, batch_size, n_epochs, gamma, gae_lambda, clip_range, clip_range_vf, ent_coef, vf_coef, max_grad_norm, target_kl. Remove any others.
- For custom callback classes that extend BaseCallback: the __init__ MUST be `def __init__(self, verbose=0, **kwargs): super().__init__(verbose=verbose)`. This accepts extra kwargs safely. Do NOT pass checkpoint_freq, save_path, or check_freq to the callback constructor call — call it as `CustomCallback()` with no arguments.
- If the error is a shape mismatch or type error, fix all data handling of that type.
- If the error is a hyperparameter issue, replace ALL occurrences with safe defaults.
- Do not restructure working code.
- DO NOT use markdown code blocks (```python ... ```).
- Return ONLY the raw corrected Python script, no explanation, no preamble, no stop tokens."""

_MAX_TRACEBACK_LINES = 50   # truncate very long tracebacks


def _patch_callback_init(code: str) -> str:
    """Ensure BaseCallback subclasses have a **kwargs-accepting __init__."""
    import re
    # Replace bare `class Foo(BaseCallback):` bodies that lack __init__
    # by inserting __init__ after the class line if missing
    if "BaseCallback" not in code:
        return code
    # Fix callback constructor calls: CustomCallback(checkpoint_freq=...) -> CustomCallback()
    code = re.sub(
        r'(\w*[Cc]allback\w*)\s*\(\s*(?:check_freq|checkpoint_freq|save_path)[^)]*\)',
        r'\1()',
        code,
    )
    return code


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
        prior_errors: Optional[list[str]] = None,
        mission_id: Optional[str] = None,
        domain: Optional[str] = None,
    ) -> str:
        """
        Generate a fixed version of the failing script.
        Writes the fix to {script_path}.fixed_{iteration}.py and returns the path.

        prior_errors: errors from previous healing attempts, so the LLM knows
                      what was already tried and can fix all issues in one pass.
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

        prior_context = ""
        if prior_errors:
            summaries = "\n".join(
                f"Attempt {i+1}: {_extract_error_type(e)} — {_extract_traceback(e).splitlines()[-1]}"
                for i, e in enumerate(prior_errors)
            )
            prior_context = (
                f"\nPrevious fix attempts also failed with these errors "
                f"(fix all of them in this pass):\n{summaries}\n"
            )

        user_prompt = (
            f"The following Python training script raised a {error_type}:\n\n"
            f"=== SCRIPT ===\n{original_code}\n\n"
            f"=== CURRENT ERROR ===\n{traceback}\n"
            f"{prior_context}\n"
            "Scan the entire script and fix ALL instances of these error classes. "
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
        fixed_code = self._patch_missing_imports(fixed_code)

        fixed_path = f"{script_path}.fixed_{iteration}.py"
        with open(fixed_path, "w") as f:
            f.write(fixed_code)
        # Also overwrite the canonical train.py so the next sandbox run uses the fix
        with open(script_path, "w") as f:
            f.write(fixed_code)

        logger.info("ErrorAnalyzer: fix written to %s (and %s)", fixed_path, script_path)

        # Store lesson in vector memory so future code generation avoids this error
        self._store_lesson(error_type, traceback, mission_id, domain)

        return fixed_path

    @staticmethod
    def _patch_missing_imports(code: str) -> str:
        """Deterministically inject imports that the LLM forgot but the script uses."""
        lines = code.splitlines()
        # Find the last existing import line to insert after it
        last_import_idx = 0
        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped.startswith("import ") or stripped.startswith("from "):
                last_import_idx = i

        to_inject = []
        if "PPO" in code and "from stable_baselines3 import PPO" not in code:
            to_inject.append("from stable_baselines3 import PPO")
        if "BaseCallback" in code and "from stable_baselines3.common.callbacks import BaseCallback" not in code:
            to_inject.append("from stable_baselines3.common.callbacks import BaseCallback")
        if "SAC" in code and "from stable_baselines3 import SAC" not in code:
            to_inject.append("from stable_baselines3 import SAC")
        if "A2C" in code and "from stable_baselines3 import A2C" not in code:
            to_inject.append("from stable_baselines3 import A2C")
        if "DQN" in code and "from stable_baselines3 import DQN" not in code:
            to_inject.append("from stable_baselines3 import DQN")

        if to_inject:
            insert_at = last_import_idx + 1
            for i, imp in enumerate(to_inject):
                lines.insert(insert_at + i, imp)
            logger.info("ErrorAnalyzer: patched missing imports: %s", to_inject)
            code = "\n".join(lines)
        # Fix callback class that has _on_step but doesn't extend BaseCallback
        import re
        code = re.sub(r'(class\s+\w*[Cc]allback\w*)\s*:', r'\1(BaseCallback):', code)
        if "def __init__(self):" in code and "(BaseCallback)" in code:
            code = code.replace(
                "def __init__(self):",
                "def __init__(self, verbose=0, **kwargs):\n        super().__init__(verbose=verbose)",
            )
        return _patch_callback_init(code)

    def _store_lesson(
        self,
        error_type: str,
        traceback: str,
        mission_id: Optional[str],
        domain: Optional[str],
    ) -> None:
        try:
            import uuid
            from backend.services import vector_memory
            last_line = traceback.strip().splitlines()[-1] if traceback else ""
            lesson = (
                f"Code generation error ({error_type}): {last_line}. "
                f"Fix: scan entire script for all instances of this error class and remove/correct them all."
            )
            vector_memory.add_lesson(
                f"codegen-{error_type}-{uuid.uuid4().hex[:8]}",
                lesson,
                run_id=mission_id or "unknown",
                domain=domain or "code_generation",
                extra={"error_type": error_type, "source": "error_analyzer"},
            )
        except Exception as exc:
            logger.debug("ErrorAnalyzer: could not store lesson: %s", exc)

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
