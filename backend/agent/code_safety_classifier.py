"""
CodeSafetyClassifier — Phase 9.

LLM-based safety classifier for training scripts awaiting EXECUTE_CODE approval.
Returns a verdict (safe/unsafe) with a brief rationale so the HUD can
auto-approve low-risk scripts without blocking on human review.
"""
from __future__ import annotations

import ast
import io
import re
import tokenize
from dataclasses import dataclass

from backend.agent.inference.base import InferenceProvider, Message, GenerationConfig
from backend.logging_config import get_logger

logger = get_logger(__name__)

_SYSTEM = """\
You are ASTRA's Code Safety Classifier.
You are given a Python training script about to be executed in a sandboxed process.
Classify whether it is safe to auto-approve execution without explicit human review.

A script is SAFE if it:
- Only imports standard ML libraries (gymnasium, stable-baselines3, sklearn, torch, numpy, requests, logging, etc.)
- Only POSTs to localhost telemetry endpoints (127.0.0.1 / localhost)
- Only writes files inside the data/missions/ directory tree
- Has no shell execution (subprocess, os.system, eval, exec)
- Has no network calls to external hosts
- Has no file deletions or moves

A script is UNSAFE if it:
- Calls subprocess, os.system, os.popen, eval(), exec(), or __import__()
- Opens network connections to non-localhost hosts
- Reads, writes, or deletes FILES outside the project directory (os.remove, shutil.rmtree, etc.)
- Contains obfuscated code or base64-decoded execution

Important clarifications — these are ALL SAFE, do NOT flag them:
- `del variable` is Python object deletion (freeing memory), NOT a file operation
- `requests.post(...)` or `requests.post(VAR, ...)` where VAR resolves to 127.0.0.1/localhost is SAFE telemetry
- Importing standard libraries (os, sys, json, logging, numpy, etc.) is SAFE
- `sys.path.insert(0, "/any/absolute/path")` adds a directory to Python's import search path — it does NOT read files and is SAFE
- `_sys.path.insert(...)` is identical to `sys.path.insert(...)` — SAFE
- Writing or reading files under the project directory (e.g. /Users/.../astra/data/missions/...) is SAFE
- Absolute paths in strings are NOT inherently unsafe — only flag actual reads/writes OUTSIDE the project tree
- `os.chdir(path)` only changes the current working directory for subsequent relative-path
  operations in THIS process — it does not read, write, move, or delete anything by itself.
  It is SAFE, including when `path` is an absolute path outside the project directory (e.g.
  changing into a sibling project's directory to run that project's own script with its own
  relative paths, such as `os.chdir("/Users/.../finetune")` before `os.execv(...)`)
- `os.execv(interpreter, argv)` replaces the current process image to run another Python
  script — it is SAFE when `argv` is a fixed, literal list (not built from string
  concatenation or eval'd input) pointing at a specific `.py` file, since it is equivalent to
  running that script directly, not arbitrary/dynamic code execution
- Some scripts are fine-tuning training scripts for a SEPARATE, related project (e.g. paths
  containing "finetune", "ensemble"), not this one — reading/writing model checkpoints,
  LoRA adapters, and logs UNDER THAT PROJECT's own directory (e.g. ~/finetune/adapters/...,
  ~/finetune/logs/...) is the intended, expected purpose of such a script and is SAFE, even
  though the path is outside THIS project's own directory. "Outside the project directory"
  only means somewhere unrelated to the task at hand (e.g. system files, another user's home
  directory) — not simply "not under astra's own repo"

Respond with JSON only:
{"safe": true, "reason": "one-sentence rationale"}
or
{"safe": false, "reason": "one-sentence explanation of the risk"}"""


@dataclass
class SafetyVerdict:
    safe: bool
    reason: str
    classifier: str = "llm"


class CodeSafetyClassifier:
    def __init__(self, provider: InferenceProvider) -> None:
        self._provider = provider

    async def classify(self, script: str) -> SafetyVerdict:
        """Classify a training script and return a SafetyVerdict."""
        # Fast-path static checks before calling LLM.
        # classifier="static" means a definitive verdict (safe or unsafe) — skip LLM.
        # classifier="static_ambiguous" means no danger found but not confirmed safe — proceed to LLM.
        static = self._static_check(script)
        if static.classifier != "static_ambiguous":
            return static

        # Strip a leading module docstring before truncating to 2000 chars —
        # a "Usage: nohup python -u dpo_train.py ..." example in a docstring
        # is documentation, not code, but a naive [:2000] slice on a script
        # with a long header docstring sends the LLM nothing BUT that
        # documentation, and it can misread example CLI invocations as real
        # dangerous actions (confirmed via a live incident on dpo_train.py).
        llm_view = self._strip_leading_docstring(script)
        messages = [
            Message(role="system", content=_SYSTEM),
            Message(role="user", content=f"Classify this script:\n\n{llm_view[:2000]}"),
        ]
        try:
            # 64 tokens was too tight for {"safe": ..., "reason": "..."} — a
            # verdict with a longer reason gets cut off mid-string, producing
            # invalid JSON ("Unterminated string...") and forcing every such
            # script to manual review even when the LLM's actual verdict
            # would have been fine (confirmed via a live incident).
            raw = await self._provider.generate(
                messages, GenerationConfig(max_tokens=128, temperature=0.0)
            )
            import json
            raw = re.sub(r"```(?:json)?\s*|```", "", raw).strip()
            # Extract first JSON object in case the LLM appends extra text
            m = re.search(r"\{[^}]*\}", raw, re.DOTALL)
            raw = m.group(0) if m else raw
            data = json.loads(raw)
            verdict = SafetyVerdict(
                safe=bool(data.get("safe", False)),
                reason=str(data.get("reason", "no reason given")),
            )
            logger.info("CodeSafetyClassifier: safe=%s reason=%s", verdict.safe, verdict.reason)
            return verdict
        except Exception as exc:
            logger.warning("CodeSafetyClassifier: LLM failed: %s", exc)
            return SafetyVerdict(safe=False, reason="LLM classifier unavailable — manual review required", classifier="static")

    @staticmethod
    def _strip_strings_and_comments(script: str) -> str:
        """Return `script` with all string-literal and comment content
        removed, using the real Python tokenizer — so danger-pattern regexes
        only ever match actual code, not English words that happen to collide
        with a dangerous-looking token inside a print/log message or comment
        (e.g. "--- Baseline eval (step 0) ---"). Falls back to the raw script
        if tokenizing fails (e.g. a truncated/invalid fragment) — regex
        matching then just has the same false-positive risk it always had,
        not a crash."""
        try:
            out: list[str] = []
            prev_end_row, prev_end_col = 1, 0
            for tok_type, tok_string, (start_row, start_col), (end_row, end_col), _ in \
                    tokenize.generate_tokens(io.StringIO(script).readline):
                if tok_type in (tokenize.STRING, tokenize.COMMENT):
                    prev_end_row, prev_end_col = end_row, end_col
                    continue
                # Reconstruct real spacing/adjacency from the token's actual
                # source position (not an arbitrary separator) — this is what
                # keeps e.g. "model.eval(" adjacent so the negative-lookbehind
                # exclusion for method calls still works on the output.
                if start_row != prev_end_row:
                    out.append("\n" * (start_row - prev_end_row))
                    prev_end_col = 0
                if start_col > prev_end_col:
                    out.append(" " * (start_col - prev_end_col))
                out.append(tok_string)
                prev_end_row, prev_end_col = end_row, end_col
            return "".join(out)
        except Exception:
            return script

    @staticmethod
    def _strip_leading_docstring(script: str) -> str:
        """Remove a module-level docstring from the start of `script`, so a
        long "Usage: nohup python -u foo.py --bar ..." example in a header
        docstring doesn't dominate (or entirely fill) the truncated slice
        sent to the LLM — it's documentation about how to invoke the script
        from a shell, not code that runs, but an LLM shown only that slice
        can mistake it for real dangerous actions (confirmed via a live
        incident on ensemble/finetune/dpo_train.py). Uses `ast` (not the
        tokenizer) since only the AST distinguishes "first statement is a
        bare string expression" (a docstring) from any other string literal.
        Falls back to the raw script if parsing fails."""
        try:
            tree = ast.parse(script)
            if (
                tree.body
                and isinstance(tree.body[0], ast.Expr)
                and isinstance(tree.body[0].value, ast.Constant)
                and isinstance(tree.body[0].value.value, str)
            ):
                lines = script.splitlines(keepends=True)
                return "".join(lines[tree.body[0].end_lineno:])
            return script
        except Exception:
            return script

    @staticmethod
    def _static_check(script: str) -> SafetyVerdict:
        """Deterministic pre-filter: fail fast on danger patterns, short-circuit safe on known-good scripts."""
        # Danger patterns must only match real code, not English text that
        # happens to contain the same words inside a print/log string (e.g.
        # "--- Baseline eval (step 0) ---") — strip string/comment content
        # before matching, using the real tokenizer rather than more regex
        # exceptions layered on top of regex exceptions.
        code_only = CodeSafetyClassifier._strip_strings_and_comments(script)

        danger_patterns = [
            (r"\bsubprocess\b", "uses subprocess"),
            (r"\bos\.system\b", "uses os.system"),
            (r"\bos\.popen\b", "uses os.popen"),
            # Negative lookbehind excludes method calls like model.eval() /
            # mx.eval(...) — standard PyTorch/MLX idioms (switch to eval mode;
            # force lazy evaluation), not the dangerous builtin eval(...).
            (r"(?<!\.)\beval\s*\(", "uses eval()"),
            (r"(?<!\.)\bexec\s*\(", "uses exec()"),
            (r'__import__\s*\(', "uses __import__()"),
        ]
        for pattern, reason in danger_patterns:
            if re.search(pattern, code_only):
                return SafetyVerdict(safe=False, reason=f"Static check failed: {reason}", classifier="static")

        # External-HTTP check still needs the actual URL strings, so it runs
        # against the original script (URLs necessarily live in string literals).
        if re.search(
            r'requests\.(get|post|put|delete|patch)\s*\(\s*["\']https?://(?!127\.0\.0\.1|localhost)',
            script,
        ):
            return SafetyVerdict(safe=False, reason="Static check failed: makes external HTTP request", classifier="static")

        # Positive short-circuit: if every requests call in the script targets localhost,
        # this is a standard ASTRA training script — skip the LLM classifier entirely.
        # Handle both literal URLs and variable-based URLs (e.g. TELEMETRY_URL = "http://127.0.0.1:...").
        localhost_url_vars: set[str] = set()
        external_url_vars: set[str] = set()
        for var, url in re.findall(r'(\w+)\s*=\s*["\']https?://([^"\']+)["\']', script):
            if url.startswith(("127.0.0.1", "localhost")):
                localhost_url_vars.add(var)
            else:
                external_url_vars.add(var)

        literal_requests = re.findall(
            r'requests\.\w+\s*\(\s*["\']([^"\']+)["\']', script
        )
        var_requests = re.findall(
            r'requests\.\w+\s*\(\s*([A-Z_][A-Z0-9_]*)\b', script
        )

        # Fail fast if a request variable resolves to a known external URL
        for var in var_requests:
            if var in external_url_vars:
                return SafetyVerdict(
                    safe=False,
                    reason=f"Static check failed: request variable {var} resolves to non-localhost URL",
                    classifier="static",
                )

        has_any_request = bool(literal_requests or var_requests)
        literals_safe = all("127.0.0.1" in u or "localhost" in u for u in literal_requests)
        vars_safe = all(v in localhost_url_vars for v in var_requests)

        if has_any_request and literals_safe and vars_safe:
            return SafetyVerdict(
                safe=True,
                reason="all requests target localhost telemetry — auto-approved",
                classifier="static",
            )

        # Static auto-approve for the dpo/grpo dispatch wrapper: no requests
        # calls (already the strictest "no network access" case established
        # above) and an os.execv(...) targeting dpo_train.py/grpo_train.py by
        # name — this is exactly and only the shape the DPO/GRPO code-gen
        # templates produce (code_generator.py's _DPO_TEMPLATE/_GRPO_TEMPLATE).
        # Matched loosely (os.execv anywhere + the script name anywhere) since
        # code-gen doesn't always produce identical call-site formatting (e.g.
        # args built as a separate list variable vs an inline list literal) —
        # a real incident showed a syntactically-different-but-equally-safe
        # variant fall through to the LLM classifier, which (a 4th time)
        # incorrectly flagged legitimate os.execv process replacement as
        # "could lead to arbitrary code execution," stalling a live mission.
        if (
            not has_any_request
            and re.search(r"\bos\.execv\s*\(", code_only)
            and re.search(r"dpo_train\.py|grpo_train\.py", script)
        ):
            return SafetyVerdict(
                safe=True,
                reason="os.execv dispatch to dpo_train.py/grpo_train.py with no network calls — auto-approved",
                classifier="static",
            )

        return SafetyVerdict(safe=True, reason="passed static checks", classifier="static_ambiguous")
