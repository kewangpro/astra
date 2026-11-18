"""One-shot LoRA completion for non-game / routing Model Registry entries.

Runs ``generate_once.py`` on the Mac Mini over SSH (same host/cwd as
``bare_eval``). Resolves adapter + base model + prompt template from the
``ModelRecord`` and linked mission plan.
"""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import time
from dataclasses import dataclass
from typing import Any, Mapping, Optional

from backend.config import settings
from backend.logging_config import get_logger

logger = get_logger(__name__)

DEFAULT_FINETUNE_DIR = "~/finetune"
DEFAULT_PYTHON_BIN = "~/finetune-env/bin/python"
DEFAULT_BASE_MODEL = "mlx-community/gemma-3-12b-it-4bit"
DEFAULT_PROMPT_TEMPLATE = "backend/prompts/conductor_min.md"
INFER_TIMEOUT_S = 120


class LoraInferError(Exception):
    """Raised when inference cannot be prepared or Mini returns an error."""

    def __init__(self, message: str, *, status_code: int = 400):
        super().__init__(message)
        self.status_code = status_code


@dataclass(frozen=True)
class InferConfig:
    finetune_dir: str
    python_bin: str
    adapter_rel: str
    base_model: str
    prompt_template: str
    wrap_routing: bool


@dataclass(frozen=True)
class InferResult:
    completion: str
    prompt_used: str
    model_id: str
    latency_ms: int
    adapter: str
    base_model: str


def _local_ensemble_finetune_dir() -> Optional[str]:
    try:
        from backend.services.recipe_paths import parse_product_recipes

        packs = parse_product_recipes()
        recipes_dir = packs.get("ensemble") or ""
        if not recipes_dir:
            return None
        root = os.path.dirname(recipes_dir.rstrip(os.sep))
        candidate = os.path.join(root, "finetune")
        return candidate if os.path.isdir(candidate) else None
    except Exception:
        return None


def push_generate_once(finetune_dir: str) -> None:
    """Best-effort scp of generate_once.py (+ deps) onto Mini."""
    local = _local_ensemble_finetune_dir()
    host = (settings.sandbox_host or "").strip()
    if not local or not host or not finetune_dir:
        return
    names = (
        "generate_once.py",
        "_repo_paths.py",
        "_platform_preflight.py",
        "prepare_finetune_data.py",
        "routing_oracle.py",
    )
    for name in names:
        src = os.path.join(local, name)
        if not os.path.isfile(src):
            continue
        for remote_rel in (name, os.path.join("finetune", name)):
            remote = f"{host}:{finetune_dir.rstrip('/')}/{remote_rel}"
            try:
                subprocess.run(
                    ["scp", "-q", src, remote],
                    capture_output=True,
                    text=True,
                    timeout=60,
                    check=False,
                )
            except Exception as exc:
                logger.warning("lora_infer: scp %s → %s failed: %s", name, remote, exc)


def adapter_rel_from_path(path: Optional[str]) -> Optional[str]:
    """Extract Mini-relative ``adapters/…`` from a checkpoint path string."""
    if not path:
        return None
    norm = path.replace("\\", "/").strip()
    # Strip remote home / finetune prefixes
    for prefix in ("~/finetune/", "finetune/", "./"):
        if norm.startswith(prefix):
            norm = norm[len(prefix) :]
    if "adapters/" in norm:
        return norm[norm.index("adapters/") :]
    if norm.startswith("adapters/") or "/adapters/" in f"/{norm}":
        return norm.lstrip("/")
    return None


def resolve_adapter_rel(
    *,
    checkpoint_path: Optional[str],
    weights_path: Optional[str],
    extra_metadata: Optional[Mapping[str, Any]],
    mission_last_checkpoint: Optional[str],
) -> str:
    meta = extra_metadata if isinstance(extra_metadata, Mapping) else {}
    candidates = [
        meta.get("adapter") if isinstance(meta, Mapping) else None,
        meta.get("adapter_path") if isinstance(meta, Mapping) else None,
        meta.get("adapter_rel") if isinstance(meta, Mapping) else None,
        checkpoint_path,
        weights_path,
        mission_last_checkpoint,
    ]
    for c in candidates:
        if not c:
            continue
        rel = adapter_rel_from_path(str(c))
        if rel:
            return rel
    raise LoraInferError(
        "No Mini adapter path on this model (set checkpoint_path to adapters/…).",
        status_code=400,
    )


def resolve_adapter_or_bare(finetune_dir: str, adapter_rel: str) -> str:
    """Prefer ``…/best`` or ``…/final`` when those weight files exist on Mini."""
    host = (settings.sandbox_host or "").strip()
    if not host:
        return adapter_rel
    # If caller already pointed at best/final, trust it.
    base = adapter_rel.rstrip("/")
    if base.endswith("/best") or base.endswith("/final"):
        return adapter_rel
    check_cmd = (
        f"test -f {finetune_dir}/{base}/best/adapters.safetensors && echo yes || "
        f"(test -f {finetune_dir}/{base}/final/adapters.safetensors && echo final || echo no)"
    )
    try:
        result = subprocess.run(
            ["ssh", host, check_cmd],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        out = (result.stdout or "").strip()
        if out == "yes":
            return f"{base}/best"
        if out == "final":
            return f"{base}/final"
    except Exception as exc:
        logger.warning("lora_infer: adapter probe failed: %s", exc)
    return adapter_rel


def resolve_infer_config(
    *,
    checkpoint_path: Optional[str],
    weights_path: Optional[str],
    extra_metadata: Optional[Mapping[str, Any]],
    mission_plan: Optional[Mapping[str, Any]],
    mission_last_checkpoint: Optional[str],
    domain: str = "",
) -> InferConfig:
    meta = dict(extra_metadata or {})
    plan = dict(mission_plan or {})
    plan_hp = dict(plan.get("hyperparameters") or {})

    hp: dict[str, Any] = {}
    recipe_key = plan.get("recipe") or plan.get("task_type") or ""
    if recipe_key:
        try:
            from backend.agent.code_generator import _resolve_hyperparams

            hp = dict(_resolve_hyperparams(str(recipe_key), plan_hp))
        except Exception as exc:
            logger.warning("lora_infer: _resolve_hyperparams failed: %s", exc)

    finetune_dir = (
        str(meta.get("finetune_dir") or hp.get("finetune_dir") or DEFAULT_FINETUNE_DIR).strip()
    )
    python_bin = (
        str(meta.get("python_bin") or hp.get("python_bin") or DEFAULT_PYTHON_BIN).strip()
    )
    base_model = (
        str(
            meta.get("base_model")
            or meta.get("model")
            or hp.get("base_model")
            or hp.get("model")
            or plan.get("model")
            or DEFAULT_BASE_MODEL
        ).strip()
    )
    prompt_template = str(
        meta.get("prompt_template") or hp.get("prompt_template") or ""
    ).strip()
    domain_l = (domain or "").lower()
    wrap_routing = domain_l == "routing" or "routing" in domain_l
    if wrap_routing and not prompt_template:
        prompt_template = DEFAULT_PROMPT_TEMPLATE

    adapter_rel = resolve_adapter_rel(
        checkpoint_path=checkpoint_path,
        weights_path=weights_path,
        extra_metadata=meta,
        mission_last_checkpoint=mission_last_checkpoint,
    )
    adapter_rel = resolve_adapter_or_bare(finetune_dir, adapter_rel)

    return InferConfig(
        finetune_dir=finetune_dir,
        python_bin=python_bin,
        adapter_rel=adapter_rel,
        base_model=base_model,
        prompt_template=prompt_template,
        wrap_routing=wrap_routing,
    )


def _shell_path(path: str) -> str:
    """Quote a path for ssh, but leave ``~/…`` / ``$VAR/…`` unquoted so the
    remote shell expands them (``shlex.quote('~/finetune')`` → ``'~/finetune'``
    which zsh treats as a literal and fails with ``no such file``).

    Matches ``_run_bare_eval`` which interpolates recipe ``finetune_dir`` /
    ``python_bin`` without quoting.
    """
    p = (path or "").strip()
    if not p:
        return p
    if p.startswith("~/") or p.startswith("$"):
        # Recipe-owned paths only; reject spaces/metacharacters rather than quote.
        if any(c in p for c in " \t\n;|&`$(){}[]\"'"):
            raise LoraInferError(f"unsafe remote path: {p!r}", status_code=400)
        return p
    return shlex.quote(p)


def build_generate_cmd(
    cfg: InferConfig,
    prompt: str,
    *,
    max_tokens: int = 256,
    temperature: float = 0.0,
) -> str:
    """Shell command run under ``ssh host``."""
    parts = [
        f"cd {_shell_path(cfg.finetune_dir)}",
        "&&",
        _shell_path(cfg.python_bin),
        "generate_once.py",
        "--model",
        shlex.quote(cfg.base_model),
        "--adapter",
        shlex.quote(cfg.adapter_rel),
        "--max-tokens",
        str(int(max_tokens)),
        "--temperature",
        str(float(temperature)),
        "--prompt",
        shlex.quote(prompt),
    ]
    if cfg.prompt_template and cfg.wrap_routing:
        parts.extend(["--prompt-template", shlex.quote(cfg.prompt_template)])
    return " ".join(parts)


def _parse_generate_stdout(stdout: str) -> dict[str, Any]:
    """Last JSON object line wins (mlx may print progress before our line)."""
    last: Optional[dict[str, Any]] = None
    for line in (stdout or "").splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and "completion" in obj:
            last = obj
    if last is None:
        raise LoraInferError(
            "Mini generate_once produced no parseable JSON completion.",
            status_code=502,
        )
    return last


def run_lora_infer(
    *,
    model_id: str,
    prompt: str,
    cfg: InferConfig,
    max_tokens: int = 256,
    temperature: float = 0.0,
) -> InferResult:
    host = (settings.sandbox_host or "").strip()
    if not host:
        raise LoraInferError(
            "ASTRA_SANDBOX_HOST is not set; routing inference requires the Mac Mini.",
            status_code=503,
        )
    if not (prompt or "").strip():
        raise LoraInferError("prompt is required", status_code=400)

    prompt_preview = prompt.strip().replace("\n", " ")[:120]
    logger.info(
        "lora_infer: start model=%s host=%s adapter=%s base=%s wrap_routing=%s "
        "max_tokens=%s temp=%s prompt=%r",
        model_id[:8],
        host,
        cfg.adapter_rel,
        cfg.base_model,
        cfg.wrap_routing,
        max_tokens,
        temperature,
        prompt_preview,
    )
    push_generate_once(cfg.finetune_dir)
    cmd = build_generate_cmd(
        cfg, prompt.strip(), max_tokens=max_tokens, temperature=temperature
    )
    # Log command without the quoted user prompt (already logged above).
    cmd_for_log = cmd
    try:
        marker = " --prompt "
        if marker in cmd_for_log:
            head, _rest = cmd_for_log.split(marker, 1)
            cmd_for_log = head + " --prompt <redacted>"
    except Exception:
        pass
    logger.info("lora_infer: ssh %s %s", host, cmd_for_log)

    t0 = time.monotonic()
    try:
        result = subprocess.run(
            ["ssh", host, cmd],
            capture_output=True,
            text=True,
            timeout=INFER_TIMEOUT_S,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        logger.warning(
            "lora_infer: timeout model=%s after %ss adapter=%s",
            model_id[:8],
            INFER_TIMEOUT_S,
            cfg.adapter_rel,
        )
        raise LoraInferError(
            f"Mini generate timed out after {INFER_TIMEOUT_S}s",
            status_code=504,
        ) from exc
    except Exception as exc:
        logger.warning("lora_infer: ssh failed model=%s: %s", model_id[:8], exc)
        raise LoraInferError(f"SSH generate failed: {exc}", status_code=502) from exc

    latency_ms = int((time.monotonic() - t0) * 1000)
    if result.returncode != 0:
        tail = ((result.stderr or "") + "\n" + (result.stdout or ""))[-800:]
        logger.warning(
            "lora_infer: generate_once rc=%s model=%s latency_ms=%s stderr/stdout_tail=%r",
            result.returncode,
            model_id[:8],
            latency_ms,
            tail[-400:],
        )
        raise LoraInferError(
            f"generate_once exited {result.returncode}: {tail}",
            status_code=502,
        )

    payload = _parse_generate_stdout(result.stdout)
    completion = str(payload.get("completion") or "")
    adapter = str(payload.get("adapter") or cfg.adapter_rel)
    base_model = str(payload.get("model") or cfg.base_model)
    logger.info(
        "lora_infer: ok model=%s adapter=%s base=%s latency_ms=%s completion=%r",
        model_id[:8],
        adapter,
        base_model,
        latency_ms,
        completion.replace("\n", " ")[:200],
    )
    return InferResult(
        completion=completion,
        prompt_used=str(payload.get("prompt") or prompt),
        model_id=model_id,
        latency_ms=latency_ms,
        adapter=adapter,
        base_model=base_model,
    )
