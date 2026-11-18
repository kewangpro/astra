"""Unit tests for Champion / non-game LoRA inference helpers."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from backend.services.lora_infer import (
    InferConfig,
    LoraInferError,
    adapter_rel_from_path,
    build_generate_cmd,
    resolve_adapter_rel,
    resolve_infer_config,
    run_lora_infer,
)


class TestAdapterRelFromPath:
    def test_extracts_adapters_suffix(self):
        assert (
            adapter_rel_from_path("~/finetune/adapters/astra_abcd1234_iter0/best")
            == "adapters/astra_abcd1234_iter0/best"
        )

    def test_already_relative(self):
        assert (
            adapter_rel_from_path("adapters/astra_abcd1234_iter0")
            == "adapters/astra_abcd1234_iter0"
        )

    def test_local_mission_mirror_without_adapters_returns_none(self):
        assert adapter_rel_from_path("data/missions/abc/checkpoints/best") is None

    def test_empty(self):
        assert adapter_rel_from_path(None) is None
        assert adapter_rel_from_path("") is None


class TestResolveAdapterRel:
    def test_prefers_checkpoint_path(self):
        rel = resolve_adapter_rel(
            checkpoint_path="adapters/astra_deadbeef_iter0/best",
            weights_path=None,
            extra_metadata={},
            mission_last_checkpoint=None,
        )
        assert rel == "adapters/astra_deadbeef_iter0/best"

    def test_falls_back_to_mission_last_checkpoint(self):
        rel = resolve_adapter_rel(
            checkpoint_path="data/missions/x/checkpoints/best",
            weights_path=None,
            extra_metadata={},
            mission_last_checkpoint="adapters/astra_deadbeef_iter3/best",
        )
        assert rel == "adapters/astra_deadbeef_iter3/best"

    def test_raises_when_missing(self):
        with pytest.raises(LoraInferError) as ei:
            resolve_adapter_rel(
                checkpoint_path=None,
                weights_path=None,
                extra_metadata={},
                mission_last_checkpoint=None,
            )
        assert ei.value.status_code == 400


class TestResolveInferConfig:
    def test_routing_domain_gets_default_template(self):
        with patch("backend.services.lora_infer.resolve_adapter_or_bare", side_effect=lambda _fd, a: a):
            cfg = resolve_infer_config(
                checkpoint_path="adapters/astra_abcd1234_iter0/best",
                weights_path=None,
                extra_metadata={},
                mission_plan=None,
                mission_last_checkpoint=None,
                domain="routing",
            )
        assert cfg.wrap_routing is True
        assert cfg.prompt_template.endswith("conductor_min.md")
        assert cfg.adapter_rel == "adapters/astra_abcd1234_iter0/best"
        assert "gemma" in cfg.base_model


class TestBuildGenerateCmd:
    def test_includes_adapter_model_prompt_and_template(self):
        cfg = InferConfig(
            finetune_dir="~/finetune",
            python_bin="~/finetune-env/bin/python",
            adapter_rel="adapters/astra_abcd_iter0/best",
            base_model="mlx-community/gemma-3-12b-it-4bit",
            prompt_template="backend/prompts/conductor_min.md",
            wrap_routing=True,
        )
        cmd = build_generate_cmd(cfg, "route this query", max_tokens=128, temperature=0.0)
        assert "generate_once.py" in cmd
        assert "--adapter" in cmd
        assert "adapters/astra_abcd_iter0/best" in cmd
        assert "--prompt-template" in cmd
        assert "conductor_min.md" in cmd
        assert "route this query" in cmd
        # Tilde must stay unquoted so the remote shell expands it.
        assert "cd ~/finetune" in cmd
        assert "'~/finetune'" not in cmd
        assert "~/finetune-env/bin/python" in cmd

    def test_omits_template_when_not_wrapping(self):
        cfg = InferConfig(
            finetune_dir="~/finetune",
            python_bin="~/finetune-env/bin/python",
            adapter_rel="adapters/x",
            base_model="m",
            prompt_template="",
            wrap_routing=False,
        )
        cmd = build_generate_cmd(cfg, "hi")
        assert "--prompt-template" not in cmd


class TestRunLoraInfer:
    def test_503_without_sandbox_host(self):
        cfg = InferConfig(
            finetune_dir="~/finetune",
            python_bin="~/finetune-env/bin/python",
            adapter_rel="adapters/x",
            base_model="m",
            prompt_template="",
            wrap_routing=False,
        )
        with patch("backend.services.lora_infer.settings.sandbox_host", ""):
            with pytest.raises(LoraInferError) as ei:
                run_lora_infer(model_id="mid", prompt="hi", cfg=cfg)
        assert ei.value.status_code == 503

    def test_parses_json_from_ssh_stdout(self):
        cfg = InferConfig(
            finetune_dir="~/finetune",
            python_bin="~/finetune-env/bin/python",
            adapter_rel="adapters/astra_x_iter0/best",
            base_model="mlx-community/gemma-3-12b-it-4bit",
            prompt_template="backend/prompts/conductor_min.md",
            wrap_routing=True,
        )
        stdout = (
            "Loading…\n"
            '{"completion": "{\\"tasks\\":[{\\"skill\\":\\"Market Analysis\\"}]}",'
            ' "prompt": "full-prompt", "adapter": "adapters/astra_x_iter0/best",'
            ' "model": "mlx-community/gemma-3-12b-it-4bit"}\n'
        )
        with patch("backend.services.lora_infer.settings.sandbox_host", "mac-mini.local"), \
             patch("backend.services.lora_infer.push_generate_once"), \
             patch("backend.services.lora_infer.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout=stdout, stderr="")
            result = run_lora_infer(model_id="mid-1", prompt="What is AAPL?", cfg=cfg)

        assert result.model_id == "mid-1"
        assert "Market Analysis" in result.completion
        assert result.prompt_used == "full-prompt"
        assert result.latency_ms >= 0
        # ssh host + remote shell command
        args = mock_run.call_args[0][0]
        assert args[0] == "ssh"
        assert args[1] == "mac-mini.local"
        assert "generate_once.py" in args[2]
