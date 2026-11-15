"""Hermetic E2E for Data → train → Eval → FailSlice → focus (plan Step 6)."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest
import yaml

from astradata.fail_slice import FailSlice, FailSliceStore
from astradata.registry import ensure_dataset
from astraeval.gate import reject_over_ceiling
from astraeval.registry import get_suite, tournament_seed
from backend.agent.code_generator import CodeGenerator, _resolve_focus_cases
from backend.config import settings
from backend.loop.state_machine import LoopStateMachine


def _tiny_ml1m(tmp_path: Path) -> Path:
    cache = tmp_path / "dl"
    ml = cache / "ml-1m"
    ml.mkdir(parents=True)
    lines = []
    for u in range(1, 8):
        for i, item in enumerate([11, 22, 33, 44, 55]):
            lines.append(f"{u}::{item}::4::{2000 + u * 20 + i}")
    (ml / "ratings.dat").write_text("\n".join(lines) + "\n")
    return cache


def test_e2e_a_hstu_dataset_suite_failslice(tmp_path: Path, monkeypatch):
    """E2E-A: dataset@rev → HstuHr10Suite → FailSlice linked on plan."""
    cache = _tiny_ml1m(tmp_path)
    data_root = tmp_path / "astradata"
    handle = ensure_dataset(
        "hstu_ml1m", 1, data_root=data_root, download_cache=cache, min_len=3
    )
    assert handle.ref == "hstu_ml1m@1"

    # Recipe declares axes
    recipe_path = Path(settings.recipes_path) / "hstu_ml1m_v1.yaml"
    with open(recipe_path) as f:
        recipe = yaml.safe_load(f)
    assert recipe["dataset_id"] == "hstu_ml1m"
    assert recipe["eval_suite"] == "hstu_hr10"
    assert reject_over_ceiling({"hr_at_10": 0.99}, "hstu_hr10@1") is not None
    assert reject_over_ceiling({"hr_at_10": 0.05}, "hstu_hr10@1") is None

    # Suite scores checkpoint-embedded metric (no torch re-eval needed)
    suite = get_suite("hstu_hr10", 1)
    result = suite.run({"hr_at_10": 0.11, "n_items": 5})
    assert result.primary == pytest.approx(0.11)

    bank = tmp_path / "fail_slices"
    monkeypatch.setattr(settings, "data_path", str(tmp_path))
    sm = LoopStateMachine.__new__(LoopStateMachine)
    plan = {
        "task_type": "hstu",
        "dataset_id": "hstu_ml1m",
        "dataset_rev": 1,
        "suite_id": "hstu_hr10",
        "suite_rev": 1,
    }
    path = sm._emit_fail_slice(
        "mission-hstu-e2e",
        plan,
        primary=result.primary,
        primary_metric=result.primary_metric,
        case_ids=result.case_ids,
        population_fingerprint=result.population_fingerprint,
        diagnostics=result.diagnostics,
    )
    assert path
    assert plan["last_fail_slice"]
    store = FailSliceStore(bank_root=Path(settings.data_path) / "astradata" / "fail_slices")
    latest = store.latest()
    assert latest is not None
    assert latest.dataset_ref == "hstu_ml1m@1"
    assert latest.suite_id == "hstu_hr10"


def test_e2e_b_ensemble_failslice_to_focus_cases(tmp_path: Path, monkeypatch):
    """E2E-B: FailSlice → focus_cases CSV → same suite_id@rev fingerprint."""
    monkeypatch.setattr(settings, "data_path", str(tmp_path))
    bank = Path(settings.data_path) / "astradata" / "fail_slices"
    store = FailSliceStore(bank_root=bank)
    store.write(
        FailSlice(
            suite_id="routing_bare",
            suite_rev=5,
            case_ids=["case_a", "case_b"],
            primary=0.70,
            mission_id="ens-e2e-001",
            population_fingerprint="fp-abc",
        )
    )

    suite = get_suite("routing_bare", 5)
    r1 = suite.run(
        None,
        report={
            "primary": 0.70,
            "case_ids": ["case_a", "case_b"],
            "frozen_clock": "clock-1",
            "case_population": ["case_a", "case_b", "case_c"],
            "population_fingerprint": "fp-abc",
        },
    )
    # Follow-on focus ⊆ FailSlice (ensemble / fail_slice_focus opts into bank)
    focus = _resolve_focus_cases({"use_case": "ensemble"})
    assert focus == "case_a,case_b"
    focus_set = set(focus.split(","))
    assert focus_set <= set(r1.case_ids)
    assert _resolve_focus_cases({}) == ""  # bare hp must not pick up stale bank

    # Re-gate same suite_rev — fingerprint must not drift when population/clock fixed
    r2 = suite.run(
        None,
        report={
            "primary": 0.85,
            "case_ids": ["case_a"],
            "frozen_clock": "clock-1",
            "case_population": ["case_a", "case_b", "case_c"],
            "population_fingerprint": "fp-abc",
        },
    )
    assert r2.suite_rev == r1.suite_rev == 5
    assert r2.population_fingerprint == r1.population_fingerprint


def test_e2e_c_rl_suite_seeds():
    """E2E-C: tournament seeds owned by Eval."""
    assert tournament_seed(0, suite_id="rl_tournament", suite_rev=1) == 2000
    assert tournament_seed(3, suite_id="rl_tournament", suite_rev=1) == 2003


def test_hstu_wrapper_includes_dataset_axes():
    gen = CodeGenerator(provider=MagicMock())
    plan = {
        "task_type": "hstu",
        "recipe": "hstu_ml1m_v1",
        "dataset_id": "hstu_ml1m",
        "dataset_rev": 1,
        "hyperparameters": {},
    }
    ctx = gen._build_hstu_context("m1", plan, "/tmp/ckpts")
    assert ctx["dataset_id"] == "hstu_ml1m"
    assert ctx["dataset_rev"] == "1"
    code = gen.__class__.__module__  # noqa: keep import path
    from backend.agent import code_generator as cg

    wrapped = cg._HSTU_LOCAL_WRAPPER.format(**ctx)
    assert "--dataset-id" in wrapped
    assert "hstu_ml1m" in wrapped
    assert "--dataset-rev" in wrapped


def test_synthetic_hstu_train_writes_item_map(tmp_path: Path):
    """Short synthetic train proves checkpoint contract (item_map + dataset axes)."""
    import argparse
    from backend.trainers.hstu_train import train as hstu_train

    args = argparse.Namespace(
        data_dir=str(tmp_path / "unused"),
        dataset_id="hstu_ml1m",
        dataset_rev=1,
        out=str(tmp_path / "out"),
        epochs=1,
        batch_size=32,
        lr=1e-3,
        weight_decay=0.01,
        d_model=32,
        n_layers=1,
        n_heads=2,
        max_len=10,
        dropout=0.0,
        seed=0,
        synthetic=True,
        synthetic_users=40,
        synthetic_items=20,
    )
    hstu_train(args)
    import torch

    ckpt = torch.load(tmp_path / "out" / "best.pt", map_location="cpu", weights_only=False)
    assert "item_map" in ckpt
    assert ckpt["dataset_id"] == "hstu_ml1m"
    assert ckpt["hr_at_10"] is not None
    # Suite can score the checkpoint dict path via embedded metric
    suite = get_suite("hstu_hr10", 1)
    scored = suite.run(tmp_path / "out" / "best.pt")
    assert scored.primary == pytest.approx(float(ckpt["hr_at_10"]))
