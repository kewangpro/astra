"""Product recipe search-path resolution."""
from __future__ import annotations

import os
from pathlib import Path

import pytest
import yaml

from backend.services.recipe_paths import (
    find_recipe_file,
    iter_disk_recipe_filenames,
    parse_product_recipes,
    recipe_search_dirs,
)


def test_parse_product_recipes_skips_missing_dirs(tmp_path, monkeypatch):
    good = tmp_path / "ensemble_pack"
    good.mkdir()
    raw = f"ensemble:{good},missing:{tmp_path / 'nope'},badentry"
    monkeypatch.setattr("backend.config.settings.product_recipes", raw)
    parsed = parse_product_recipes()
    assert list(parsed) == ["ensemble"]
    assert parsed["ensemble"] == str(good.resolve())


def test_recipe_search_dirs_platform_first(tmp_path, monkeypatch):
    plat = tmp_path / "plat"
    pack = tmp_path / "pack"
    plat.mkdir()
    pack.mkdir()
    monkeypatch.setattr("backend.config.settings.recipes_path", str(plat))
    monkeypatch.setattr(
        "backend.config.settings.product_recipes", f"ensemble:{pack}",
    )
    dirs = recipe_search_dirs()
    assert dirs[0] == str(plat.resolve())
    assert dirs[1] == str(pack.resolve())
    assert recipe_search_dirs(use_case="ensemble") == dirs
    assert recipe_search_dirs(use_case="other") == [str(plat.resolve())]


def test_find_recipe_file_prefers_platform_over_pack(tmp_path, monkeypatch):
    plat = tmp_path / "plat"
    pack = tmp_path / "pack"
    plat.mkdir()
    pack.mkdir()
    (plat / "snake_ppo_v1.yaml").write_text("name: snake_ppo_v1\ntask_type: rl\n")
    (pack / "ensemble_dpo_v1.yaml").write_text(
        "name: ensemble_dpo_v1\ntask_type: dpo\nlearner_id: mlx_dpo\n"
    )
    (pack / "snake_ppo_v1.yaml").write_text("name: pack_snake\ntask_type: rl\n")
    monkeypatch.setattr("backend.config.settings.recipes_path", str(plat))
    monkeypatch.setattr(
        "backend.config.settings.product_recipes", f"ensemble:{pack}",
    )
    snake = find_recipe_file("snake_ppo_v1")
    assert snake == str((plat / "snake_ppo_v1.yaml").resolve())
    dpo = find_recipe_file("ensemble_dpo_v1")
    assert dpo == str((pack / "ensemble_dpo_v1.yaml").resolve())
    assert find_recipe_file("nope") is None


def test_iter_disk_recipe_filenames_dedupes_by_stem(tmp_path, monkeypatch):
    plat = tmp_path / "plat"
    pack = tmp_path / "pack"
    plat.mkdir()
    pack.mkdir()
    (plat / "a.yaml").write_text("name: a\n")
    (pack / "a.yaml").write_text("name: a_pack\n")
    (pack / "b.yaml").write_text("name: b\n")
    monkeypatch.setattr("backend.config.settings.recipes_path", str(plat))
    monkeypatch.setattr(
        "backend.config.settings.product_recipes", f"ensemble:{pack}",
    )
    items = list(iter_disk_recipe_filenames())
    stems = [os.path.splitext(f)[0] for _, f in items]
    assert stems == ["a", "b"]
    assert items[0][0] == str(plat.resolve())


def test_load_recipe_for_env_reads_product_pack(tmp_path, monkeypatch):
    plat = tmp_path / "plat"
    pack = tmp_path / "pack"
    plat.mkdir()
    pack.mkdir()
    (pack / "ensemble_dpo_v1.yaml").write_text(
        yaml.dump({
            "name": "ensemble_dpo_v1",
            "task_type": "dpo",
            "learner_id": "mlx_dpo",
            "hyperparameters": {"beta": 0.42, "finetune_dir": "/tmp/ft"},
        })
    )
    monkeypatch.setattr("backend.config.settings.recipes_path", str(plat))
    monkeypatch.setattr(
        "backend.config.settings.product_recipes", f"ensemble:{pack}",
    )
    from backend.agent.code_generator import _load_recipe_for_env, _resolve_hyperparams

    rcp = _load_recipe_for_env("ensemble_dpo_v1")
    assert rcp["hyperparameters"]["beta"] == 0.42
    hp = _resolve_hyperparams("ensemble_dpo_v1", {})
    assert hp["beta"] == 0.42
