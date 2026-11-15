"""Unit tests for AstraData registry + FailSlice."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from astradata.fail_slice import FailSlice, FailSliceStore
from astradata.hstu_ml1m import materialize_hstu_ml1m
from astradata.registry import ensure_dataset, parse_ref, resolve
from astradata.splits import assert_disjoint, leave_one_out_user_split


def test_parse_ref():
    assert parse_ref("hstu_ml1m@3") == ("hstu_ml1m", 3)
    assert parse_ref("hstu_ml1m") == ("hstu_ml1m", 1)


def test_fail_slice_round_trip(tmp_path: Path):
    store = FailSliceStore(bank_root=tmp_path)
    slice_ = FailSlice(
        suite_id="routing_bare",
        suite_rev=5,
        case_ids=["a", "b", "c"],
        primary=0.72,
        mission_id="abcd1234-ffff",
        population_fingerprint="deadbeef",
    )
    path = store.write(slice_)
    loaded = store.read(path)
    assert loaded.suite_id == "routing_bare"
    assert loaded.suite_rev == 5
    assert loaded.case_ids == ["a", "b", "c"]
    assert store.focus_cases_csv() == "a,b,c"
    assert store.latest(suite_id="routing_bare") is not None


def test_routing_focus_bank(tmp_path: Path):
    handle = ensure_dataset("routing_focus", 1, data_root=tmp_path)
    assert handle.ref == "routing_focus@1"
    assert handle.content_hash
    again = resolve("routing_focus", 1, data_root=tmp_path)
    assert again.content_hash == handle.content_hash


def test_hstu_ml1m_materialize_synthetic_style(tmp_path: Path):
    """Materialize without network: write a tiny ratings.dat then freeze."""
    cache = tmp_path / "dl"
    ml = cache / "ml-1m"
    ml.mkdir(parents=True)
    # uid::iid::rating::ts — enough for min_len=3
    lines = []
    for u in range(1, 6):
        for i, item in enumerate([10, 20, 30, 40]):
            lines.append(f"{u}::{item}::5::{1000 + u * 10 + i}")
    (ml / "ratings.dat").write_text("\n".join(lines) + "\n")

    rev_root = tmp_path / "hstu_ml1m" / "1"
    meta = materialize_hstu_ml1m(rev_root, rev=1, min_len=3, download_cache=cache)
    assert meta["n_users"] == 5
    assert meta["n_items"] == 4
    assert (rev_root / "item_map.json").is_file()
    assert (rev_root / "sequences.json").is_file()

    # ensure_dataset reuses frozen hash (no re-download)
    handle = ensure_dataset(
        "hstu_ml1m", 1, data_root=tmp_path, download_cache=cache, min_len=3
    )
    assert handle.content_hash == meta["content_hash"]
    same = ensure_dataset(
        "hstu_ml1m", 1, data_root=tmp_path, download_cache=cache, min_len=3
    )
    assert same.content_hash == handle.content_hash


def test_splits_disjoint():
    assert_disjoint([1, 2], [3, 4], label="ok")
    with pytest.raises(ValueError):
        assert_disjoint([1, 2], [2, 3], label="leak")
    split = leave_one_out_user_split({1: [1, 2, 3], 2: [4]})
    assert 1 in split["train"]
    assert 2 not in split["train"]  # too short
