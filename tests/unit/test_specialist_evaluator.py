"""Unit tests for SpecialistEvaluator._latest_checkpoint."""
from __future__ import annotations

import os
import time

from backend.evaluator.specialist import SpecialistEvaluator


def test_returns_none_when_dir_missing(tmp_path):
    result = SpecialistEvaluator._latest_checkpoint(str(tmp_path / "nonexistent"))
    assert result is None


def test_returns_none_when_dir_empty(tmp_path):
    result = SpecialistEvaluator._latest_checkpoint(str(tmp_path))
    assert result is None


def test_prefers_best_model_over_last_model(tmp_path):
    last = tmp_path / "last_model.zip"
    best = tmp_path / "best_model.zip"
    last.write_bytes(b"last")
    time.sleep(0.01)
    best.write_bytes(b"best")  # newer mtime, but should be preferred by name
    result = SpecialistEvaluator._latest_checkpoint(str(tmp_path))
    assert result == str(best)


def test_prefers_best_model_even_when_older(tmp_path):
    best = tmp_path / "best_model.zip"
    best.write_bytes(b"best")
    time.sleep(0.01)
    last = tmp_path / "last_model.zip"
    last.write_bytes(b"last")  # newer mtime
    result = SpecialistEvaluator._latest_checkpoint(str(tmp_path))
    assert result == str(best)


def test_falls_back_to_newest_when_no_best_model(tmp_path):
    old = tmp_path / "checkpoint_1000.zip"
    old.write_bytes(b"old")
    time.sleep(0.01)
    new = tmp_path / "last_model.zip"
    new.write_bytes(b"new")
    result = SpecialistEvaluator._latest_checkpoint(str(tmp_path))
    assert result == str(new)


def test_skips_hidden_files(tmp_path):
    hidden = tmp_path / ".DS_Store"
    hidden.write_bytes(b"hidden")
    result = SpecialistEvaluator._latest_checkpoint(str(tmp_path))
    assert result is None


def test_returns_only_file_when_one_checkpoint(tmp_path):
    ckpt = tmp_path / "checkpoint_500.zip"
    ckpt.write_bytes(b"ckpt")
    result = SpecialistEvaluator._latest_checkpoint(str(tmp_path))
    assert result == str(ckpt)
