from __future__ import annotations

import pytest

from backend.loop.pivots import PivotEngine


TARGET = {"mean_reward": 100.0}


def _engine():
    return PivotEngine(TARGET)


def test_goal_not_met_below_threshold():
    e = _engine()
    assert not e.is_goal_met({"mean_reward": 99.9})


def test_goal_met_at_threshold():
    e = _engine()
    assert e.is_goal_met({"mean_reward": 100.0})


def test_goal_met_above_threshold():
    e = _engine()
    assert e.is_goal_met({"mean_reward": 250.0})


def test_no_pivot_before_window():
    e = _engine()
    e.record(0, {"mean_reward": 10.0})
    e.record(1, {"mean_reward": 10.0})
    assert not e.needs_pivot()  # only 2 points, window=3


def test_pivot_triggered_on_plateau():
    e = _engine()
    for i in range(3):
        e.record(i, {"mean_reward": 10.0})
    assert e.needs_pivot()


def test_no_pivot_when_improving():
    e = _engine()
    e.record(0, {"mean_reward": 10.0})
    e.record(1, {"mean_reward": 15.0})
    e.record(2, {"mean_reward": 20.0})
    assert not e.needs_pivot()


def test_best_metric_value_empty():
    e = _engine()
    assert e.best_metric_value() is None


def test_best_metric_value_tracked():
    e = _engine()
    e.record(0, {"mean_reward": 10.0})
    e.record(1, {"mean_reward": 50.0})
    e.record(2, {"mean_reward": 30.0})
    assert e.best_metric_value() == 50.0


def test_best_metric_iteration_returns_correct_iter():
    e = _engine()
    e.record(0, {"mean_reward": 10.0})
    e.record(3, {"mean_reward": 50.0})
    e.record(5, {"mean_reward": 30.0})
    assert e.best_metric_iteration() == 3


def test_best_metric_iteration_seed_entry_returns_none():
    e = _engine()
    # Seed entry uses iteration=-1; best_metric_iteration should hide that
    e.record(-1, {"mean_reward": 52.0})
    assert e.best_metric_iteration() is None


def test_best_metric_iteration_seed_beaten_by_real_iter():
    e = _engine()
    e.record(-1, {"mean_reward": 52.0})
    e.record(7, {"mean_reward": 60.0})
    assert e.best_metric_iteration() == 7
    assert e.best_metric_value() == 60.0


def test_best_metric_iteration_empty():
    e = _engine()
    assert e.best_metric_iteration() is None


def test_history_snapshot_is_copy():
    e = _engine()
    e.record(0, {"mean_reward": 5.0})
    snap = e.history_snapshot()
    snap.append({"iteration": 999, "mean_reward": 999})
    assert len(e.history_snapshot()) == 1
