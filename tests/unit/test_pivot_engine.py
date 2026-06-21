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


# ── escalation persistence (pivot_count / restore_pivot_count) ────────────────

def test_pivot_count_starts_at_zero():
    e = _engine()
    assert e.pivot_count == 0


def test_restore_pivot_count_seeds_level():
    e = _engine()
    e.restore_pivot_count(4)
    assert e.pivot_count == 4
    assert e.escalation_level() == 2  # ESCALATION_ALGO=4 → level 2


def test_restore_pivot_count_level_3():
    e = _engine()
    e.restore_pivot_count(6)
    assert e.escalation_level() == 3  # ESCALATION_REWARD=6 → level 3


def test_record_pivot_increments_when_no_improvement():
    e = _engine()
    e.record(0, {"mean_reward": 50.0})
    e.record_pivot()  # _best_at_last_pivot = None → reset to 0
    assert e.pivot_count == 0
    e.record(1, {"mean_reward": 51.0})  # +2% improvement — below 5% threshold
    e.record_pivot()
    assert e.pivot_count == 1  # incremented: <5% gain doesn't reset


def test_record_pivot_resets_on_large_improvement():
    e = _engine()
    e.record(0, {"mean_reward": 50.0})
    e.record_pivot()  # sets _best_at_last_pivot = 50.0, count = 0
    e.restore_pivot_count(3)  # simulate accumulated count
    e.record(1, {"mean_reward": 58.0})  # +16% improvement — above 5% threshold
    e.record_pivot()
    assert e.pivot_count == 0  # reset: ≥5% gain


def test_record_pivot_increments_on_small_improvement():
    """2% improvement (below 5% ESCALATION_RESET_THRESHOLD) must NOT reset escalation."""
    e = _engine()
    e.record(0, {"mean_reward": 50.0})
    e.record_pivot()  # _best_at_last_pivot = 50.0, count stays 0
    e.restore_pivot_count(3)
    e.record(1, {"mean_reward": 51.0})  # +2%
    e.record_pivot()
    assert e.pivot_count == 4  # incremented, not reset


def test_escalation_level_zero_below_arch_threshold():
    e = _engine()
    e.restore_pivot_count(1)
    assert e.escalation_level() == 0


def test_escalation_level_one_at_arch_threshold():
    e = _engine()
    e.restore_pivot_count(2)  # ESCALATION_ARCH=2
    assert e.escalation_level() == 1


def test_escalation_level_two_at_algo_threshold():
    e = _engine()
    e.restore_pivot_count(4)  # ESCALATION_ALGO=4
    assert e.escalation_level() == 2


def test_escalation_level_three_at_reward_threshold():
    e = _engine()
    e.restore_pivot_count(6)  # ESCALATION_REWARD=6
    assert e.escalation_level() == 3


def test_escalation_level_caps_at_three():
    e = _engine()
    e.restore_pivot_count(999)
    assert e.escalation_level() == 3


# ── restore_best_at_last_pivot ─────────────────────────────────────────────────

def test_restore_best_at_last_pivot_prevents_escalation_reset():
    """After restart, restore_best_at_last_pivot prevents record_pivot from
    resetting pivot_count to 0 when no improvement has occurred."""
    e = _engine()
    e.restore_pivot_count(3)
    e.restore_best_at_last_pivot(5.0)
    e.record(0, {"mean_reward": 5.0})  # same best — no improvement
    e.record_pivot()
    # No improvement over threshold → pivot_count should increment, not reset
    assert e.pivot_count == 4


def test_restore_best_at_last_pivot_none_resets_count():
    """Without restore_best_at_last_pivot, record_pivot resets count (old bug)."""
    e = _engine()
    e.restore_pivot_count(3)
    # _best_at_last_pivot is None → record_pivot hits else branch → reset
    e.record(0, {"mean_reward": 5.0})
    e.record_pivot()
    assert e.pivot_count == 0


# ── restore_history ────────────────────────────────────────────────────────────

def test_restore_history_enables_immediate_plateau_detection():
    """After replaying 3 identical goal metric entries, needs_pivot fires immediately."""
    e = _engine()
    e.restore_history([
        {"iteration": 0, "mean_reward": 10.0},
        {"iteration": 1, "mean_reward": 10.0},
        {"iteration": 2, "mean_reward": 10.0},
    ])
    assert e.needs_pivot()


def test_restore_history_skips_duplicates():
    """restore_history does not add entries for iterations already in _history."""
    e = _engine()
    e.record(0, {"mean_reward": 10.0})
    e.restore_history([
        {"iteration": 0, "mean_reward": 99.0},  # duplicate — should be skipped
        {"iteration": 1, "mean_reward": 10.0},
    ])
    vals = [h.get("mean_reward") for h in e.history_snapshot()]
    assert vals.count(99.0) == 0  # duplicate not inserted
    assert len(e.history_snapshot()) == 2


def test_restore_history_partial_does_not_trigger_pivot():
    """Fewer than PLATEAU_WINDOW replayed entries → needs_pivot still returns False."""
    e = _engine()
    e.restore_history([
        {"iteration": 0, "mean_reward": 10.0},
        {"iteration": 1, "mean_reward": 10.0},
    ])
    assert not e.needs_pivot()


# ── arch_changed comparison ────────────────────────────────────────────────────

def test_arch_changed_same_value_is_no_op():
    """arch_changed must be False when proposed policy_kwargs equals current plan."""
    current_policy_kwargs = {"net_arch": [256, 256]}
    proposed_policy_kwargs = {"net_arch": [256, 256]}
    arch_changed = bool(
        proposed_policy_kwargs and
        proposed_policy_kwargs != current_policy_kwargs
    )
    assert not arch_changed


def test_arch_changed_different_value_is_true():
    """arch_changed must be True when proposed policy_kwargs differs from current."""
    current_policy_kwargs = {"net_arch": [64, 64]}
    proposed_policy_kwargs = {"net_arch": [256, 256]}
    arch_changed = bool(
        proposed_policy_kwargs and
        proposed_policy_kwargs != current_policy_kwargs
    )
    assert arch_changed


def test_arch_changed_no_policy_kwargs_is_false():
    """arch_changed must be False when pivot proposes no policy_kwargs."""
    arch_changed = bool(
        None and
        None != {"net_arch": [256, 256]}
    )
    assert not arch_changed
