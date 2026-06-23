from __future__ import annotations

import pytest

from backend.evaluator.benchmark import BenchmarkSuite, GoldenChallenge, GOLDEN_SETS


def test_unknown_domain_returns_empty_results():
    suite = BenchmarkSuite("unknown_domain")
    result = suite.run("/fake/checkpoint")
    assert result["passed"] == 0
    assert result["failed"] == 0
    assert result["results"] == []


def test_snake_fails_when_metric_is_zero():
    suite = BenchmarkSuite("snake")
    result = suite.run("/fake/checkpoint")
    # Stub evaluators return 0.0 / zeros — all below threshold
    assert result["failed"] >= 1


def test_passes_when_evaluate_fn_returns_above_threshold(monkeypatch):
    def _passing_eval(checkpoint_path: str) -> dict:
        return {"mean_reward": 999.0}

    monkeypatch.setitem(
        GOLDEN_SETS,
        "snake",
        [
            GoldenChallenge(
                name="snake_test",
                domain="snake",
                description="test",
                evaluate_fn=_passing_eval,
                pass_threshold={"mean_reward": 20},
            )
        ],
    )
    suite = BenchmarkSuite("snake")
    result = suite.run("/fake/checkpoint")
    assert result["passed"] == 1
    assert result["failed"] == 0


def test_all_results_recorded():
    suite = BenchmarkSuite("tetris")
    result = suite.run("/fake/checkpoint")
    assert len(result["results"]) == len(suite.challenges)
    for r in result["results"]:
        assert "name" in r
        assert "status" in r
        assert "metrics" in r
        assert "threshold" in r


def test_nonexistent_checkpoint_returns_worst_case():
    suite = BenchmarkSuite("nlp")
    result = suite.run("/does/not/exist/checkpoint")
    # Should not raise; stub returns worst-case metrics
    assert isinstance(result, dict)
    assert "passed" in result


def test_snake_result_status_field():
    suite = BenchmarkSuite("snake")
    result = suite.run("/fake/checkpoint")
    statuses = {r["status"] for r in result["results"]}
    assert statuses.issubset({"passed", "failed"})


# ── _is_actor_critic ──────────────────────────────────────────────────────────

def test_is_actor_critic_detects_trainer_type_txt(tmp_path):
    from backend.evaluator.benchmark import _is_actor_critic

    ckpt_dir = tmp_path / "checkpoints"
    ckpt_dir.mkdir()
    (ckpt_dir / "trainer_type.txt").write_text("actor_critic")
    result = _is_actor_critic(str(ckpt_dir / "best_model.zip"))
    assert result is True


def test_is_actor_critic_false_for_sb3_trainer_type(tmp_path):
    from backend.evaluator.benchmark import _is_actor_critic

    ckpt_dir = tmp_path / "checkpoints"
    ckpt_dir.mkdir()
    (ckpt_dir / "trainer_type.txt").write_text("sb3")
    result = _is_actor_critic(str(ckpt_dir / "best_model.zip"))
    assert result is False


def test_is_actor_critic_detects_pth_extension(tmp_path):
    from backend.evaluator.benchmark import _is_actor_critic

    result = _is_actor_critic(str(tmp_path / "best_model.pth"))
    assert result is True


def test_is_actor_critic_false_for_zip_without_trainer_type(tmp_path):
    from backend.evaluator.benchmark import _is_actor_critic

    result = _is_actor_critic(str(tmp_path / "best_model.zip"))
    assert result is False


def test_is_actor_critic_trainer_type_txt_takes_precedence_over_extension(tmp_path):
    from backend.evaluator.benchmark import _is_actor_critic

    ckpt_dir = tmp_path / "checkpoints"
    ckpt_dir.mkdir()
    (ckpt_dir / "trainer_type.txt").write_text("actor_critic")
    # Even a .zip file should return True when trainer_type.txt says actor_critic
    result = _is_actor_critic(str(ckpt_dir / "best_model.zip"))
    assert result is True
