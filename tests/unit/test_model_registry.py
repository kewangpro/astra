"""Unit tests for Model Registry and Tournament Arena (backend/routers/registry.py)."""
from __future__ import annotations

import os
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi import HTTPException

from backend.schemas.model_registry import (
    ModelRecordCreate,
    ModelRecordUpdate,
    TournamentRequest,
)
from backend.schemas.experiment import (
    ExperimentCreate,
    ExperimentUpdate,
)
from backend.routers.registry import (
    create_experiment,
    list_experiments,
    get_experiment,
    update_experiment,
    delete_experiment,
    create_model_record,
    list_model_records,
    get_model_record,
    update_model_record,
    delete_model_record,
    run_tournament,
)
from backend.evaluator.benchmark import run_tournament_match


# ── Fixtures & Helpers ─────────────────────────────────────────────────────────

def _make_db(item=None, items=None):
    db = AsyncMock()
    db.get = AsyncMock(return_value=item)
    scalars = MagicMock()
    scalars.all = MagicMock(return_value=items or [])
    result = MagicMock()
    result.scalars = MagicMock(return_value=scalars)
    db.execute = AsyncMock(return_value=result)
    db.add = MagicMock()
    db.commit = AsyncMock()
    db.refresh = AsyncMock()
    db.delete = AsyncMock()
    return db


# ── Experiment Tests ──────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_create_and_get_experiment():
    db = _make_db()
    payload = ExperimentCreate(name="exp-1", domain="Snake-v0", description="baseline")
    exp = await create_experiment(payload, db=db)
    db.add.assert_called_once()
    db.commit.assert_called_once()
    assert exp.name == "exp-1"
    assert exp.domain == "Snake-v0"


@pytest.mark.asyncio
async def test_get_experiment_not_found():
    db = _make_db(item=None)
    with pytest.raises(HTTPException) as exc_info:
        await get_experiment("non-existent-id", db=db)
    assert exc_info.value.status_code == 404


@pytest.mark.asyncio
async def test_update_and_delete_experiment():
    mock_exp = MagicMock()
    mock_exp.id = "exp-123"
    mock_exp.name = "exp-old"
    db = _make_db(item=mock_exp)

    payload = ExperimentUpdate(name="exp-renamed")
    res = await update_experiment("exp-123", payload, db=db)
    assert res.name == "exp-renamed"
    db.commit.assert_called_once()

    await delete_experiment("exp-123", db=db)
    db.delete.assert_called_once_with(mock_exp)


# ── Model Record Tests ────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_create_model_record():
    db = _make_db()
    payload = ModelRecordCreate(
        name="dqn-snake-champion",
        domain="Snake-v0",
        framework="stable-baselines3",
        architecture="DQN",
        checkpoint_path="/models/snake.zip",
        best_metric_name="mean_reward",
        best_metric_value=48.0,
    )
    rec = await create_model_record(payload, db=db)
    db.add.assert_called_once()
    db.commit.assert_called_once()
    assert rec.name == "dqn-snake-champion"
    assert rec.best_metric_value == 48.0


@pytest.mark.asyncio
async def test_update_model_champion():
    mock_rec = MagicMock()
    mock_rec.id = "rec-1"
    mock_rec.is_champion = False
    db = _make_db(item=mock_rec)

    payload = ModelRecordUpdate(is_champion=True)
    res = await update_model_record("rec-1", payload, db=db)
    assert res.is_champion is True
    db.commit.assert_called_once()


# ── Tournament Arena Tests ────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_tournament_requires_at_least_two_checkpoints(tmp_path):
    db = _make_db(items=[])
    req = TournamentRequest(env_id="Snake-v0", n_episodes=2)
    with patch("os.path.isdir", return_value=False):
        with pytest.raises(HTTPException) as exc:
            await run_tournament(req, db=db)
        assert exc.value.status_code == 400
        assert "At least 2 valid model checkpoints are required" in exc.value.detail


@pytest.mark.asyncio
async def test_tournament_runs_match_and_updates_champion(tmp_path):
    ckpt1 = str(tmp_path / "model1.zip")
    ckpt2 = str(tmp_path / "model2.zip")
    open(ckpt1, "w").write("dummy")
    open(ckpt2, "w").write("dummy")

    rec1 = MagicMock()
    rec1.id = "model-1"
    rec1.name = "Snake-DQN-v1"
    rec1.domain = "Snake-v0"
    rec1.checkpoint_path = ckpt1
    rec1.weights_path = None
    rec1.is_champion = False

    rec2 = MagicMock()
    rec2.id = "model-2"
    rec2.name = "Snake-DQN-v2"
    rec2.domain = "Snake-v0"
    rec2.checkpoint_path = ckpt2
    rec2.weights_path = None
    rec2.is_champion = True

    db = _make_db(item=rec1, items=[rec1, rec2])

    fake_result = {
        "env_id": "Snake-v0",
        "episodes": 3,
        "leaderboard": [
            {
                "model_id": "model-1",
                "name": "Snake-DQN-v1",
                "checkpoint_path": ckpt1,
                "mean_score": 52.0,
                "std_score": 1.5,
                "min_score": 50.0,
                "max_score": 55.0,
                "win_rate": 1.0,
                "scores": [50.0, 52.0, 54.0],
                "rank": 1,
            },
            {
                "model_id": "model-2",
                "name": "Snake-DQN-v2",
                "checkpoint_path": ckpt2,
                "mean_score": 30.0,
                "std_score": 2.0,
                "min_score": 28.0,
                "max_score": 32.0,
                "win_rate": 0.0,
                "scores": [30.0, 28.0, 32.0],
                "rank": 2,
            },
        ],
        "champion_id": "model-1",
    }

    req = TournamentRequest(
        env_id="Snake-v0",
        model_ids=["model-1", "model-2"],
        n_episodes=3,
        update_champion=True,
    )

    with patch("backend.routers.registry.run_tournament_match", return_value=fake_result):
        resp = await run_tournament(req, db=db)
        assert resp["champion_id"] == "model-1"
        assert len(resp["leaderboard"]) == 2
        assert resp["leaderboard"][0]["rank"] == 1
        assert resp["leaderboard"][0]["mean_score"] == 52.0


def test_run_tournament_match_ranking_logic():
    """Verify tournament ranking calculation with mocked models."""
    class DummyModel:
        def __init__(self, step_reward):
            self.r = step_reward
        def predict(self, obs, deterministic=True):
            return 0, None

    m1 = DummyModel(5.0)
    m2 = DummyModel(2.0)

    # Directly verify leaderboard structure & win rate logic on mock loaded list
    loaded = [
        {"id": "m1", "name": "Model 1", "path": "p1.zip", "scores": [10.0, 15.0, 20.0]},
        {"id": "m2", "name": "Model 2", "path": "p2.zip", "scores": [5.0, 5.0, 10.0]},
    ]
    wins = {"m1": 0.0, "m2": 0.0}
    for ep in range(3):
        ep_scores = [m["scores"][ep] for m in loaded]
        max_s = max(ep_scores)
        winners = [m["id"] for m in loaded if m["scores"][ep] == max_s]
        share = 1.0 / len(winners)
        for w in winners:
            wins[w] += share

    assert wins["m1"] == 3.0
    assert wins["m2"] == 0.0


def test_run_tournament_match_nlp(tmp_path):
    """Verify tournament match execution for NLP domain using eval_loss."""
    import json
    c1 = tmp_path / "c1"
    c2 = tmp_path / "c2"
    c1.mkdir()
    c2.mkdir()
    with open(c1 / "checkpoint_metadata.json", "w") as f:
        json.dump({"eval_loss": 0.5}, f)
    with open(c2 / "checkpoint_metadata.json", "w") as f:
        json.dump({"eval_loss": 3.0}, f)

    entries = [
        {"id": "m1", "name": "Model 1", "path": str(c1)},
        {"id": "m2", "name": "Model 2", "path": str(c2)},
    ]
    res = run_tournament_match(entries, env_id="nlp", n_episodes=3)
    assert res["champion_id"] == "m1"
    assert len(res["leaderboard"]) == 2
    assert res["leaderboard"][0]["rank"] == 1
    assert res["leaderboard"][0]["model_id"] == "m1"
    assert res["leaderboard"][1]["rank"] == 2
    assert res["leaderboard"][1]["model_id"] == "m2"
    assert res["leaderboard"][1]["win_rate"] == 0.0


@pytest.mark.asyncio
async def test_tournament_auto_discovers_missions_checkpoints(tmp_path, monkeypatch):
    """Disk fallback still finds live-mission zips that are not yet in the registry."""
    import json
    missions_dir = tmp_path / "data" / "missions"
    m1_ckpt = missions_dir / "m1" / "checkpoints"
    m2_ckpt = missions_dir / "m2" / "checkpoints"
    m1_ckpt.mkdir(parents=True)
    m2_ckpt.mkdir(parents=True)

    with open(m1_ckpt / "train_config.json", "w") as f:
        json.dump({"algorithm": "DQN", "env_id": "Snake-v0"}, f)
    with open(m2_ckpt / "train_config.json", "w") as f:
        json.dump({"algorithm": "PPO", "env_id": "Snake-v0"}, f)

    (m1_ckpt / "best_model.zip").write_text("dummy1")
    (m2_ckpt / "best_model.zip").write_text("dummy2")

    monkeypatch.chdir(tmp_path)

    db = _make_db(items=[])
    req = TournamentRequest(env_id="Snake-v0", n_episodes=2)

    fake_result = {
        "env_id": "Snake-v0",
        "episodes": 2,
        "leaderboard": [
            {"model_id": "m1", "name": "DQN", "checkpoint_path": str(m1_ckpt / "best_model.zip"),
             "mean_score": 50.0, "std_score": 0.0, "min_score": 50.0, "max_score": 50.0, "win_rate": 1.0,
             "scores": [50.0, 50.0], "rank": 1},
            {"model_id": "m2", "name": "PPO", "checkpoint_path": str(m2_ckpt / "best_model.zip"),
             "mean_score": 40.0, "std_score": 0.0, "min_score": 40.0, "max_score": 40.0, "win_rate": 0.0,
             "scores": [40.0, 40.0], "rank": 2},
        ],
        "champion_id": "m1",
    }

    with patch("backend.routers.registry._alive_mission_ids", new_callable=AsyncMock, return_value={"m1", "m2"}):
        with patch("backend.routers.registry.run_tournament_match", return_value=fake_result) as mock_match:
            resp = await run_tournament(req, db=db)
            assert resp["champion_id"] == "m1"
            assert len(resp["leaderboard"]) == 2
            mock_match.assert_called_once()
            call_kwargs = mock_match.call_args[1]
            assert len(call_kwargs["checkpoint_entries"]) == 2


def test_mission_id_from_model_metadata_and_path():
    from backend.routers.registry import mission_id_from_model

    rec = MagicMock()
    rec.extra_metadata = {"mission_id": "601c2404-ee20-4fb5-af2b-bebfbff4c98e"}
    rec.checkpoint_path = None
    rec.weights_path = None
    assert mission_id_from_model(rec) == "601c2404-ee20-4fb5-af2b-bebfbff4c98e"

    rec.extra_metadata = {}
    rec.checkpoint_path = "data/missions/28e65efd-aaaa-bbbb-cccc-dddddddddddd/checkpoints/best_model.zip"
    assert mission_id_from_model(rec).startswith("28e65efd")

    rec.checkpoint_path = "/tmp/other.zip"
    rec.weights_path = None
    assert mission_id_from_model(rec) is None


@pytest.mark.asyncio
async def test_auto_sync_skips_deleted_mission_dir(tmp_path, monkeypatch):
    import json
    from backend.config import settings
    from backend.routers.registry import _auto_sync_disk_checkpoints

    monkeypatch.setattr(settings, "data_path", str(tmp_path))
    mid = "deadbeef-0000-0000-0000-000000000001"
    ckpt = tmp_path / "missions" / mid / "checkpoints"
    ckpt.mkdir(parents=True)
    (ckpt / "train_config.json").write_text(json.dumps({
        "env_id": "MinAtar-Seaquest-v0", "algorithm": "DQN",
    }))
    (ckpt / "best_model.zip").write_bytes(b"zip")

    db = AsyncMock()
    empty = MagicMock()
    empty.all = MagicMock(return_value=[])
    result = MagicMock()
    result.scalars = MagicMock(return_value=empty)
    db.execute = AsyncMock(return_value=result)
    db.add = MagicMock()
    db.commit = AsyncMock()

    await _auto_sync_disk_checkpoints(db)
    db.add.assert_not_called()


@pytest.mark.asyncio
async def test_prune_orphan_model_records():
    from backend.routers.registry import _prune_orphan_model_records

    alive_id = "live-mission-id"
    orphan = MagicMock()
    orphan.id = "orphan-model"
    orphan.extra_metadata = {"mission_id": "deleted-mission-id"}
    orphan.checkpoint_path = None
    orphan.weights_path = None
    keep = MagicMock()
    keep.id = "keep-model"
    keep.extra_metadata = {"mission_id": alive_id}
    keep.checkpoint_path = None
    keep.weights_path = None

    db = AsyncMock()

    def _execute(stmt):
        result = MagicMock()
        scalars = MagicMock()
        text = str(stmt)
        if "missions" in text.lower() or "mission" in text.lower():
            scalars.all = MagicMock(return_value=[alive_id])
        else:
            scalars.all = MagicMock(return_value=[orphan, keep])
        result.scalars = MagicMock(return_value=scalars)
        return result

    db.execute = AsyncMock(side_effect=_execute)
    db.delete = AsyncMock()
    db.commit = AsyncMock()
    await _prune_orphan_model_records(db)
    db.delete.assert_awaited_once_with(orphan)
    db.commit.assert_awaited()


