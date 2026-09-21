"""Unit tests for recipe search-index orphan prune."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

from backend.services.recipe_library import prune_orphan_index


def test_prune_orphan_index_deletes_ids_not_in_db():
    col = MagicMock()
    col.get.return_value = {"ids": ["alive", "ghost-train-rl", "ghost-ml"]}
    with patch("backend.services.recipe_library._get_collection", return_value=col):
        n = prune_orphan_index({"alive"})
    assert n == 2
    col.delete.assert_called_once()
    stale = set(col.delete.call_args.kwargs["ids"])
    assert stale == {"ghost-train-rl", "ghost-ml"}


def test_prune_orphan_index_noop_when_in_sync():
    col = MagicMock()
    col.get.return_value = {"ids": ["a", "b"]}
    with patch("backend.services.recipe_library._get_collection", return_value=col):
        n = prune_orphan_index({"a", "b"})
    assert n == 0
    col.delete.assert_not_called()


def test_prune_orphan_index_clears_all_when_db_empty():
    col = MagicMock()
    col.get.return_value = {"ids": ["train_rl_v1", "train_ml_v1"]}
    with patch("backend.services.recipe_library._get_collection", return_value=col):
        n = prune_orphan_index(set())
    assert n == 2
    col.delete.assert_called_once()
