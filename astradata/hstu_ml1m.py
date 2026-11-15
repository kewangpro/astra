"""MovieLens-1M materializer for ``hstu_ml1m@rev``."""
from __future__ import annotations

import hashlib
import json
import urllib.request
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _download_ml1m(cache_dir: Path) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    ratings = cache_dir / "ml-1m" / "ratings.dat"
    if ratings.is_file():
        return ratings
    zip_path = cache_dir / "ml-1m.zip"
    url = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"
    urllib.request.urlretrieve(url, zip_path)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(cache_dir)
    zip_path.unlink(missing_ok=True)
    if not ratings.is_file():
        raise FileNotFoundError(f"expected {ratings} after extract")
    return ratings


def build_sequences(
    ratings_path: Path, min_len: int = 5
) -> Tuple[Dict[int, List[int]], Dict[int, int]]:
    by_user: Dict[int, List[Tuple[int, int]]] = defaultdict(list)
    with open(ratings_path, encoding="latin-1") as f:
        for line in f:
            uid, iid, _rating, ts = line.strip().split("::")
            by_user[int(uid)].append((int(ts), int(iid)))
    raw: Dict[int, List[int]] = {}
    item_ids: set[int] = set()
    for uid, events in by_user.items():
        events.sort()
        items = [iid for _, iid in events]
        if len(items) < min_len:
            continue
        raw[uid] = items
        item_ids.update(items)
    item_map = {old: i + 1 for i, old in enumerate(sorted(item_ids))}
    seqs = {u: [item_map[i] for i in items] for u, items in raw.items()}
    return seqs, item_map


def _hash_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def materialize_hstu_ml1m(
    rev_root: Path,
    *,
    rev: int,
    min_len: int = 5,
    download_cache: Optional[Path] = None,
) -> Dict[str, Any]:
    """Freeze ML-1M sequences + item_map under ``rev_root``."""
    rev_root.mkdir(parents=True, exist_ok=True)
    cache = download_cache or (rev_root / "_download")
    ratings = _download_ml1m(cache)
    seqs, item_map = build_sequences(ratings, min_len=min_len)

    item_map_path = rev_root / "item_map.json"
    # Store as str keys for JSON; values are contiguous 1..N ids.
    item_map_json = {str(k): v for k, v in item_map.items()}
    item_map_bytes = (json.dumps(item_map_json, sort_keys=True) + "\n").encode()
    item_map_path.write_bytes(item_map_bytes)

    seqs_path = rev_root / "sequences.json"
    seqs_json = {str(k): v for k, v in seqs.items()}
    seqs_bytes = (json.dumps(seqs_json, sort_keys=True) + "\n").encode()
    seqs_path.write_bytes(seqs_bytes)

    content_hash = _hash_bytes(item_map_bytes + seqs_bytes)
    return {
        "dataset_id": "hstu_ml1m",
        "rev": rev,
        "content_hash": content_hash,
        "n_users": len(seqs),
        "n_items": len(item_map),
        "min_len": min_len,
        "item_map": "item_map.json",
        "sequences": "sequences.json",
    }


def load_frozen_sequences(handle_root: Path) -> Tuple[Dict[int, List[int]], int, Dict[int, int]]:
    """Load sequences + item_map from a frozen rev directory."""
    seqs_raw = json.loads((handle_root / "sequences.json").read_text())
    item_map_raw = json.loads((handle_root / "item_map.json").read_text())
    seqs = {int(k): list(v) for k, v in seqs_raw.items()}
    item_map = {int(k): int(v) for k, v in item_map_raw.items()}
    return seqs, len(item_map), item_map
