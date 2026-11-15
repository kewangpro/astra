"""Dataset registry: ``dataset_id@rev`` → frozen assets under the data root."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

DEFAULT_DATA_ROOT = Path("data/astradata")


@dataclass(frozen=True)
class DatasetHandle:
    dataset_id: str
    rev: int
    root: Path
    content_hash: str
    meta: Dict[str, Any]

    @property
    def ref(self) -> str:
        return f"{self.dataset_id}@{self.rev}"

    def path(self, *parts: str) -> Path:
        return self.root.joinpath(*parts)


def parse_ref(ref: str) -> Tuple[str, int]:
    """Parse ``dataset_id@rev`` (rev defaults to 1 if omitted)."""
    text = (ref or "").strip()
    if not text:
        raise ValueError("empty dataset ref")
    if "@" in text:
        name, rev_s = text.rsplit("@", 1)
        return name.strip(), int(rev_s)
    return text, 1


def _rev_dir(data_root: Path, dataset_id: str, rev: int) -> Path:
    return Path(data_root) / dataset_id / str(rev)


def _hash_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _write_meta(rev_root: Path, meta: Dict[str, Any]) -> None:
    rev_root.mkdir(parents=True, exist_ok=True)
    (rev_root / "meta.json").write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")


def _read_meta(rev_root: Path) -> Dict[str, Any]:
    meta_path = rev_root / "meta.json"
    if not meta_path.is_file():
        return {}
    return json.loads(meta_path.read_text())


def resolve(
    dataset_id: str,
    rev: int = 1,
    *,
    data_root: Optional[Union[str, Path]] = None,
) -> DatasetHandle:
    """Load an already-frozen dataset revision (raises if missing)."""
    root = Path(data_root) if data_root is not None else DEFAULT_DATA_ROOT
    rev_root = _rev_dir(root, dataset_id, rev)
    meta = _read_meta(rev_root)
    if not meta.get("content_hash"):
        raise FileNotFoundError(
            f"dataset {dataset_id}@{rev} not frozen under {rev_root}; call ensure_dataset()"
        )
    return DatasetHandle(
        dataset_id=dataset_id,
        rev=rev,
        root=rev_root,
        content_hash=str(meta["content_hash"]),
        meta=meta,
    )


def ensure_dataset(
    dataset_id: str,
    rev: int = 1,
    *,
    data_root: Optional[Union[str, Path]] = None,
    force: bool = False,
    **kwargs: Any,
) -> DatasetHandle:
    """Ensure ``dataset_id@rev`` exists; build via the registered materializer."""
    root = Path(data_root) if data_root is not None else DEFAULT_DATA_ROOT
    rev_root = _rev_dir(root, dataset_id, rev)
    meta_path = rev_root / "meta.json"
    if meta_path.is_file() and not force:
        return resolve(dataset_id, rev, data_root=root)

    if dataset_id == "hstu_ml1m":
        from astradata.hstu_ml1m import materialize_hstu_ml1m

        meta = materialize_hstu_ml1m(rev_root, rev=rev, **kwargs)
    elif dataset_id == "routing_focus":
        # Product-neutral FailSlice bank placeholder (empty until Eval writes).
        rev_root.mkdir(parents=True, exist_ok=True)
        bank = rev_root / "bank"
        bank.mkdir(exist_ok=True)
        marker = rev_root / "EMPTY"
        marker.write_text("routing_focus FailSlice bank\n")
        content_hash = _hash_file(marker)
        meta = {
            "dataset_id": dataset_id,
            "rev": rev,
            "content_hash": content_hash,
            "kind": "fail_slice_bank",
        }
        _write_meta(rev_root, meta)
    else:
        raise KeyError(f"unknown dataset_id: {dataset_id!r}")

    _write_meta(rev_root, meta)
    return resolve(dataset_id, rev, data_root=root)
