"""AstraData — versioned fuel for the flywheel (dataset_id@rev + FailSlice bank)."""
from __future__ import annotations

from astradata.fail_slice import FailSlice, FailSliceStore
from astradata.registry import DatasetHandle, ensure_dataset, parse_ref, resolve

__all__ = [
    "DatasetHandle",
    "FailSlice",
    "FailSliceStore",
    "ensure_dataset",
    "parse_ref",
    "resolve",
]
