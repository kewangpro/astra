"""Split / leak helpers for versioned datasets."""
from __future__ import annotations

from typing import Dict, Iterable, List, Sequence, Set, TypeVar

T = TypeVar("T")


def assert_disjoint(*groups: Iterable[T], label: str = "split") -> None:
    """Raise if any two groups share an element."""
    seen: Set[T] = set()
    for i, group in enumerate(groups):
        gset = set(group)
        overlap = seen & gset
        if overlap:
            sample = list(overlap)[:5]
            raise ValueError(f"{label} leak between group {i} and prior: {sample!r}…")
        seen |= gset


def leave_one_out_user_split(
    sequences: Dict[int, Sequence[int]],
) -> Dict[str, Dict[int, List[int]]]:
    """Train = all-but-last transition; test = final item (HSTU-style).

    Returns dict with ``train`` / ``test`` mapping user → full sequence
    (callers slice prefixes themselves).
    """
    train: Dict[int, List[int]] = {}
    test: Dict[int, List[int]] = {}
    for uid, items in sequences.items():
        seq = list(items)
        if len(seq) < 2:
            continue
        train[uid] = seq
        test[uid] = seq
    return {"train": train, "test": test}
