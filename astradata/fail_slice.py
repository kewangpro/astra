"""FailSlice schema + on-disk bank (Eval → Data feedback)."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

DEFAULT_BANK_ROOT = Path("data/astradata/fail_slices")


@dataclass
class FailSlice:
    suite_id: str
    suite_rev: int
    case_ids: List[str]
    primary: float
    primary_metric: str = "pass_rate"
    mission_id: Optional[str] = None
    dataset_ref: Optional[str] = None
    population_fingerprint: Optional[str] = None
    diagnostics: Dict[str, Any] = field(default_factory=dict)
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FailSlice":
        return cls(
            suite_id=str(data["suite_id"]),
            suite_rev=int(data["suite_rev"]),
            case_ids=[str(x) for x in data.get("case_ids") or []],
            primary=float(data.get("primary", 0.0)),
            primary_metric=str(data.get("primary_metric") or "pass_rate"),
            mission_id=data.get("mission_id"),
            dataset_ref=data.get("dataset_ref"),
            population_fingerprint=data.get("population_fingerprint"),
            diagnostics=dict(data.get("diagnostics") or {}),
            created_at=str(data.get("created_at") or datetime.now(timezone.utc).isoformat()),
        )


class FailSliceStore:
    """Append-only JSON FailSlice bank under ``bank_root``."""

    def __init__(self, bank_root: Optional[Union[str, Path]] = None) -> None:
        self.bank_root = Path(bank_root) if bank_root is not None else DEFAULT_BANK_ROOT
        self.bank_root.mkdir(parents=True, exist_ok=True)

    def _path_for(self, slice_: FailSlice, index: int) -> Path:
        safe_suite = slice_.suite_id.replace("/", "_")
        mid = (slice_.mission_id or "anon")[:12]
        return self.bank_root / f"{safe_suite}_r{slice_.suite_rev}_{mid}_{index:04d}.json"

    def write(self, slice_: FailSlice) -> Path:
        existing = list(self.bank_root.glob("*.json"))
        path = self._path_for(slice_, len(existing))
        path.write_text(json.dumps(slice_.to_dict(), indent=2, sort_keys=True) + "\n")
        latest = self.bank_root / "latest.json"
        latest.write_text(json.dumps(slice_.to_dict(), indent=2, sort_keys=True) + "\n")
        if slice_.mission_id:
            by_mission = self.bank_root / f"mission_{slice_.mission_id[:12]}.json"
            by_mission.write_text(json.dumps(slice_.to_dict(), indent=2, sort_keys=True) + "\n")
        return path

    def read(self, path: Union[str, Path]) -> FailSlice:
        return FailSlice.from_dict(json.loads(Path(path).read_text()))

    def latest(self, suite_id: Optional[str] = None) -> Optional[FailSlice]:
        latest_path = self.bank_root / "latest.json"
        if not latest_path.is_file():
            return None
        slice_ = self.read(latest_path)
        if suite_id and slice_.suite_id != suite_id:
            # Scan newest matching suite
            matches = sorted(
                self.bank_root.glob(f"{suite_id.replace('/', '_')}_*.json"),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )
            for p in matches:
                if p.name == "latest.json":
                    continue
                return self.read(p)
            return None
        return slice_

    def focus_cases_csv(self, suite_id: Optional[str] = None) -> str:
        """CSV suitable for recipe ``focus_cases`` / ``--focus-cases``."""
        slice_ = self.latest(suite_id=suite_id)
        if slice_ is None or not slice_.case_ids:
            return ""
        return ",".join(slice_.case_ids)
