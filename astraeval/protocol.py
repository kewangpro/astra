"""Suite protocol: ``suite.run(ckpt, suite_id) → primary + case_ids``."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable


@dataclass
class SuiteResult:
    primary: float
    primary_metric: str
    case_ids: List[str] = field(default_factory=list)
    diagnostics: Dict[str, Any] = field(default_factory=dict)
    population_fingerprint: Optional[str] = None
    suite_id: str = ""
    suite_rev: int = 1

    @property
    def failed_case_ids(self) -> List[str]:
        """Alias used when emitting FailSlice (failed or focus-worthy cases)."""
        return list(self.case_ids)


@runtime_checkable
class Suite(Protocol):
    suite_id: str
    suite_rev: int
    primary_metric: str
    ceiling: Dict[str, float]

    def run(self, ckpt: Any, **kwargs: Any) -> SuiteResult:
        """Score checkpoint; return primary + optional fail/focus case_ids."""
        ...
