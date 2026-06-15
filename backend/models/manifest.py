"""
RequirementManifest — Step 7.2.

Structured list of granular, checkable requirements for a mission.
Stored as JSON at data/missions/{id}/requirements.json.
A mission may only be marked COMPLETED when every requirement has passed=True.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Optional, List


@dataclass
class Requirement:
    id: str
    description: str
    category: str        # "performance" | "artifact" | "stability"
    check_type: str      # "metric_threshold" | "file_exists" | "no_sandbox_error"

    # metric_threshold fields
    metric_name: Optional[str] = None
    threshold: Optional[float] = None
    operator: str = ">="   # ">=" | "<=" | ">" | "<"

    # file_exists fields
    path_pattern: Optional[str] = None   # glob relative to mission dir

    # state
    passed: bool = False
    passed_at: Optional[str] = None
    evidence: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Requirement":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class RequirementManifest:
    mission_id: str
    generated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    version: str = "1.0"
    requirements: List[Requirement] = field(default_factory=list)

    def is_complete(self) -> bool:
        return bool(self.requirements) and all(r.passed for r in self.requirements)

    def passed_count(self) -> int:
        return sum(1 for r in self.requirements if r.passed)

    def summary(self) -> dict:
        total = len(self.requirements)
        passed = self.passed_count()
        return {
            "total": total,
            "passed": passed,
            "pending": total - passed,
            "complete": self.is_complete(),
        }

    def to_dict(self) -> dict:
        return {
            "version": self.version,
            "mission_id": self.mission_id,
            "generated_at": self.generated_at,
            "requirements": [r.to_dict() for r in self.requirements],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "RequirementManifest":
        reqs = [Requirement.from_dict(r) for r in d.get("requirements", [])]
        return cls(
            mission_id=d["mission_id"],
            generated_at=d.get("generated_at", ""),
            version=d.get("version", "1.0"),
            requirements=reqs,
        )

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "RequirementManifest":
        with open(path, "r") as f:
            return cls.from_dict(json.load(f))
