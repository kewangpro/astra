"""Routing bare_eval suite — product oracle + frozen clock + population fingerprint."""
from __future__ import annotations

import hashlib
import re
import subprocess
from typing import Any, Dict, List, Optional, Tuple

from astraeval.protocol import SuiteResult

_PASS_RATE_RE = re.compile(
    r"Pass rate:\s*([0-9.]+)%\s*\((\d+)/(\d+)\)", re.IGNORECASE
)
_CASE_FAIL_RE = re.compile(r"FAIL\s+(\S+)|case[_\s-]?id[=:]\s*(\S+)", re.IGNORECASE)

# Suite-rev 1 ceiling — Ensemble routing empirical; products may override via kwargs.
_CEILING_V1 = {"pass_rate": 0.96}


class RoutingBareEvalSuite:
    suite_id = "routing_bare"
    suite_rev = 1
    primary_metric = "pass_rate"
    ceiling: Dict[str, float] = dict(_CEILING_V1)

    def __init__(self, suite_rev: int = 1, ceiling: Optional[Dict[str, float]] = None) -> None:
        self.suite_rev = int(suite_rev)
        self.ceiling = dict(ceiling or _CEILING_V1)

    def run(self, ckpt: Any, **kwargs: Any) -> SuiteResult:
        """Score via product ``bare_eval`` or a pre-parsed report dict.

        kwargs:
          - ``bare_eval_cmd``: full shell command (SSH or local)
          - ``report``: dict with primary / case_ids / fingerprint (tests)
          - ``ssh_host``: optional host for ``ssh host cmd``
          - ``frozen_clock``: ISO timestamp string baked into fingerprint
          - ``case_population``: iterable of case ids for fingerprint
        """
        report = kwargs.get("report")
        if isinstance(report, dict):
            return self._from_report(report)

        if isinstance(ckpt, dict) and "primary" in ckpt:
            return self._from_report(ckpt)

        cmd = kwargs.get("bare_eval_cmd")
        if not cmd:
            raise ValueError("RoutingBareEvalSuite requires bare_eval_cmd or report=")

        ssh_host = kwargs.get("ssh_host") or ""
        try:
            if ssh_host:
                result = subprocess.run(
                    ["ssh", ssh_host, cmd],
                    capture_output=True,
                    text=True,
                    timeout=int(kwargs.get("timeout", 5400)),
                )
            else:
                result = subprocess.run(
                    cmd,
                    shell=True,
                    capture_output=True,
                    text=True,
                    timeout=int(kwargs.get("timeout", 5400)),
                )
        except Exception as exc:
            raise RuntimeError(f"RoutingBareEvalSuite bare_eval failed: {exc}") from exc

        stdout = result.stdout or ""
        primary, n_ok, n_tot = self._parse_pass_rate(stdout)
        case_ids = self._parse_fail_cases(stdout)
        population = list(kwargs.get("case_population") or [])
        frozen_clock = str(kwargs.get("frozen_clock") or f"suite_rev={self.suite_rev}")
        fingerprint = self._fingerprint(population or case_ids, frozen_clock)

        return SuiteResult(
            primary=primary,
            primary_metric=self.primary_metric,
            case_ids=case_ids,
            diagnostics={
                "n_ok": n_ok,
                "n_total": n_tot,
                "returncode": result.returncode,
                "stdout_tail": stdout[-500:],
            },
            population_fingerprint=fingerprint,
            suite_id=self.suite_id,
            suite_rev=self.suite_rev,
        )

    def _from_report(self, report: Dict[str, Any]) -> SuiteResult:
        case_ids = [str(x) for x in report.get("case_ids") or []]
        frozen_clock = str(report.get("frozen_clock") or f"suite_rev={self.suite_rev}")
        population = list(report.get("case_population") or case_ids)
        fingerprint = report.get("population_fingerprint") or self._fingerprint(
            population, frozen_clock
        )
        return SuiteResult(
            primary=float(report["primary"]),
            primary_metric=str(report.get("primary_metric") or self.primary_metric),
            case_ids=case_ids,
            diagnostics=dict(report.get("diagnostics") or {}),
            population_fingerprint=fingerprint,
            suite_id=self.suite_id,
            suite_rev=self.suite_rev,
        )

    @staticmethod
    def _parse_pass_rate(text: str) -> Tuple[float, Optional[int], Optional[int]]:
        best = None
        n_ok = n_tot = None
        for m in _PASS_RATE_RE.finditer(text):
            pct = float(m.group(1)) / 100.0
            n_ok, n_tot = int(m.group(2)), int(m.group(3))
            if best is None or pct > best:
                best = pct
        if best is None:
            raise ValueError("no parseable Pass rate in bare_eval output")
        return best, n_ok, n_tot

    @staticmethod
    def _parse_fail_cases(text: str) -> List[str]:
        ids: List[str] = []
        for m in _CASE_FAIL_RE.finditer(text):
            cid = m.group(1) or m.group(2)
            if cid and cid not in ids:
                ids.append(cid.rstrip(",;"))
        return ids

    @staticmethod
    def _fingerprint(case_ids: List[str], frozen_clock: str) -> str:
        blob = frozen_clock + "\n" + "\n".join(sorted(case_ids))
        return hashlib.sha256(blob.encode()).hexdigest()[:16]
