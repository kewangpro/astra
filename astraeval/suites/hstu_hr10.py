"""Authoritative HR@10 suite for HSTU checkpoints."""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from astraeval.protocol import SuiteResult

# Frozen ceiling for suite_rev=1 (matches recipes/hstu_ml1m_v1.yaml metric_ceiling).
_CEILING_V1 = {"hr_at_10": 0.35}


class HstuHr10Suite:
    suite_id = "hstu_hr10"
    suite_rev = 1
    primary_metric = "hr_at_10"
    ceiling: Dict[str, float] = dict(_CEILING_V1)

    def __init__(self, suite_rev: int = 1) -> None:
        self.suite_rev = int(suite_rev)
        if self.suite_rev == 1:
            self.ceiling = dict(_CEILING_V1)

    def run(self, ckpt: Any, **kwargs: Any) -> SuiteResult:
        """Score an HSTU checkpoint.

        ``ckpt`` may be:
        - path to ``best.pt`` / ``last.pt``
        - dict already containing ``hr_at_10``
        - float primary (tests)
        """
        primary: Optional[float] = None
        diagnostics: Dict[str, Any] = {}

        if isinstance(ckpt, (int, float)):
            primary = float(ckpt)
        elif isinstance(ckpt, dict) and "hr_at_10" in ckpt:
            primary = float(ckpt["hr_at_10"])
            diagnostics = {k: ckpt[k] for k in ("ndcg_at_10", "n_items") if k in ckpt}
        else:
            path = Path(str(ckpt))
            if path.is_dir():
                for name in ("best.pt", "last.pt"):
                    cand = path / name
                    if cand.is_file():
                        path = cand
                        break
            if path.is_file() and path.suffix == ".pt":
                primary, diagnostics = self._score_pt(path, **kwargs)
            elif path.is_file():
                # Allow a tiny JSON sidecar for hermetic tests.
                import json

                data = json.loads(path.read_text())
                primary = float(data["hr_at_10"])
                diagnostics = {k: data[k] for k in data if k != "hr_at_10"}

        if primary is None or math.isnan(primary):
            raise ValueError(f"HstuHr10Suite could not score ckpt={ckpt!r}")

        # HSTU has no case_ids; empty FailSlice is still emitted for the bank link.
        return SuiteResult(
            primary=float(primary),
            primary_metric=self.primary_metric,
            case_ids=[],
            diagnostics=diagnostics,
            population_fingerprint=f"hstu_hr10@r{self.suite_rev}",
            suite_id=self.suite_id,
            suite_rev=self.suite_rev,
        )

    def _score_pt(self, path: Path, **kwargs: Any) -> Tuple[float, Dict[str, Any]]:
        import torch

        blob = torch.load(path, map_location="cpu", weights_only=False)
        if isinstance(blob, dict) and "hr_at_10" in blob:
            return float(blob["hr_at_10"]), {
                k: blob[k] for k in ("ndcg_at_10", "n_items") if k in blob
            }
        # Re-evaluate if weights present + dataset handle provided
        data_root = kwargs.get("dataset_root")
        if data_root is None or "model" not in blob:
            raise ValueError(f"checkpoint {path} has no hr_at_10 and no re-eval context")
        from backend.trainers.hstu_train import HSTUModel, NextItemDataset, evaluate
        from astradata.hstu_ml1m import load_frozen_sequences
        from torch.utils.data import DataLoader

        seqs, n_items, _item_map = load_frozen_sequences(Path(data_root))
        args = blob.get("args") or {}
        max_len = int(args.get("max_len", 50))
        model = HSTUModel(
            n_items=int(blob.get("n_items", n_items)),
            d_model=int(args.get("d_model", 64)),
            n_layers=int(args.get("n_layers", 2)),
            n_heads=int(args.get("n_heads", 2)),
            max_len=max_len,
            dropout=float(args.get("dropout", 0.1)),
        )
        model.load_state_dict(blob["model"])
        test_ds = NextItemDataset(seqs, max_len, "test")
        loader = DataLoader(test_ds, batch_size=256, shuffle=False)
        device = torch.device("cpu")
        model.to(device)
        metrics = evaluate(model, loader, device, ks=(10,))
        return float(metrics["hr_at_10"]), metrics
