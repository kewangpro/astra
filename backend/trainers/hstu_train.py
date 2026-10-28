"""
Minimal from-scratch HSTU trainer (LSTM-style loop).

Trains Hierarchical Sequential Transduction Units on MovieLens-1M for next-item
prediction. No Meta generative-recommenders checkout, gin, or multi-GPU stack —
just PyTorch, like training a small LSTM recommender.

Prints `HR@10: X.XXXX` / `NDCG@10: X.XXXX` for Astra's log scrapers.
"""
from __future__ import annotations

import argparse
import math
import os
import random
import urllib.request
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


# ── HSTU (paper eqs. 1–3, simplified) ─────────────────────────────────────────

class HSTUBlock(nn.Module):
    """One HSTU layer: pointwise split → SiLU attention + relative bias → gated FFN."""

    def __init__(self, d_model: int, n_heads: int, max_len: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.f1 = nn.Linear(d_model, 4 * d_model)
        self.f2 = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        # Relative position bias bucketed by distance in [-max_len+1, max_len-1]
        self.max_len = max_len
        self.rel_bias = nn.Parameter(torch.zeros(n_heads, 2 * max_len - 1))

    def _rel_pos_bias(self, seq_len: int, device: torch.device) -> torch.Tensor:
        # (L, L) relative offsets → (H, L, L)
        pos = torch.arange(seq_len, device=device)
        rel = pos[None, :] - pos[:, None]  # (L, L), in [-(L-1), L-1]
        rel = rel + (self.max_len - 1)
        rel = rel.clamp(0, 2 * self.max_len - 2)
        return self.rel_bias[:, rel]  # (H, L, L)

    def forward(self, x: torch.Tensor, pad_mask: torch.Tensor) -> torch.Tensor:
        # x: (B, L, D); pad_mask True where real tokens
        b, l, d = x.shape
        u, v, q, k = F.silu(self.f1(x)).chunk(4, dim=-1)

        def split_heads(t: torch.Tensor) -> torch.Tensor:
            return t.view(b, l, self.n_heads, self.d_head).transpose(1, 2)

        q, k, v, u_h = split_heads(q), split_heads(k), split_heads(v), split_heads(u)
        # SiLU attention (not softmax) — HSTU's distinguishing choice
        attn = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_head)
        attn = attn + self._rel_pos_bias(l, x.device)
        # Causal + padding: disallow attending to pad / future
        causal = torch.tril(torch.ones(l, l, device=x.device, dtype=torch.bool))
        allow = pad_mask[:, None, None, :] & causal[None, None, :, :]
        attn = attn.masked_fill(~allow, 0.0)
        attn = F.silu(attn)
        ctx = torch.matmul(attn, v)  # (B, H, L, Dh)
        gated = self.norm((ctx * u_h).transpose(1, 2).contiguous().view(b, l, d))
        return x + self.dropout(self.f2(gated))


class HSTUModel(nn.Module):
    def __init__(
        self,
        n_items: int,
        d_model: int = 64,
        n_layers: int = 2,
        n_heads: int = 2,
        max_len: int = 50,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.max_len = max_len
        self.item_emb = nn.Embedding(n_items + 1, d_model, padding_idx=0)
        self.blocks = nn.ModuleList(
            [HSTUBlock(d_model, n_heads, max_len, dropout) for _ in range(n_layers)]
        )
        self.out_norm = nn.LayerNorm(d_model)
        self.n_items = n_items

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        # seq: (B, L) item ids, 0 = pad
        pad_mask = seq != 0
        x = self.item_emb(seq)
        for block in self.blocks:
            x = block(x, pad_mask)
        return self.out_norm(x)  # (B, L, D)

    def logits_at_last(self, seq: torch.Tensor) -> torch.Tensor:
        h = self.forward(seq)
        # last non-pad position per row
        lengths = (seq != 0).sum(dim=1).clamp(min=1) - 1
        idx = lengths.view(-1, 1, 1).expand(-1, 1, h.size(-1))
        last = h.gather(1, idx).squeeze(1)  # (B, D)
        return last @ self.item_emb.weight[1:].t()  # (B, n_items) — skip pad row


# ── Data ──────────────────────────────────────────────────────────────────────

def _download_ml1m(data_dir: Path) -> Path:
    data_dir.mkdir(parents=True, exist_ok=True)
    ratings = data_dir / "ml-1m" / "ratings.dat"
    if ratings.is_file():
        return ratings
    zip_path = data_dir / "ml-1m.zip"
    url = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"
    print(f"Downloading MovieLens-1M from {url} …", flush=True)
    urllib.request.urlretrieve(url, zip_path)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(data_dir)
    zip_path.unlink(missing_ok=True)
    if not ratings.is_file():
        raise FileNotFoundError(f"expected {ratings} after extract")
    return ratings


def _synthetic_sequences(n_users: int = 200, n_items: int = 50, seed: int = 0) -> Dict[int, List[int]]:
    rng = random.Random(seed)
    seqs: Dict[int, List[int]] = {}
    for u in range(n_users):
        length = rng.randint(5, 20)
        # Prefer a small item subset so patterns are learnable
        fav = [rng.randint(1, n_items) for _ in range(3)]
        seq = [rng.choice(fav) if rng.random() < 0.7 else rng.randint(1, n_items) for _ in range(length)]
        seqs[u] = seq
    return seqs


def load_ml1m_sequences(data_dir: Path, min_len: int = 5) -> Tuple[Dict[int, List[int]], int]:
    ratings = _download_ml1m(data_dir)
    by_user: Dict[int, List[Tuple[int, int]]] = defaultdict(list)
    with open(ratings, encoding="latin-1") as f:
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
    # Remap to contiguous 1..N (0 = pad)
    item_map = {old: i + 1 for i, old in enumerate(sorted(item_ids))}
    seqs = {u: [item_map[i] for i in items] for u, items in raw.items()}
    return seqs, len(item_map)


class NextItemDataset(Dataset):
    """Leave-one-out style: each prefix predicts the next item (train) or holdout."""

    def __init__(self, sequences: Dict[int, List[int]], max_len: int, mode: str):
        self.samples: List[Tuple[List[int], int]] = []
        self.max_len = max_len
        for items in sequences.values():
            if len(items) < 2:
                continue
            if mode == "train":
                # All next-item steps except last transition (held for test)
                for t in range(1, len(items) - 1):
                    hist = items[:t][-max_len:]
                    self.samples.append((hist, items[t]))
            else:
                hist = items[:-1][-max_len:]
                self.samples.append((hist, items[-1]))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        hist, target = self.samples[idx]
        pad_len = self.max_len - len(hist)
        seq = [0] * pad_len + hist
        return torch.tensor(seq, dtype=torch.long), torch.tensor(target - 1, dtype=torch.long)


# ── Train / eval ──────────────────────────────────────────────────────────────

def _device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@torch.no_grad()
def evaluate(model: HSTUModel, loader: DataLoader, device: torch.device, ks=(10,)) -> dict:
    model.eval()
    hits = {k: 0 for k in ks}
    ndcg = {k: 0 for k in ks}
    n = 0
    for seq, target in loader:
        seq, target = seq.to(device), target.to(device)
        logits = model.logits_at_last(seq)  # (B, n_items)
        # Rank descending
        ranks = logits.argsort(dim=-1, descending=True)
        for i in range(target.size(0)):
            tgt = int(target[i].item())
            rank_list = ranks[i].tolist()
            try:
                rank = rank_list.index(tgt)  # 0-based
            except ValueError:
                rank = model.n_items
            n += 1
            for k in ks:
                if rank < k:
                    hits[k] += 1
                    ndcg[k] += 1.0 / math.log2(rank + 2)
    out = {}
    for k in ks:
        out[f"hr_at_{k}"] = hits[k] / max(n, 1)
        out[f"ndcg_at_{k}"] = ndcg[k] / max(n, 1)
    return out


def train(args: argparse.Namespace) -> None:
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    data_dir = Path(args.data_dir)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = _device()
    print(f"device={device}", flush=True)

    if args.synthetic:
        seqs = _synthetic_sequences(n_users=args.synthetic_users, n_items=args.synthetic_items)
        n_items = args.synthetic_items
    else:
        seqs, n_items = load_ml1m_sequences(data_dir)
    print(f"users={len(seqs)} items={n_items}", flush=True)

    train_ds = NextItemDataset(seqs, args.max_len, "train")
    test_ds = NextItemDataset(seqs, args.max_len, "test")
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = HSTUModel(
        n_items=n_items,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        max_len=args.max_len,
        dropout=args.dropout,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_hr = -1.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss, steps = 0.0, 0
        for seq, target in train_loader:
            seq, target = seq.to(device), target.to(device)
            logits = model.logits_at_last(seq)
            loss = F.cross_entropy(logits, target)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total_loss += float(loss.item())
            steps += 1
        avg_loss = total_loss / max(steps, 1)
        metrics = evaluate(model, test_loader, device, ks=(10,))
        hr, ndcg = metrics["hr_at_10"], metrics["ndcg_at_10"]
        print(
            f"epoch {epoch}/{args.epochs} loss={avg_loss:.4f} "
            f"HR@10: {hr:.4f} NDCG@10: {ndcg:.4f}",
            flush=True,
        )
        ckpt = {
            "model": model.state_dict(),
            "n_items": n_items,
            "args": vars(args),
            "hr_at_10": hr,
            "ndcg_at_10": ndcg,
        }
        torch.save(ckpt, out_dir / "last.pt")
        if hr > best_hr:
            best_hr = hr
            torch.save(ckpt, out_dir / "best.pt")
            print(f"saved best checkpoint hr_at_10={hr:.4f}", flush=True)

    print(f"HR@10: {best_hr:.4f}", flush=True)
    print(f"done best_hr_at_10={best_hr:.4f}", flush=True)


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train a small HSTU next-item recommender")
    p.add_argument("--data-dir", default="data/hstu", help="MovieLens cache directory")
    p.add_argument("--out", required=True, help="Checkpoint directory")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-2)
    p.add_argument("--d-model", type=int, default=64)
    p.add_argument("--n-layers", type=int, default=2)
    p.add_argument("--n-heads", type=int, default=2)
    p.add_argument("--max-len", type=int, default=50)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--synthetic", action="store_true", help="Skip MovieLens; tiny toy data")
    p.add_argument("--synthetic-users", type=int, default=200)
    p.add_argument("--synthetic-items", type=int, default=50)
    return p


def main(argv: Optional[List[str]] = None) -> None:
    args = build_argparser().parse_args(argv)
    train(args)


if __name__ == "__main__":
    main()
