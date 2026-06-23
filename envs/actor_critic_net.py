"""Canonical Actor-Critic network for Tetris-v0.

Defined here so torch.load can resolve the class regardless of which
process loads the checkpoint (train.py, uvicorn play endpoint, benchmark).
"""
from __future__ import annotations

import torch.nn as nn


class ActorCriticNet(nn.Module):
    """Shared MLP [4→64→64] + scalar critic head."""

    def __init__(self) -> None:
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(4, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
        )
        self.critic = nn.Linear(64, 1)
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)

    def forward(self, x):
        return self.critic(self.shared(x))
