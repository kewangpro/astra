"""Canonical Actor-Critic network for Tetris-v0.

Defined here so torch.load can resolve the class regardless of which
process loads the checkpoint (train.py, uvicorn play endpoint, benchmark).
"""
from __future__ import annotations

import torch.nn as nn


class ActorCriticNet(nn.Module):
    """Shared MLP [input_dim→64→64] + scalar critic head."""

    def __init__(self, input_dim: int = 4) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.shared = nn.Sequential(
            nn.Linear(input_dim, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
        )
        self.critic = nn.Linear(64, 1)
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)

    def forward(self, x):
        return self.critic(self.shared(x))


class Game2048ValueNet(nn.Module):
    """Deep MLP [16→128→128→64→1] for Game2048 lookahead state evaluation."""

    def __init__(self, input_dim: int = 16) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.shared = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
        )
        self.critic = nn.Linear(64, 1)
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)

    def forward(self, x):
        return self.critic(self.shared(x))

