"""PyTorch models. Selected with ``model.type`` in the config.

* ``mlp`` — RideClassifier from research/train_neural_network.py (team's first
  model): 30 aggregate features -> 64 -> 32 -> 5, ReLU, dropout 0.2.
* ``cnn`` — 1D-CNN on the resampled sequence (seq_len, 6) -> 5.

Both are small on purpose (see ai-infrastructure/distributed-training/).
"""
from __future__ import annotations

import torch
from torch import nn


class RideMLP(nn.Module):
    def __init__(self, n_features: int, n_classes: int, hidden: int = 64, dropout: float = 0.2):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(n_features, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2), nn.ReLU(),
            nn.Linear(hidden // 2, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class RideCNN(nn.Module):
    def __init__(self, n_features: int, n_classes: int, hidden: int = 64, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(n_features, hidden, kernel_size=7, stride=2, padding=3), nn.BatchNorm1d(hidden), nn.ReLU(),
            nn.Conv1d(hidden, hidden * 2, kernel_size=5, padding=2), nn.BatchNorm1d(hidden * 2), nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(hidden * 2, hidden * 2, kernel_size=3, padding=1), nn.BatchNorm1d(hidden * 2), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Sequential(nn.Flatten(), nn.Dropout(dropout), nn.Linear(hidden * 2, n_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.net(x.transpose(1, 2)))  # (B, T, F) -> (B, F, T)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def build_model(cfg: dict, input_shape: tuple[int, ...]) -> nn.Module:
    m = cfg["model"]
    n_classes = len(cfg["project"]["labels"])
    if m["type"] == "mlp":
        return RideMLP(input_shape[-1], n_classes, m["hidden"], m["dropout"])
    if m["type"] == "cnn":
        return RideCNN(input_shape[-1], n_classes, m["hidden"], m["dropout"])
    raise ValueError(f"unknown model.type {m['type']!r}")
