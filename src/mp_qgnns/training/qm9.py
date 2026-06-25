"""Training and evaluation for the QM9 regression model."""
from __future__ import annotations

import torch


def train(model, batch, targets, *, epochs: int = 300, lr: float = 1e-3):
    """Fit with L1 loss. ``batch`` is (adj, atom_types, bond_types, node_mask)."""
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    mean, std = targets.mean(), targets.std() + 1e-9
    for _ in range(epochs):
        opt.zero_grad()
        pred = model(*batch)
        loss = torch.nn.functional.l1_loss(pred, (targets - mean) / std)
        loss.backward()
        opt.step()
    model._target_mean, model._target_std = mean, std
    return model


def mae(model, batch, targets) -> float:
    """Mean absolute error in the target units."""
    model.eval()
    mean = getattr(model, "_target_mean", 0.0)
    std = getattr(model, "_target_std", 1.0)
    with torch.no_grad():
        pred = model(*batch) * std + mean
    return float((pred - targets).abs().mean())
