"""
Loss functions for HOPE.

IC loss: maximize rank correlation between predictions and targets.
MSE loss: standard mean squared error.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def ic_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Information Coefficient loss — maximize Pearson correlation.

    Falls back to MSE when predictions are near-constant,
    preventing the zero-gradient trap at initialization.

    Returns NEGATIVE IC so that minimizing = maximizing IC.
    """
    if pred.numel() < 4:
        return torch.tensor(0.0, device=pred.device, requires_grad=True)

    pred_std = pred.std()

    # Bootstrap: if predictions are constant, use MSE
    if pred_std < 1e-6:
        return F.mse_loss(pred, target)

    pred_z = (pred - pred.mean()) / (pred_std + 1e-8)
    target_z = (target - target.mean()) / (target.std() + 1e-8)
    ic = (pred_z * target_z).mean()
    return -ic  # minimize negative IC = maximize IC


def mse_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Standard MSE loss."""
    return F.mse_loss(pred, target)
