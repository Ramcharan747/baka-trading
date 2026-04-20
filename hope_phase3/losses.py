"""
Loss functions for Phase 3.

Primary: Sharpe loss — directly optimizes risk-adjusted return.
Secondary: IC loss — rank correlation (used for comparison).
"""
from __future__ import annotations

import torch


def sharpe_loss(pred: torch.Tensor, target: torch.Tensor,
                eps: float = 1e-6) -> torch.Tensor:
    """
    Directly optimizes risk-adjusted return (Sharpe ratio).

    pred:   [T] — model output (unbounded float)
    target: [T] — 15-bar-ahead net return (label)

    Uses tanh(pred*10) as position sizing:
    - When |pred| > 0.3, this approximates sign(pred) (long/short)
    - Near zero, position scales smoothly → no discontinuity in gradient
    - The *10 factor ensures gradient flows even for small predictions
    """
    positions = torch.tanh(pred * 10)
    pnl = positions * target          # realized PnL per bar
    mean_pnl = pnl.mean()
    std_pnl = pnl.std().clamp(min=eps)
    return -mean_pnl / std_pnl  # minimize negative Sharpe


def ic_loss(pred: torch.Tensor, target: torch.Tensor,
            eps: float = 1e-8) -> torch.Tensor:
    """Rank correlation loss — used for LightGBM comparison."""
    pred_z = (pred - pred.mean()) / (pred.std() + eps)
    tgt_z = (target - target.mean()) / (target.std() + eps)
    return -(pred_z * tgt_z).mean()
