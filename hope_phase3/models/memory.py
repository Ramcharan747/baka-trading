"""
Memory modules for HOPE — the building blocks shared across all components.

Implements:
  1. MemoryModule: 2-layer MLP with residual (Eq 89/91 from paper)
     M_□(x) = x + W_1 σ(W_2 x)

This is the architecture for ALL six memory modules in Self-Referential Titans.
Each memory is both a projection function AND an associative memory that
generates its own training targets (self-referential).
"""
from __future__ import annotations

import torch
import torch.nn as nn


class MemoryModule(nn.Module):
    """
    2-layer MLP with residual connection (Eq 89 from paper).

    M_□(x) = x + W_1 σ(W_2 x)

    Used for:
      - M_k: adaptive key projection
      - M_v: adaptive value projection
      - M_η: adaptive learning rate prediction
      - M_α: adaptive weight decay prediction
      - M_mem: main memory retrieval (when used as MLP form)

    The output layer is initialized to near-zero to prevent
    cold-start problems where the residual path dominates.

    Args:
        d_model:  Input dimension
        out_dim:  Output dimension (defaults to d_model for residual)
    """

    def __init__(self, d_model: int, out_dim: int | None = None):
        super().__init__()
        out_dim = out_dim or d_model
        self.fc1 = nn.Linear(d_model, d_model, bias=True)
        self.fc2 = nn.Linear(d_model, out_dim, bias=True)
        self.act = nn.SiLU()
        self.residual = out_dim == d_model

        # Initialize output layer to near-zero so that at init:
        #   M(x) ≈ x  (residual path dominates)
        # This is critical — without it, random projections dominate
        # before the memory has learned anything useful.
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [..., d_model]
        returns: [..., out_dim]
        """
        h = self.act(self.fc1(x))
        out = self.fc2(h)
        if self.residual:
            return x + out
        return out
