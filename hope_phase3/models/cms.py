"""
Continuum Memory System (CMS) — multi-timescale persistent memory.

Implements Equation 71 and 97 from the Nested Learning paper.

CMS = chain of MLP blocks, each updating at a DIFFERENT frequency:
  y_t = MLP^(f_K)( MLP^(f_{K-1})( ... MLP^(f_1)(o_t) ... ))

Each MLP updates every C^(l) steps by accumulating gradients:
  θ^(f_l)_{i+1} = θ^(f_l)_i - Σ η_t ∇L(θ; x_t)    if i ≡ 0 (mod C^(l))
                = θ^(f_l)_i                          otherwise

For daily financial bars:
  Level 0: schedule=5   → updates every 5 bars (1 trading week)
  Level 1: schedule=21  → updates every 21 bars (1 trading month)
  Level 2: schedule=63  → updates every 63 bars (1 trading quarter)

Key difference from old CMS:
  - OLD: CMS gates were inside torch.no_grad() → dead params, NO GRADIENT
  - NEW: CMS MLPs participate in the OUTER autograd graph (Eq 71).
    Their gradients are accumulated, then applied on schedule.
    This means CMS params DO receive gradient through backprop.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class CMSBlock(nn.Module):
    """
    One level of the Continuum Memory System.

    Parameters update every `schedule` steps via accumulated gradients.
    This creates a "slow memory" that persists patterns longer than Titans.

    The MLP IS in the autograd graph — it generates output that flows into
    the loss. Its gradients are accumulated in a buffer and applied on schedule.

    Args:
        d_model:   Hidden dimension
        schedule:  Number of steps between parameter updates
        cms_lr:    Learning rate for parameter updates
    """

    def __init__(self, d_model: int, schedule: int, cms_lr: float):
        super().__init__()
        self.schedule = schedule
        self.cms_lr = cms_lr

        # 2-layer MLP (same architecture as MemoryModule, Eq 89)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.SiLU(),
            nn.Linear(d_model * 2, d_model),
        )
        # Initialize output layer to small nonzero (NOT zeros — zeros make
        # CMS a pure identity x+0=x with near-zero gradients)
        nn.init.normal_(self.mlp[2].weight, std=0.02)
        nn.init.zeros_(self.mlp[2].bias)

        # Gradient buffer: accumulates gradients between updates
        self._grad_buffer = {}
        self.steps_since_update = 0

    def _ensure_grad_buffer(self):
        """Lazily initialize gradient buffer (needs device info from params)."""
        if not self._grad_buffer:
            for name, param in self.mlp.named_parameters():
                self._grad_buffer[name] = torch.zeros_like(param)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [batch, seq_len, d_model]
        Returns: x + MLP(x) (residual connection)
        """
        h = self.mlp(x)
        return x + h

    def accumulate_gradients(self):
        """
        Called after loss.backward() — accumulate CMS gradients into buffer.

        This is the CMS update mechanism (Eq 71):
        - Gradients flow through the MLP via normal backprop
        - Instead of letting AdamW apply them immediately, we capture them
        - They accumulate until the schedule fires
        - Then we apply them manually and zero the buffer

        CRITICAL: We zero the param.grad after capturing, so the outer
        optimizer (AdamW) does NOT also update these params.
        """
        self._ensure_grad_buffer()
        for name, param in self.mlp.named_parameters():
            if param.grad is not None:
                self._grad_buffer[name] += param.grad.clone()
                param.grad.zero_()  # prevent outer optimizer from updating

    def maybe_update(self, global_step: int):
        """
        Apply accumulated gradients if schedule fires (Eq 71).

        θ_{i+1} = θ_i - Σ η_t · ∇L(θ_i; x_t)   if step ≡ 0 (mod C)
        """
        self._ensure_grad_buffer()
        self.steps_since_update += 1

        if self.steps_since_update >= self.schedule:
            with torch.no_grad():
                for name, param in self.mlp.named_parameters():
                    param.data -= self.cms_lr * self._grad_buffer[name]
                    self._grad_buffer[name].zero_()
            self.steps_since_update = 0

    def reset_buffer(self):
        """Reset gradient buffer (call at start of each epoch)."""
        self._grad_buffer = {}
        self.steps_since_update = 0
