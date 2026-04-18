"""
LSTM Baseline — same capacity as MiniHOPE for fair comparison.

Uses the same streaming training paradigm (TBPTT, never shuffle,
state persists across chunks) as HOPE.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class LSTMBaseline(nn.Module):
    """
    LSTM baseline with matched parameter count.

    For Phase 1: hidden_size=24, n_layers=2 gives ~5K params
    to match MiniHOPE(d_model=24, n_layers=1).
    """

    def __init__(self, n_features: int = 1, hidden_size: int = 24,
                 n_layers: int = 2, n_outputs: int = 1):
        super().__init__()
        self.hidden_size = hidden_size
        self.n_layers = n_layers

        if n_features != hidden_size:
            self.input_proj = nn.Linear(n_features, hidden_size)
        else:
            self.input_proj = nn.Identity()

        self.lstm = nn.LSTM(
            hidden_size, hidden_size, n_layers,
            batch_first=True, dropout=0.0,
        )

        self.head = nn.Linear(hidden_size, n_outputs)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, x: torch.Tensor,
                state=None) -> tuple[torch.Tensor, tuple]:
        """
        x: [batch, seq_len, n_features]
        state: (h, c) tuple or None
        """
        h = self.input_proj(x)
        out, new_state = self.lstm(h, state)
        pred = self.head(out)
        return pred, new_state

    def init_state(self, batch_size: int,
                   device: torch.device) -> None:
        """LSTM initializes to zero automatically when state=None."""
        return None
