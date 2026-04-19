"""
Parameter-matched LSTM baseline for Phase 3.
hidden_size=250, n_layers=4 → ~1.28M params (matches HOPE ~1.26M)
"""
from __future__ import annotations

import torch
import torch.nn as nn


class LSTMBaseline(nn.Module):
    def __init__(self, n_features: int, hidden_size: int = 250,
                 n_layers: int = 4, dropout: float = 0.1,
                 n_outputs: int = 1):
        super().__init__()
        self.hidden_size = hidden_size
        self.n_layers = n_layers

        self.input_proj = nn.Linear(n_features, hidden_size)
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )
        self.norm = nn.LayerNorm(hidden_size)
        self.output = nn.Linear(hidden_size, n_outputs)

        # Initialize output to near-zero
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, x, state=None):
        """
        x: [batch, T, n_features]
        state: (h, c) or None
        Returns: (pred [batch, T], new_state)
        """
        x = self.input_proj(x)
        out, state = self.lstm(x, state)
        out = self.norm(out)
        pred = self.output(out).squeeze(-1)  # [batch, T]
        return pred, state

    def init_state(self, batch_size: int, device):
        h = torch.zeros(self.n_layers, batch_size,
                        self.hidden_size, device=device)
        c = torch.zeros_like(h)
        return (h, c)
