"""
Streaming LSTM baseline — matched parameter count to Mini-BAKA.

Also implements the streaming forward(x, state) → (pred, new_state)
interface so it plugs into the same StreamingTrainer.

~4.4K parameters with hidden_size=32, num_layers=1.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class MiniLSTM(nn.Module):
    """
    Streaming LSTM baseline.

    Uses the same forward(x, state) → (pred, new_state) interface as MiniBAKA.
    This allows fair comparison with the same StreamingTrainer.

    LSTM's hidden state IS a form of persistent memory.
    The question: is BAKA's multi-timescale memory BETTER than LSTM's?
    """

    def __init__(
        self,
        n_features: int = 1,
        hidden_size: int = 32,
        num_layers: int = 1,
        n_outputs: int = 1,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.head = nn.Linear(hidden_size, n_outputs)

    def init_state(self, batch_size: int = 1, device: torch.device = None) -> dict:
        """Initialize LSTM hidden and cell states to zeros."""
        device = device or torch.device("cpu")
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
        return {"h": h0, "c": c0}

    def forward(
        self, x: torch.Tensor, state: dict
    ) -> tuple[torch.Tensor, dict]:
        """
        Streaming forward pass.

        Args:
            x: [B, T, n_features]
            state: {"h": [layers, B, hidden], "c": [layers, B, hidden]}

        Returns:
            pred: [B, T] per-timestep predictions
            new_state: updated hidden/cell states
        """
        h_in = (state["h"], state["c"])
        out, (h_new, c_new) = self.lstm(x, h_in)  # out: [B, T, hidden]
        pred = self.head(out).squeeze(-1)           # [B, T]
        return pred, {"h": h_new, "c": c_new}

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    model = MiniLSTM()
    print(f"MiniLSTM params: {model.param_count():,}")

    state = model.init_state(batch_size=2)
    x = torch.randn(2, 16, 1)
    pred, new_state = model(x, state)
    print(f"Input: {x.shape} → Pred: {pred.shape}")
    print(f"Hidden norm: {new_state['h'].norm():.4f}")
