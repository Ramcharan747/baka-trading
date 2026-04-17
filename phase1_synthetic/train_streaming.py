"""
Streaming training utilities — TBPTT, state management, loss functions.

This is the CORRECT training paradigm for BAKA:
1. Feed data as one long continuous sequence
2. Never reset model state between consecutive chunks
3. Truncated BPTT: backprop within each chunk, detach at boundaries
4. W_current / CMS buffers flow forward continuously — only gradient tape is cut
"""
from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn as nn


# ============================================================ state utils


def detach_state(state: Any) -> Any:
    """
    Detach the gradient tape from state values, but KEEP the values.

    This is the key to TBPTT:
    - Memory content (W_current, CMS summaries) continues to hold learned values
    - Only the gradient graph is cut — preventing memory explosion from
      backpropagating through the entire sequence
    - This is NOT the same as resetting state to zeros
    """
    if isinstance(state, dict):
        return {k: detach_state(v) for k, v in state.items()}
    elif isinstance(state, (list, tuple)):
        detached = [detach_state(v) for v in state]
        return type(state)(detached)
    elif isinstance(state, torch.Tensor):
        return state.detach()
    return state


# ============================================================ losses


def ic_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Differentiable IC (information coefficient) loss.

    Maximizes Pearson correlation between predictions and targets.
    Uses z-scoring for scale invariance.

    Returns NEGATIVE IC so that minimizing this = maximizing IC.
    """
    # Need enough samples for meaningful correlation
    if pred.numel() < 4:
        return torch.tensor(0.0, device=pred.device, requires_grad=True)

    p = pred - pred.mean()
    t = target - target.mean()
    ic = (p * t).mean() / (p.std() * t.std() + 1e-8)
    return -ic  # minimize negative IC = maximize IC


def mse_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Standard MSE loss — simpler alternative to IC for synthetic data."""
    return ((pred - target) ** 2).mean()


# ============================================================ streaming trainer


class StreamingTrainer:
    """
    TBPTT (Truncated Backpropagation Through Time) trainer.

    Feeds data as one continuous sequence. State flows between chunks.
    Gradient is cut at chunk boundaries. This is the CORRECT training
    for any model with persistent memory (BAKA, stateful LSTM).

    Key difference from the previous WRONG approach:
    - Old: create sliding windows → treat as i.i.d. samples → reset state each
    - New: feed entire sequence → state persists → cut gradient at chunk boundary
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        chunk_size: int = 64,
        loss_fn: str = "mse",
        device: str = "cpu",
        log_every: int = 500,
    ):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.chunk_size = chunk_size
        self.loss_fn = mse_loss if loss_fn == "mse" else ic_loss
        self.device = torch.device(device)
        self.log_every = log_every
        self.state = None  # persists across chunks within an epoch

    def train_epoch(
        self,
        x: np.ndarray,
        y: np.ndarray,
        reset_state: bool = True,
    ) -> float:
        """
        Train on the FULL sequence as one continuous pass.

        Args:
            x: [T] or [T, F] input features
            y: [T] prediction targets
            reset_state: if True, initialize state to zeros at epoch start.
                         Set False to warm-start from previous epoch.
        """
        self.model.train()

        # Convert to tensors
        x_t = torch.from_numpy(x.astype(np.float32)).to(self.device)
        y_t = torch.from_numpy(y.astype(np.float32)).to(self.device)

        # Add feature dimension if 1D
        if x_t.dim() == 1:
            x_t = x_t.unsqueeze(-1)  # [T, 1]

        T = len(x_t)
        chunk = self.chunk_size
        n_chunks = T // chunk

        # Initialize state ONCE per epoch
        if reset_state or self.state is None:
            self.state = self.model.init_state(batch_size=1, device=self.device)

        total_loss = 0.0
        for i in range(n_chunks):
            s = i * chunk
            e = s + chunk

            x_chunk = x_t[s:e].unsqueeze(0)  # [1, chunk, F]
            y_chunk = y_t[s:e].unsqueeze(0)  # [1, chunk]

            # Forward: state flows continuously from previous chunk
            pred, self.state = self.model(x_chunk, self.state)

            # TBPTT: detach state for next chunk
            # This cuts the gradient graph but PRESERVES memory values
            self.state = detach_state(self.state)

            # Loss on this chunk
            loss = self.loss_fn(pred.squeeze(), y_chunk.squeeze())

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += loss.item()

            if (i + 1) % self.log_every == 0:
                avg = total_loss / (i + 1)
                print(f"  chunk {i+1}/{n_chunks}  avg_loss={avg:.6f}")

        return total_loss / max(n_chunks, 1)

    @torch.no_grad()
    def evaluate(
        self,
        x: np.ndarray,
        y: np.ndarray,
        reset_state: bool = False,
    ) -> dict:
        """
        Evaluate on held-out data.

        Args:
            reset_state: False = state continues from training (production behavior)
                         True  = state resets (to measure if memory helps)
        """
        self.model.eval()

        x_t = torch.from_numpy(x.astype(np.float32)).to(self.device)
        y_t = torch.from_numpy(y.astype(np.float32)).to(self.device)
        if x_t.dim() == 1:
            x_t = x_t.unsqueeze(-1)

        T = len(x_t)
        chunk = self.chunk_size

        if reset_state:
            eval_state = self.model.init_state(batch_size=1, device=self.device)
        else:
            # Continue from training — this is the production path
            eval_state = self.state if self.state is not None else \
                self.model.init_state(batch_size=1, device=self.device)

        all_preds = []
        n_chunks = T // chunk
        for i in range(n_chunks):
            s = i * chunk
            e = s + chunk
            x_chunk = x_t[s:e].unsqueeze(0)
            pred, eval_state = self.model(x_chunk, eval_state)
            all_preds.append(pred.squeeze().cpu())

        if not all_preds:
            return {"MSE": float("nan"), "IC": 0.0, "p": 1.0, "n": 0}

        preds = torch.cat(all_preds).numpy()
        true = y[:len(preds)]

        from scipy.stats import spearmanr
        mse_val = float(((preds - true) ** 2).mean())
        if len(preds) >= 20:
            ic, p = spearmanr(preds, true)
        else:
            ic, p = 0.0, 1.0

        return {"MSE": mse_val, "IC": float(ic), "p": float(p), "n": len(preds)}
