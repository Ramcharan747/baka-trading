"""
Training loops for HOPE and LSTM on minute-bar financial data.

Key design:
- Streaming chunks (TBPTT) with state persistence across epochs
- Sharpe loss (directly optimizes risk-adjusted return)
- Gradient clipping on outer parameters
"""
from __future__ import annotations

from typing import Union

import torch
import torch.nn as nn
import numpy as np

from losses import sharpe_loss


def detach_states(states, model_type: str = "hope"):
    """Detach states from computation graph (TBPTT truncation)."""
    if model_type == "lstm":
        if states is None:
            return None
        return (states[0].detach(), states[1].detach())
    else:
        # HOPE: list of dicts per layer
        new_states = []
        for layer_state in states:
            new_layer = {}
            for k, v in layer_state.items():
                if isinstance(v, torch.Tensor):
                    new_layer[k] = v.detach()
                else:
                    new_layer[k] = v
            new_states.append(new_layer)
        return new_states


def init_states_hope(model, n_stocks: int, device):
    """Initialize HOPE states for all layers."""
    return model.init_state(n_stocks, device)


def init_states_lstm(model, n_stocks: int, device):
    """Initialize LSTM states."""
    return model.init_state(n_stocks, device)


def train_epoch_minute(model, feat_tensor: torch.Tensor,
                       lab_tensor: torch.Tensor,
                       optimizer, scheduler,
                       states, config,
                       model_type: str = "hope") -> tuple:
    """
    Train one epoch on minute-bar data with Sharpe loss.

    Args:
        model: MiniHOPE or LSTMBaseline
        feat_tensor: [n_stocks, T, n_features]
        lab_tensor: [n_stocks, T]
        optimizer: AdamW
        scheduler: CosineAnnealing or None
        states: model states (list of dicts for HOPE, tuple for LSTM)
        config: PhaseConfig
        model_type: "hope" or "lstm"

    Returns: (avg_loss, new_states)
    """
    model.train()
    n_stocks, T, _ = feat_tensor.shape
    chunk_size = config.chunk_size
    n_chunks = T // chunk_size
    total_loss = 0.0

    for ci in range(n_chunks):
        s = ci * chunk_size
        e = s + chunk_size
        x = feat_tensor[:, s:e, :]   # [n_stocks, chunk, n_features]
        y = lab_tensor[:, s:e]        # [n_stocks, chunk]

        if model_type == "hope":
            pred, states = model(x, states, step=s)
            states = detach_states(states, "hope")
        else:
            pred, states = model(x, states)
            states = detach_states(states, "lstm")

        pred = pred.squeeze(-1)  # [n_stocks, chunk]

        # Sharpe loss per stock, then average
        losses = []
        for si in range(n_stocks):
            losses.append(sharpe_loss(pred[si], y[si]))
        loss = torch.stack(losses).mean()

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), config.grad_clip_outer)
        optimizer.step()

        # CMS post-backward (HOPE only)
        if model_type == "hope":
            model.post_backward(ci)

        total_loss += loss.item()

    if scheduler is not None:
        scheduler.step()

    return total_loss / max(n_chunks, 1), states
