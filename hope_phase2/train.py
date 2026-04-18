"""
Multi-stock streaming training for HOPE Phase 2.

Each stock is processed with batch_size=1 (matching Phase 1 titans.py).
Losses are averaged across all stocks per chunk, then backprop'd.
Each stock maintains its own Titans state dict.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn as nn

from losses import ic_loss, mse_loss


def detach_states(states: list) -> list:
    """Detach gradient tape from states (TBPTT)."""
    if states is None:
        return None
        
    new_states = []
    for state in states:
        if state is None:
            new_states.append(None)
        elif isinstance(state, tuple):
            # LSTM states: (h, c)
            new_states.append(tuple(s.detach().clone() for s in state))
        elif isinstance(state, dict):
            # HOPE states: dict
            new_state = {}
            for k, v in state.items():
                if k == "step":
                    new_state[k] = v
                elif isinstance(v, list):
                    new_state[k] = [m.detach().clone() for m in v]
                elif isinstance(v, torch.Tensor):
                    new_state[k] = v.detach().clone()
                else:
                    new_state[k] = v
            new_states.append(new_state)
    return new_states


def train_epoch_finance(model, feat_tensor, lab_tensor,
                        optimizer, chunk_size, device,
                        all_stock_states=None):
    """
    Train on all stocks simultaneously, processing each stock with batch=1.

    feat_tensor: [n_stocks, T, n_features]
    lab_tensor:  [n_stocks, T]
    all_stock_states: list of n_stocks state dicts (one per stock)

    Each stock gets its own Titans state, processed sequentially per chunk.
    Losses are averaged across stocks for each chunk position.

    Returns: (mean_loss, all_stock_states)
    """
    model.train()
    n_stocks, T, n_features = feat_tensor.shape
    n_chunks = T // chunk_size

    feat_tensor = feat_tensor.to(device)
    lab_tensor = lab_tensor.to(device)

    # Initialize per-stock states if not provided
    if all_stock_states is None:
        all_stock_states = []
        for _ in range(n_stocks):
            all_stock_states.append(
                model.init_state(batch_size=1, device=torch.device(device))
            )

    # Reset CMS gradient buffers (HOPE only)
    if hasattr(model, 'reset_cms_buffers'):
        model.reset_cms_buffers()

    total_loss = 0.0
    step = 0

    for chunk_idx in range(n_chunks):
        s = chunk_idx * chunk_size
        e = s + chunk_size

        # Accumulate loss across all stocks for this chunk
        losses = []

        new_stock_states = []
        for stock_idx in range(n_stocks):
            # [1, chunk, n_features]
            x_chunk = feat_tensor[stock_idx:stock_idx + 1, s:e, :]
            # [chunk]
            y_chunk = lab_tensor[stock_idx, s:e]

            # Forward with this stock's state
            pred, stock_state = model(
                x_chunk, all_stock_states[stock_idx], step=step,
            )
            pred = pred.squeeze(-1).squeeze(0)  # [chunk]

            # IC loss for this stock
            stock_loss = ic_loss(pred, y_chunk)
            losses.append(stock_loss)

            new_stock_states.append(stock_state)

        # Average loss across stocks
        chunk_loss = torch.stack(losses).mean()

        # Backward
        optimizer.zero_grad()
        chunk_loss.backward()

        # CMS gradient accumulation (HOPE only)
        if hasattr(model, 'post_backward'):
            model.post_backward(step)

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # Detach all stock states (TBPTT)
        all_stock_states = [detach_states(st) for st in new_stock_states]

        total_loss += chunk_loss.item()
        step += chunk_size

    return total_loss / max(n_chunks, 1), all_stock_states
