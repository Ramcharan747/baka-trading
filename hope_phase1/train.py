"""
Streaming training utilities for HOPE and LSTM.

CORRECT training paradigm:
  1. Feed data as one long continuous sequence
  2. Never reset model state between consecutive chunks
  3. TBPTT: backprop within each chunk, detach state at boundaries
  4. CMS gradient accumulation happens after loss.backward()
  5. Never shuffle — sequential order IS the signal
"""
from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn as nn

from losses import ic_loss, mse_loss


# ─── State management ──────────────────────────────────────────────

def detach_states(states: list) -> list:
    """
    Detach gradient tape from HOPE states (TBPTT).

    CRITICAL: This cuts the GRADIENT graph but preserves the MEMORY VALUES.
    Fast-weight matrices keep their learned values — only autograd is cut.
    Without this, GPU memory explodes over long sequences.
    """
    new_states = []
    for state in states:
        new_state = {}
        for k, v in state.items():
            if k == "step":
                new_state[k] = v
            elif isinstance(v, list):
                # List of per-batch memory matrices
                new_state[k] = [m.detach().clone() for m in v]
            elif isinstance(v, torch.Tensor):
                new_state[k] = v.detach().clone()
            else:
                new_state[k] = v
        new_states.append(new_state)
    return new_states


def detach_lstm_state(state):
    """Detach LSTM (h, c) state for TBPTT."""
    if state is None:
        return None
    return (state[0].detach(), state[1].detach())


# ─── HOPE training ─────────────────────────────────────────────────

def train_epoch_hope(
    model: nn.Module,
    x_train: np.ndarray,
    y_train: np.ndarray,
    optimizer: torch.optim.Optimizer,
    chunk_size: int = 16,
    device: str = "cuda",
    loss_fn: str = "mse",
    scheduler=None,
) -> float:
    """
    Train HOPE on FULL sequence as one continuous stream.

    Never resets state. Never shuffles. Exactly mirrors inference.

    Args:
        model:      MiniHOPE instance
        x_train:    [T] float array — full training sequence
        y_train:    [T] float array — targets for every step
        optimizer:  AdamW optimizer for outer loop params
        chunk_size: TBPTT chunk size (= Titans update frequency)
        device:     cuda or cpu
        loss_fn:    "mse" or "ic"
        scheduler:  Optional LR scheduler (stepped per chunk)

    Returns:
        Mean loss over all chunks
    """
    model.train()
    loss_func = mse_loss if loss_fn == "mse" else ic_loss

    T = len(x_train)
    n_chunks = T // chunk_size

    # Initialize state ONCE per epoch
    states = model.init_state(batch_size=1, device=torch.device(device))

    # Reset CMS gradient buffers
    model.reset_cms_buffers()

    total_loss = 0.0
    step = 0

    for chunk_idx in range(n_chunks):
        s = chunk_idx * chunk_size
        e = s + chunk_size

        # Prepare chunk
        x_chunk = torch.tensor(
            x_train[s:e], dtype=torch.float32, device=device
        ).unsqueeze(0).unsqueeze(-1)  # [1, chunk, 1]
        y_chunk = torch.tensor(
            y_train[s:e], dtype=torch.float32, device=device
        ).unsqueeze(0)  # [1, chunk]

        # Forward
        pred, states = model(x_chunk, states, step=step)
        pred = pred.squeeze(-1)  # [1, chunk]

        # Loss
        loss = loss_func(pred.squeeze(), y_chunk.squeeze())

        # Backward — outer loop only
        optimizer.zero_grad()
        loss.backward()

        # CMS gradient accumulation (MUST happen after backward, before step)
        model.post_backward(step)

        # Clip outer gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        if scheduler is not None:
            scheduler.step()

        # TBPTT: detach state (cut gradient tape, keep memory values)
        states = detach_states(states)

        total_loss += loss.item()
        step += chunk_size

    return total_loss / max(n_chunks, 1)


@torch.no_grad()
def evaluate_hope(
    model: nn.Module,
    x_eval: np.ndarray,
    y_eval: np.ndarray,
    states_in: list,
    chunk_size: int = 16,
    device: str = "cuda",
    reset_state: bool = False,
) -> dict:
    """
    Evaluate HOPE on held-out data.

    reset_state=False: state continues from training (production behavior)
    reset_state=True:  state reset to zero (measures memory contribution)

    The difference IC_persistent - IC_reset is the memory contribution.
    """
    model.eval()

    if reset_state:
        states = model.init_state(batch_size=1, device=torch.device(device))
    else:
        states = detach_states(states_in)

    T = len(x_eval)
    n_chunks = T // chunk_size
    all_preds = []
    step = 0

    for chunk_idx in range(n_chunks):
        s = chunk_idx * chunk_size
        e = s + chunk_size

        x_chunk = torch.tensor(
            x_eval[s:e], dtype=torch.float32, device=device
        ).unsqueeze(0).unsqueeze(-1)  # [1, chunk, 1]

        pred, states = model(x_chunk, states, step=step)
        pred = pred.squeeze().cpu().numpy()
        if pred.ndim == 0:
            pred = pred.reshape(1)
        all_preds.extend(pred.tolist())
        step += chunk_size

    if not all_preds:
        return {"IC": 0.0, "p": 1.0, "MSE": float("nan"), "W_norm": 0.0, "n": 0}

    from scipy.stats import spearmanr

    preds = np.array(all_preds)
    true = y_eval[: len(preds)]

    ic_val, p_val = spearmanr(preds, true)
    mse_val = float(((preds - true) ** 2).mean())

    # W_norm: measure if M_mem is changing
    w_norms = []
    for state in states:
        if "M_mem" in state:
            for M in state["M_mem"]:
                w_norms.append(M.norm().item())
    w_norm = np.mean(w_norms) if w_norms else 0.0

    return {
        "IC": float(ic_val),
        "p": float(p_val),
        "MSE": mse_val,
        "W_norm": w_norm,
        "n": len(preds),
    }


# ─── LSTM training (same paradigm for fair comparison) ──────────────

def train_epoch_lstm(
    model: nn.Module,
    x_train: np.ndarray,
    y_train: np.ndarray,
    optimizer: torch.optim.Optimizer,
    chunk_size: int = 16,
    device: str = "cuda",
    loss_fn: str = "mse",
    scheduler=None,
) -> float:
    """Same streaming training paradigm as HOPE for fair comparison."""
    model.train()
    loss_func = mse_loss if loss_fn == "mse" else ic_loss

    T = len(x_train)
    n_chunks = T // chunk_size
    state = None
    total_loss = 0.0

    for chunk_idx in range(n_chunks):
        s = chunk_idx * chunk_size
        e = s + chunk_size

        x_chunk = torch.tensor(
            x_train[s:e], dtype=torch.float32, device=device
        ).unsqueeze(0).unsqueeze(-1)
        y_chunk = torch.tensor(
            y_train[s:e], dtype=torch.float32, device=device
        )

        pred, state = model(x_chunk, state)
        pred = pred.squeeze(-1)

        loss = loss_func(pred.squeeze(), y_chunk.squeeze())

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        if scheduler is not None:
            scheduler.step()

        # Detach LSTM state (TBPTT)
        state = detach_lstm_state(state)
        total_loss += loss.item()

    return total_loss / max(n_chunks, 1)


@torch.no_grad()
def evaluate_lstm(
    model: nn.Module,
    x_eval: np.ndarray,
    y_eval: np.ndarray,
    device: str = "cuda",
    chunk_size: int = 16,
    reset_state: bool = True,
) -> dict:
    """Evaluate LSTM baseline."""
    model.eval()
    state = None
    all_preds = []

    T = len(x_eval)
    n_chunks = T // chunk_size

    for chunk_idx in range(n_chunks):
        s = chunk_idx * chunk_size
        e = s + chunk_size

        x_chunk = torch.tensor(
            x_eval[s:e], dtype=torch.float32, device=device
        ).unsqueeze(0).unsqueeze(-1)

        pred, state = model(x_chunk, state)
        state = detach_lstm_state(state)

        pred_np = pred.squeeze().cpu().numpy()
        if pred_np.ndim == 0:
            pred_np = pred_np.reshape(1)
        all_preds.extend(pred_np.tolist())

    if not all_preds:
        return {"IC": 0.0, "p": 1.0, "MSE": float("nan"), "n": 0}

    from scipy.stats import spearmanr

    preds = np.array(all_preds)
    true = y_eval[: len(preds)]

    ic_val, p_val = spearmanr(preds, true)
    mse_val = float(((preds - true) ** 2).mean())

    return {"IC": float(ic_val), "p": float(p_val), "MSE": mse_val, "n": len(preds)}
