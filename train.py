"""
Training loop for Mini-BAKA / LSTM on forward-return labels.

Key invariants (from CLAUDE.md):
    - NEVER shuffle across time. Sequences are consecutive.
    - Labels are returns (net of costs), never raw price.
    - Walk-forward validation — train on [0, t], validate on (t, t+v].
    - Loss is Sharpe (or IC) over confident predictions, not MSE.

Public entry points:
    train_one_window(model, X_train, y_train, ...)
    walk_forward_evaluation(model_factory, features, labels, ...)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import spearmanr
from torch.utils.data import Dataset


# ============================================================ dataset


class SequentialWindowDataset(Dataset):
    """
    Sliding-window dataset over consecutive time steps.
    Each item is (X[t-window+1 : t+1], y[t]) — NO shuffling across boundaries.

    The returned windows may overlap, which is fine: it's the *ordering* of
    items that matters, and we preserve that by feeding batches in index
    order (shuffle=False in the DataLoader).
    """

    def __init__(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        window: int = 64,
    ):
        assert features.shape[0] == labels.shape[0]
        self.features = features.astype(np.float32)
        self.labels = labels.astype(np.float32)
        self.window = window

    def __len__(self) -> int:
        return max(0, len(self.features) - self.window + 1)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        X = self.features[i : i + self.window]       # [window, n_features]
        y = self.labels[i + self.window - 1]         # scalar
        return torch.from_numpy(X), torch.tensor(y, dtype=torch.float32)


# ============================================================ losses


def sharpe_loss(
    preds: torch.Tensor,
    actual: torch.Tensor,
    threshold: float = 5e-4,
) -> torch.Tensor:
    """
    -(mean / std) of the returns the model chose to trade on.
    Signed by prediction direction; trades filtered by |pred| > threshold.
    """
    trade_returns = actual * torch.sign(preds)
    confident = preds.abs() > threshold
    if int(confident.sum().item()) < 2:
        # Fall back to a differentiable signal so training doesn't stall.
        return -((preds * actual).mean())
    taken = trade_returns[confident]
    mean = taken.mean()
    std = taken.std() + 1e-8
    return -(mean / std)


def ic_loss(preds: torch.Tensor, actual: torch.Tensor) -> torch.Tensor:
    """Negative Pearson correlation — a smooth proxy for IC."""
    p = preds - preds.mean()
    a = actual - actual.mean()
    ic = (p * a).mean() / (p.std() * a.std() + 1e-8)
    return -ic


# ============================================================ config


@dataclass
class TrainConfig:
    window: int = 64
    batch_size: int = 64
    epochs: int = 5
    lr: float = 1e-3
    weight_decay: float = 1e-5
    loss: str = "ic"          # "ic" or "sharpe"
    device: str = "cpu"
    log_every: int = 50


# ============================================================ training


def _batches(
    features: np.ndarray,
    labels: np.ndarray,
    window: int,
    batch_size: int,
):
    """Yield consecutive batches with NO shuffling via zero-copy stride tricks."""
    n_samples = len(features) - window + 1
    if n_samples <= 0:
        return

    # Convert to flat tensors
    X_f = torch.from_numpy(features.astype(np.float32))
    y_f = torch.from_numpy(labels.astype(np.float32))

    # O(1) sliding window view: [T, D] -> [T-window+1, D, window] -> [T-window+1, window, D]
    X_windows = X_f.unfold(0, window, 1).transpose(1, 2)
    y_targets = y_f[window - 1 : window - 1 + n_samples]

    idx = 0
    while idx + batch_size <= n_samples:
        yield X_windows[idx : idx + batch_size], y_targets[idx : idx + batch_size]
        idx += batch_size


def train_one_window(
    model: nn.Module,
    features: np.ndarray,
    labels: np.ndarray,
    cfg: TrainConfig = TrainConfig(),
) -> nn.Module:
    """Train `model` in place on a single contiguous window of data."""
    device = torch.device(cfg.device)
    model = model.to(device)
    model.train()
    if hasattr(model, "reset_memory"):
        model.reset_memory()

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    loss_fn = ic_loss if cfg.loss == "ic" else sharpe_loss

    step = 0
    for epoch in range(cfg.epochs):
        epoch_losses: list[float] = []
        if hasattr(model, "reset_memory"):
            model.reset_memory()
        for X, y in _batches(features, labels, cfg.window, cfg.batch_size):
            X = X.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            preds = model(X).squeeze(-1)
            loss = loss_fn(preds, y)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            epoch_losses.append(loss.detach())
            step += 1
            if step % cfg.log_every == 0:
                recent_loss = np.mean([l.item() for l in epoch_losses[-cfg.log_every:]])
                print(f"  epoch {epoch} step {step} loss={recent_loss:.4f}")
        mean_loss = np.mean([l.item() for l in epoch_losses])
        print(f"epoch {epoch}: mean loss={mean_loss:.4f}")
    return model


@torch.no_grad()
def predict(
    model: nn.Module,
    features: np.ndarray,
    labels: np.ndarray,
    cfg: TrainConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Returns (predictions, aligned_labels) over the full window."""
    device = torch.device(cfg.device)
    model = model.to(device)
    model.eval()
    if hasattr(model, "reset_memory"):
        model.reset_memory()

    preds: list[float] = []
    aligned: list[float] = []
    for X, y in _batches(features, labels, cfg.window, cfg.batch_size):
        X = X.to(device)
        out = model(X).squeeze(-1).detach().cpu().numpy()
        preds.extend(out.tolist())
        aligned.extend(y.numpy().tolist())
    return np.array(preds), np.array(aligned)


# ============================================================ walk-forward


@dataclass
class WFResult:
    period: int
    train_end: int
    val_end: int
    ic: float
    p_value: float
    n: int


def walk_forward_evaluation(
    model_factory: Callable[[], nn.Module],
    features: pd.DataFrame,
    labels: pd.Series,
    train_frac: float = 0.6,
    val_frac: float = 0.1,
    cfg: TrainConfig = TrainConfig(),
) -> list[WFResult]:
    """
    Anchored walk-forward: first window = [0, train_frac), validate on the
    next val_frac. Advance by val_frac until end-of-data.
    """
    n = len(features)
    train_size = int(n * train_frac)
    val_size = max(1, int(n * val_frac))

    X = features.to_numpy()
    y = labels.to_numpy()

    results: list[WFResult] = []
    start = 0
    train_end = start + train_size
    while train_end + val_size <= n:
        val_end = train_end + val_size
        print(
            f"\n--- Period {len(results)+1}: "
            f"train=[{start}:{train_end}) val=[{train_end}:{val_end}) ---"
        )
        model = model_factory()
        train_one_window(model, X[start:train_end], y[start:train_end], cfg)
        preds, aligned = predict(model, X[train_end:val_end], y[train_end:val_end], cfg)
        if len(preds) < 20:
            print("  (too few val samples — skipping)")
            train_end += val_size
            continue
        ic, p = spearmanr(preds, aligned)
        results.append(
            WFResult(
                period=len(results) + 1,
                train_end=train_end,
                val_end=val_end,
                ic=float(ic),
                p_value=float(p),
                n=int(len(preds)),
            )
        )
        print(f"  val IC={ic:.4f}  p={p:.4f}  n={len(preds)}")
        train_end += val_size  # anchor the start, slide the val window

    if results:
        mean_ic = float(np.mean([r.ic for r in results]))
        pos = sum(1 for r in results if r.ic > 0)
        print(f"\n== Walk-forward summary: mean IC={mean_ic:.4f}, positive in {pos}/{len(results)} periods ==")
    return results


if __name__ == "__main__":
    # Smoke test.
    from models import LSTMBaseline
    rng = np.random.default_rng(0)
    n_features = 11
    n = 2000
    feats = rng.normal(0, 1, (n, n_features)).astype(np.float32)
    labs = (feats[:, 0] * 0.01 + rng.normal(0, 0.005, n)).astype(np.float32)
    results = walk_forward_evaluation(
        lambda: LSTMBaseline(n_features=n_features, hidden_dim=32, n_layers=1, n_outputs=1),
        pd.DataFrame(feats),
        pd.Series(labs),
        train_frac=0.6,
        val_frac=0.1,
        cfg=TrainConfig(window=32, batch_size=32, epochs=2, log_every=200),
    )
    print(f"{len(results)} periods evaluated")
