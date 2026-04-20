"""
Phase 3 configuration.

Two modes:
  --mode dev   : 10 stocks, 3 months, 10 epochs (Colab debug)
  --mode prod  : 50 stocks, full data, 20 epochs (HPC)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass
class PhaseConfig:
    # ── Data ─────────────────────────────────────────────────────
    mode: str = "dev"            # "dev" or "prod"
    from_date: str = "2022-01-03"
    to_date: str = "2026-04-18"
    label_horizon: int = 15      # bars ahead
    transaction_cost: float = 0.0003  # 3bps

    # ── Features ──────────────────────────────────────────────────
    n_features: int = 70

    # ── Model ─────────────────────────────────────────────────────
    d_model: int = 128
    n_layers: int = 4
    d_memory: int = 64
    d_ffn: int = 256
    n_heads: int = 8
    inner_lr: float = 0.01
    inner_decay: float = 0.99
    grad_clip_inner: float = 1.0
    cms_levels: int = 3
    cms_schedule: List[int] = field(default_factory=lambda: [30, 120, 375])
    cms_lr: List[float] = field(default_factory=lambda: [1e-3, 1e-4, 1e-5])
    chunk_size: int = 64

    # ── LSTM (matched params) ─────────────────────────────────────
    lstm_hidden: int = 197
    lstm_layers: int = 4

    # ── Training ──────────────────────────────────────────────────
    epochs: int = 20
    lr: float = 1e-3
    weight_decay: float = 0.01
    min_lr: float = 3e-5
    grad_clip_outer: float = 1.0

    # ── Walk-forward splits ───────────────────────────────────────
    train_end: str = "2023-12-31"
    val_end: str = "2024-06-30"
    # test: val_end to to_date

    # ── Seeds ─────────────────────────────────────────────────────
    seeds: List[int] = field(default_factory=lambda: [0, 42, 123, 7, 99])

    @property
    def dev_stocks(self):
        return ["RELIANCE", "TCS", "INFY", "HDFCBANK", "ICICIBANK",
                "SBIN", "WIPRO", "MARUTI", "SUNPHARMA", "TATASTEEL"]

    @property
    def dev_from_date(self):
        return "2023-10-01"

    @property
    def dev_to_date(self):
        return "2024-01-31"

    @property
    def dev_epochs(self):
        return 10
