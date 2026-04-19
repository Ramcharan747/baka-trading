"""
Label computation for Phase 3: 15-bar-ahead net return minus 3bps cost.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def compute_labels(close: pd.Series, horizon: int = 15,
                   cost: float = 0.0003) -> pd.Series:
    """
    Forward-looking label: net return after transaction cost.

    label_t = close_{t+horizon} / close_t - 1 - cost

    This is computed ONCE during data prep and stored alongside features.
    Never recomputed during evaluation.
    """
    gross_return = close.shift(-horizon) / close - 1
    net_return = gross_return - cost
    return net_return
