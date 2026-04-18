"""
Label computation for HOPE Phase 2.
Labels = net forward return after transaction costs.
"""
from __future__ import annotations

import pandas as pd


def compute_labels(df: pd.DataFrame, lookahead: int = 5,
                   cost_bps: float = 3.0) -> pd.Series:
    """
    Label = net return of a long trade entered now, closed in `lookahead` bars.
    Net of transaction costs (spread + commission).

    cost_bps = 3.0: conservative estimate for NSE (1bp spread + 2bp commission)

    CRITICAL: Uses shift(-lookahead) which looks FORWARD.
    In live trading, labels are never available at entry — only for training.
    No lookahead bias because the label IS the future return we want to predict.
    """
    gross_return = (df['close'].shift(-lookahead) - df['close']) / df['close']
    net_return = gross_return - (cost_bps / 10000)
    return net_return
