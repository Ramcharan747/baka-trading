"""
Backtest: simulated trading on model predictions.

Returns: (total_pnl, n_trades, win_rate, sharpe)
"""
from __future__ import annotations

import numpy as np


def compute_backtest_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                             threshold: float = 0.001):
    """
    Simulates trading based on predictions.

    y_true: Net return (already minus 3bps for longs).
    y_pred: Model prediction.
    threshold: Minimum absolute prediction to enter a trade.

    Returns: (total_pnl, n_trades, win_rate, sharpe)
    """
    long_trades = y_pred > threshold
    short_trades = y_pred < -threshold

    # Long returns: already net of 3bps in y_true
    long_returns = y_true[long_trades]
    long_pnl = np.sum(long_returns)
    long_wins = np.sum(long_returns > 0)

    # Short returns: -(gross) - 3bps = -y_true - 6bps
    short_returns = -y_true[short_trades] - 0.0006
    short_pnl = np.sum(short_returns)
    short_wins = np.sum(short_returns > 0)

    total_pnl = long_pnl + short_pnl
    n_trades = np.sum(long_trades) + np.sum(short_trades)

    if n_trades > 0:
        win_rate = (long_wins + short_wins) / n_trades
    else:
        win_rate = 0.0

    # Annualized Sharpe from per-bar PnL
    daily_pnl = np.zeros(len(y_true))
    daily_pnl[long_trades] = long_returns
    daily_pnl[short_trades] = short_returns

    if daily_pnl.std() > 1e-10:
        sharpe = (daily_pnl.mean() / daily_pnl.std()) * np.sqrt(252 * 375)
    else:
        sharpe = 0.0

    return total_pnl, n_trades, win_rate, sharpe


# Alias
run_backtest = compute_backtest_metrics
