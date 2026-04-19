import numpy as np


def compute_backtest_metrics(y_true: np.ndarray, y_pred: np.ndarray, threshold: float = 0.001):
    """
    Simulates trading based on predictions.
    
    y_true: Net 5-day return of a long trade (already assumes gross - 3bps cost).
    y_pred: Model prediction.
    threshold: Minimum absolute prediction score to enter a trade.
    
    Returns: (total_pnl, n_trades, win_rate)
    """
    long_trades = y_pred > threshold
    short_trades = y_pred < -threshold
    
    # Long returns: already net of 3bps in y_true (as defined in labels.py)
    long_returns = y_true[long_trades]
    long_pnl = np.sum(long_returns)
    long_wins = np.sum(long_returns > 0)
    
    # Short returns: selling short incurs its own 3bps cost.
    # If y_true = gross - 3bps, then gross = y_true + 3bps.
    # Short net = -gross - 3bps = -(y_true + 3bps) - 3bps = -y_true - 6bps 
    # Or more simply: -gross - 3bps = -(y_true + 0.0003) - 0.0003 = -y_true - 0.0006
    short_returns = -y_true[short_trades] - 0.0006
    short_pnl = np.sum(short_returns)
    short_wins = np.sum(short_returns > 0)
    
    total_pnl = long_pnl + short_pnl
    n_trades = np.sum(long_trades) + np.sum(short_trades)
    
    if n_trades > 0:
        win_rate = (long_wins + short_wins) / n_trades
    else:
        win_rate = 0.0
        
    return total_pnl, n_trades, win_rate


# Alias for backward compatibility
run_backtest = compute_backtest_metrics
