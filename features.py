"""
Feature engineering for NSE intraday / daily data.

All features are STATIONARY and computed using only past data (no lookahead).
Ordered set mirrors CLAUDE.md spec exactly. Extend with LOB features only
once Level-2 data is available (see compute_lob_features).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


DEFAULT_COST_BPS = 3  # NSE round-trip cost in basis points


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    df: DataFrame with columns [open, high, low, close, volume], DatetimeIndex.
    Returns: DataFrame of stationary, informative features (NaN-dropped).
    """
    required = {"open", "high", "low", "close", "volume"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"compute_features missing OHLCV columns: {missing}")

    f = pd.DataFrame(index=df.index)

    # --- Returns (stationary versions of price) ---
    f["ret_1"] = df["close"].pct_change(1)
    f["ret_5"] = df["close"].pct_change(5)
    f["ret_20"] = df["close"].pct_change(20)
    f["ret_60"] = df["close"].pct_change(60)

    # --- Volatility ---
    f["vol_5"] = f["ret_1"].rolling(5).std()
    f["vol_20"] = f["ret_1"].rolling(20).std()

    # --- Volume (normalized against its own 20-period mean) ---
    vol_ma = df["volume"].rolling(20).mean()
    f["vol_surprise"] = (df["volume"] / vol_ma) - 1

    # --- Price position within recent range ---
    high_20 = df["high"].rolling(20).max()
    low_20 = df["low"].rolling(20).min()
    f["range_position"] = (df["close"] - low_20) / (high_20 - low_20 + 1e-8)

    # --- Momentum ---
    f["mom_5"] = df["close"] / df["close"].shift(5) - 1
    f["mom_20"] = df["close"] / df["close"].shift(20) - 1

    # --- Mean reversion ---
    sma_20 = df["close"].rolling(20).mean()
    f["reversion"] = (df["close"] - sma_20) / sma_20

    # --- Time-of-day (only meaningful for intraday data) ---
    if getattr(df.index, "freqstr", None) and "min" in (df.index.freqstr or ""):
        _add_time_features(f, df.index)
    elif hasattr(df.index, "hour") and df.index.hour.nunique() > 1:
        _add_time_features(f, df.index)

    return f.dropna()


def _add_time_features(f: pd.DataFrame, idx: pd.DatetimeIndex) -> None:
    # NSE session: 09:15 to 15:30 IST — 375 minutes total.
    minutes_since_open = (idx.hour - 9) * 60 + (idx.minute - 15)
    total_minutes = 375
    f["time_sin"] = np.sin(2 * np.pi * minutes_since_open / total_minutes)
    f["time_cos"] = np.cos(2 * np.pi * minutes_since_open / total_minutes)


def compute_lob_features(lob: pd.DataFrame, close: pd.Series) -> pd.DataFrame:
    """
    Optional Level-2 order book features. Strongest short-horizon predictors
    when available.

    lob columns required: bid_qty, ask_qty, bid_price, ask_price
    """
    required = {"bid_qty", "ask_qty", "bid_price", "ask_price"}
    missing = required - set(lob.columns)
    if missing:
        raise ValueError(f"compute_lob_features missing columns: {missing}")

    f = pd.DataFrame(index=lob.index)
    denom = lob["bid_qty"] + lob["ask_qty"] + 1e-8
    f["order_imbalance"] = (lob["bid_qty"] - lob["ask_qty"]) / denom
    f["spread_bps"] = (lob["ask_price"] - lob["bid_price"]) / close * 10000
    return f.dropna()


def make_labels(
    df: pd.DataFrame,
    lookahead: int = 5,
    cost_bps: float = DEFAULT_COST_BPS,
) -> pd.Series:
    """
    Forward-return label, NET of transaction costs.
    Always uses returns — never raw price. Shifted so the label at time t
    is the net return from t to t+lookahead.
    """
    gross = (df["close"].shift(-lookahead) - df["close"]) / df["close"]
    net = gross - (cost_bps / 10000)
    return net.rename(f"ret_fwd_{lookahead}_net")


def align_features_labels(
    features: pd.DataFrame, labels: pd.Series
) -> tuple[pd.DataFrame, pd.Series]:
    """Inner-join on index and drop rows with NaN on either side."""
    df = features.join(labels, how="inner").dropna()
    y = df[labels.name]
    X = df.drop(columns=[labels.name])
    return X, y


if __name__ == "__main__":
    # Smoke test with a small synthetic dataset.
    idx = pd.date_range("2023-01-01 09:15", periods=500, freq="1min")
    price = 100 + np.cumsum(np.random.randn(500) * 0.1)
    df = pd.DataFrame(
        {
            "open": price + np.random.randn(500) * 0.01,
            "high": price + np.abs(np.random.randn(500)) * 0.02,
            "low": price - np.abs(np.random.randn(500)) * 0.02,
            "close": price,
            "volume": np.random.randint(1000, 5000, 500),
        },
        index=idx,
    )
    feats = compute_features(df)
    labels = make_labels(df, lookahead=5)
    X, y = align_features_labels(feats, labels)
    print(f"Features shape: {X.shape}, labels shape: {y.shape}")
    print(f"Feature columns: {list(X.columns)}")
