"""
25 quant features for HOPE Phase 2. All stationary, normalized, causal.
No raw price or volume — only returns, ratios, z-scores.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    df: DataFrame with columns [open, high, low, close, volume]
        sorted ascending by timestamp, one row per bar.

    Returns: DataFrame with 25 features, same index as df.
    All features are stationary and normalized.

    CRITICAL: Every rolling/shift operation uses only past data.
    Never use df['close'].shift(-1) or any negative shift.
    """
    f = pd.DataFrame(index=df.index)
    eps = 1e-8  # prevent division by zero everywhere

    # ── GROUP 1: RETURNS (5 features) ──────────────────────────────────────
    f['ret_1'] = df['close'].pct_change(1)
    f['ret_5'] = df['close'].pct_change(5)
    f['ret_15'] = df['close'].pct_change(15)
    f['ret_30'] = df['close'].pct_change(30)
    f['gap'] = (df['open'] - df['close'].shift(1)) / (df['close'].shift(1) + eps)

    # ── GROUP 2: VOLATILITY (4 features) ───────────────────────────────────
    f['rvol_10'] = f['ret_1'].rolling(10).std()
    f['rvol_30'] = f['ret_1'].rolling(30).std()

    high_low = df['high'] - df['low']
    high_close = (df['high'] - df['close'].shift(1)).abs()
    low_close = (df['low'] - df['close'].shift(1)).abs()
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr_14 = true_range.rolling(14).mean()
    f['atr_ratio'] = atr_14 / (df['close'] + eps)

    f['vol_ratio'] = (f['rvol_10'] + eps) / (f['rvol_30'] + eps)

    # ── GROUP 3: VOLUME (4 features) ───────────────────────────────────────
    vol_ma_20 = df['volume'].rolling(20).mean()
    f['vol_surprise'] = (df['volume'] / (vol_ma_20 + eps)) - 1

    vol_ma_5 = df['volume'].rolling(5).mean()
    f['vol_trend'] = (vol_ma_5 / (vol_ma_20 + eps)) - 1

    f['pv_corr'] = f['ret_1'].rolling(30).corr(df['volume'].pct_change(1))

    f['log_vol'] = np.log(df['volume'] + 1)
    f['log_vol'] = (f['log_vol'] - f['log_vol'].rolling(30).mean()) / \
                   (f['log_vol'].rolling(30).std() + eps)

    # ── GROUP 4: VWAP FEATURES (4 features) ────────────────────────────────
    typical_price = (df['high'] + df['low'] + df['close']) / 3
    rolling_tp_vol = (typical_price * df['volume']).rolling(375).sum()
    rolling_vol = df['volume'].rolling(375).sum()
    vwap = rolling_tp_vol / (rolling_vol + eps)
    f['vwap_dev'] = (df['close'] - vwap) / (vwap + eps)

    f['vwap_zscore'] = (f['vwap_dev'] - f['vwap_dev'].rolling(60).mean()) / \
                       (f['vwap_dev'].rolling(60).std() + eps)

    f['vwap_slope'] = vwap.pct_change(5)

    vwap_std = f['vwap_dev'].rolling(30).std()
    f['vwap_band'] = f['vwap_dev'] / (vwap_std + eps)

    # ── GROUP 5: PRICE MICROSTRUCTURE (4 features) ─────────────────────────
    bar_range = df['high'] - df['low'] + eps
    bar_body = (df['close'] - df['open']).abs()
    f['body_ratio'] = bar_body / bar_range

    f['buy_pressure'] = (df['close'] - df['open']) / bar_range

    f['close_position'] = (df['close'] - df['low']) / bar_range

    f['illiquidity'] = f['ret_1'].abs() / (df['volume'] + eps)
    f['illiquidity'] = np.log(f['illiquidity'] + eps)
    f['illiquidity'] = (f['illiquidity'] - f['illiquidity'].rolling(60).mean()) / \
                       (f['illiquidity'].rolling(60).std() + eps)

    # ── GROUP 6: MOMENTUM & MEAN REVERSION (4 features) ───────────────────
    delta = df['close'].diff(1)
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / (loss + eps)
    f['rsi'] = rs / (1 + rs)

    sma_20 = df['close'].rolling(20).mean()
    f['sma_dev'] = (df['close'] - sma_20) / (sma_20 + eps)

    high_60 = df['high'].rolling(60).max()
    low_60 = df['low'].rolling(60).min()
    f['range_pos_60'] = (df['close'] - low_60) / (high_60 - low_60 + eps)

    # Time of day — cyclical encoding
    if hasattr(df.index, 'hour'):
        minutes_since_open = (df.index.hour - 9) * 60 + (df.index.minute - 15)
        total_minutes = 375
        f['time_sin'] = np.sin(2 * np.pi * minutes_since_open / total_minutes)
        f['time_cos'] = np.cos(2 * np.pi * minutes_since_open / total_minutes)
    else:
        # Daily bars: use day-of-week encoding
        try:
            f['time_sin'] = np.sin(2 * np.pi * df.index.dayofweek / 5)
            f['time_cos'] = np.cos(2 * np.pi * df.index.dayofweek / 5)
        except AttributeError:
            f['time_sin'] = 0.0
            f['time_cos'] = 0.0

    # ── FINAL CLEANUP ───────────────────────────────────────────────────────
    f = f.replace([np.inf, -np.inf], np.nan)
    f = f.ffill()
    f = f.dropna()

    return f
