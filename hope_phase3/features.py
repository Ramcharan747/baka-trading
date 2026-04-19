"""
70 engineered features for minute-bar financial prediction.

8 groups:
  1. Multi-Scale Returns (10)
  2. Volatility (8)
  3. Volume (10)
  4. VWAP (8)
  5. Microstructure (10)
  6. Intraday Seasonality (8)
  7. Momentum & Mean Reversion (10)
  8. Cross-Asset / Market Factor (6)

All features are z-scored: clip to [-3, 3] then divide by 3.
All NaN/inf replaced with 0 after z-scoring.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# ── Canonical feature names (order matters for checkpoint compat) ────
FEATURE_NAMES = [
    # Group 1: Multi-Scale Returns (10)
    "ret_1", "ret_3", "ret_5", "ret_10", "ret_15", "ret_30", "ret_75",
    "ret_prev_day", "gap", "ret_open_to_now",
    # Group 2: Volatility (8)
    "rvol_5", "rvol_20", "rvol_60", "rvol_375", "vol_ratio",
    "atr_ratio", "vol_zscore", "vol_regime",
    # Group 3: Volume (10)
    "vol_surprise_5", "vol_surprise_20", "vol_surprise_375",
    "log_vol_zscore", "pv_corr_20", "vol_trend",
    "dollar_vol_zscore", "vol_acceleration", "tick_intensity",
    "vwap_vol_ratio",
    # Group 4: VWAP (8)
    "vwap_dev", "vwap_zscore", "vwap_slope_5", "vwap_slope_20",
    "vwap_band", "vwap_touch", "vwap_above_pct", "prev_day_vwap_dev",
    # Group 5: Microstructure (10)
    "body_ratio", "buy_pressure", "close_pos", "upper_wick",
    "lower_wick", "bar_accel", "amihud", "consec_dir",
    "range_expansion", "price_impact",
    # Group 6: Intraday Seasonality (8)
    "time_sin", "time_cos", "is_opening_30", "is_closing_30",
    "session_progress", "day_of_week_sin", "day_of_week_cos",
    "is_expiry_week",
    # Group 7: Momentum & Mean Reversion (10)
    "rsi_14", "rsi_30", "sma_dev_20", "sma_dev_60",
    "range_pos_20", "range_pos_60", "macd_hist",
    "momentum_5", "momentum_20", "reversal",
    # Group 8: Cross-Asset (6)
    "nifty_ret_1", "nifty_ret_5", "stock_vs_nifty",
    "beta_adj_ret", "relative_str_20", "sector_zscore",
]
assert len(FEATURE_NAMES) == 70


def _zscore(s: pd.Series, clip_val: float = 3.0) -> pd.Series:
    """Robust z-score: clip to [-3, 3] then divide by 3."""
    m = s.rolling(375 * 5, min_periods=20).mean()
    sd = s.rolling(375 * 5, min_periods=20).std()
    z = (s - m) / (sd + 1e-8)
    return z.clip(-clip_val, clip_val) / clip_val


def _rsi(close: pd.Series, n: int) -> pd.Series:
    """Exponential RSI."""
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(span=n, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(span=n, adjust=False).mean()
    rs = gain / (loss + 1e-8)
    return (100 - 100 / (1 + rs)) / 50 - 1  # normalize to [-1, 1]


def _signed_run_length(signs: pd.Series) -> pd.Series:
    """Count consecutive same-sign bars. Positive for up, negative for down."""
    result = np.zeros(len(signs))
    run = 0
    for i in range(len(signs)):
        if signs.iloc[i] > 0:
            run = max(run, 0) + 1
        elif signs.iloc[i] < 0:
            run = min(run, 0) - 1
        else:
            run = 0
        result[i] = run
    return pd.Series(result, index=signs.index)


def compute_features(df: pd.DataFrame,
                     nifty_df: pd.DataFrame = None,
                     sector_close: pd.Series = None) -> pd.DataFrame:
    """
    Compute all 70 features from OHLCV data.

    Args:
        df: DataFrame with [datetime, open, high, low, close, volume]
        nifty_df: NIFTY50 index DataFrame (same format), aligned timestamps
        sector_close: Mean close price of the stock's sector peers

    Returns: DataFrame with 70 feature columns
    """
    close = df['close'].astype(float)
    open_ = df['open'].astype(float)
    high = df['high'].astype(float)
    low = df['low'].astype(float)
    volume = df['volume'].astype(float)
    dt = pd.to_datetime(df['datetime'])

    feats = pd.DataFrame(index=df.index)

    # ══ GROUP 1: Multi-Scale Returns (10) ══════════════════════════
    feats['ret_1'] = close.pct_change(1)
    feats['ret_3'] = close.pct_change(3)
    feats['ret_5'] = close.pct_change(5)
    feats['ret_10'] = close.pct_change(10)
    feats['ret_15'] = close.pct_change(15)
    feats['ret_30'] = close.pct_change(30)
    feats['ret_75'] = close.pct_change(75)

    # Previous day's total return (close at 15:29 vs day-1 close at 15:29)
    date = dt.dt.date
    day_close = close.groupby(date).transform('last')
    prev_day_close = day_close.groupby(date).first().shift(1)
    prev_day_close_aligned = date.map(prev_day_close)
    feats['ret_prev_day'] = (day_close / prev_day_close_aligned - 1)

    # Overnight gap
    day_open = close.groupby(date).transform('first')
    feats['gap'] = (day_open / prev_day_close_aligned - 1)

    # Open-to-now return
    feats['ret_open_to_now'] = close / day_open - 1

    # ══ GROUP 2: Volatility (8) ════════════════════════════════════
    log_ret = np.log(close / close.shift(1))
    feats['rvol_5'] = log_ret.rolling(5).std()
    feats['rvol_20'] = log_ret.rolling(20).std()
    feats['rvol_60'] = log_ret.rolling(60).std()
    feats['rvol_375'] = log_ret.rolling(375).std()
    feats['vol_ratio'] = feats['rvol_5'] / (feats['rvol_60'] + 1e-8)

    # ATR ratio
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs(),
    ], axis=1).max(axis=1)
    atr_14 = tr.ewm(span=14, adjust=False).mean()
    feats['atr_ratio'] = atr_14 / (close + 1e-8)

    # Volatility z-score and regime
    rvol_5 = feats['rvol_5']
    feats['vol_zscore'] = (rvol_5 - rvol_5.rolling(375 * 5, min_periods=100).mean()) / \
                          (rvol_5.rolling(375 * 5, min_periods=100).std() + 1e-8)
    feats['vol_regime'] = (rvol_5 > feats['rvol_375']).astype(float)

    # ══ GROUP 3: Volume (10) ═══════════════════════════════════════
    feats['vol_surprise_5'] = volume / (volume.rolling(5).mean() + 1)
    feats['vol_surprise_20'] = volume / (volume.rolling(20).mean() + 1)
    feats['vol_surprise_375'] = volume / (volume.rolling(375).mean() + 1)
    feats['log_vol_zscore'] = _zscore(np.log1p(volume))
    feats['pv_corr_20'] = close.rolling(20).corr(volume)
    feats['vol_trend'] = volume.rolling(5).mean() / (volume.rolling(20).mean() + 1)

    dollar_vol = close * volume
    feats['dollar_vol_zscore'] = _zscore(np.log1p(dollar_vol))
    feats['vol_acceleration'] = feats['vol_surprise_5'] / (feats['vol_surprise_20'] + 1e-3)
    feats['tick_intensity'] = volume / (high - low + close * 1e-4)
    feats['vwap_vol_ratio'] = feats['vol_surprise_5'] / (feats['vol_surprise_20'] + 1e-3)

    # ══ GROUP 4: VWAP (8) ══════════════════════════════════════════
    cum_dollar = (close * volume).groupby(date).cumsum()
    cum_vol = volume.groupby(date).cumsum()
    vwap = cum_dollar / (cum_vol + 1)

    vwap_dev = (close - vwap) / (vwap + 1e-8)
    feats['vwap_dev'] = vwap_dev
    feats['vwap_zscore'] = vwap_dev / (vwap_dev.rolling(30).std() + 1e-8)
    feats['vwap_slope_5'] = vwap.diff(5) / (vwap.shift(5) + 1e-8)
    feats['vwap_slope_20'] = vwap.diff(20) / (vwap.shift(20) + 1e-8)
    feats['vwap_band'] = vwap_dev / (vwap_dev.rolling(60).std() + 1e-8)

    vwap_cross = ((close > vwap) != (close.shift(1) > vwap.shift(1))).astype(float)
    feats['vwap_touch'] = vwap_cross.rolling(30).sum()
    feats['vwap_above_pct'] = (close > vwap).astype(float).rolling(30).mean()

    # Previous day VWAP
    day_final_vwap = vwap.groupby(date).transform('last')
    prev_day_vwap_series = day_final_vwap.groupby(date).first().shift(1)
    prev_day_vwap = date.map(prev_day_vwap_series)
    feats['prev_day_vwap_dev'] = (close - prev_day_vwap) / (prev_day_vwap + 1e-8)

    # ══ GROUP 5: Microstructure (10) ═══════════════════════════════
    range_ = high - low + close * 1e-4
    feats['body_ratio'] = (close - open_).abs() / range_
    feats['buy_pressure'] = (close - open_) / range_
    feats['close_pos'] = (close - low) / range_

    oc_max = pd.concat([open_, close], axis=1).max(axis=1)
    oc_min = pd.concat([open_, close], axis=1).min(axis=1)
    feats['upper_wick'] = (high - oc_max) / range_
    feats['lower_wick'] = (oc_min - low) / range_

    feats['bar_accel'] = log_ret / (rvol_5 + 1e-8)
    feats['amihud'] = log_ret.abs() / (dollar_vol + 1)
    feats['consec_dir'] = _signed_run_length(np.sign(log_ret)) / 10.0
    feats['range_expansion'] = range_ / (range_.rolling(20).mean() + 1e-4)
    feats['price_impact'] = log_ret.abs() / (feats['vol_surprise_5'] + 1e-3)

    # ══ GROUP 6: Intraday Seasonality (8) ══════════════════════════
    minutes_since_open = (dt.dt.hour * 60 + dt.dt.minute) - (9 * 60 + 15)
    session_progress = minutes_since_open / 375.0

    feats['time_sin'] = np.sin(2 * np.pi * session_progress)
    feats['time_cos'] = np.cos(2 * np.pi * session_progress)
    feats['is_opening_30'] = (minutes_since_open <= 30).astype(float)
    feats['is_closing_30'] = (minutes_since_open >= 345).astype(float)
    feats['session_progress'] = session_progress

    dow = dt.dt.dayofweek
    feats['day_of_week_sin'] = np.sin(2 * np.pi * dow / 5)
    feats['day_of_week_cos'] = np.cos(2 * np.pi * dow / 5)
    # Option expiry effect (Thursday)
    feats['is_expiry_week'] = (dow == 3).astype(float)  # Thursday = expiry day

    # ══ GROUP 7: Momentum & Mean Reversion (10) ════════════════════
    feats['rsi_14'] = _rsi(close, 14)
    feats['rsi_30'] = _rsi(close, 30)
    feats['sma_dev_20'] = (close - close.rolling(20).mean()) / \
                          (close.rolling(20).mean() + 1e-8)
    feats['sma_dev_60'] = (close - close.rolling(60).mean()) / \
                          (close.rolling(60).mean() + 1e-8)
    feats['range_pos_20'] = (close - close.rolling(20).min()) / \
                            (close.rolling(20).max() - close.rolling(20).min() + 1e-8)
    feats['range_pos_60'] = (close - close.rolling(60).min()) / \
                            (close.rolling(60).max() - close.rolling(60).min() + 1e-8)

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    macd_signal = macd_line.ewm(span=9, adjust=False).mean()
    feats['macd_hist'] = (macd_line - macd_signal) / (close + 1e-8)

    feats['momentum_5'] = feats['ret_5'] / (feats['rvol_20'] + 1e-8)
    feats['momentum_20'] = feats['ret_30'] / (feats['rvol_60'] + 1e-8)
    feats['reversal'] = -feats['ret_1'] / (feats['rvol_5'] + 1e-8)

    # ══ GROUP 8: Cross-Asset / Market Factor (6) ═══════════════════
    if nifty_df is not None and len(nifty_df) == len(df):
        nifty_close = nifty_df['close'].astype(float).values
        nifty_close = pd.Series(nifty_close, index=df.index)
        nifty_ret_1 = nifty_close.pct_change(1)
        nifty_ret_5 = nifty_close.pct_change(5)

        feats['nifty_ret_1'] = nifty_ret_1
        feats['nifty_ret_5'] = nifty_ret_5
        feats['stock_vs_nifty'] = feats['ret_5'] - nifty_ret_5

        # Rolling 60-bar beta
        cov_60 = feats['ret_1'].rolling(60).cov(nifty_ret_1)
        var_60 = nifty_ret_1.rolling(60).var() + 1e-8
        beta_60 = cov_60 / var_60
        feats['beta_adj_ret'] = feats['ret_5'] - beta_60 * nifty_ret_5

        # Relative strength vs sector
        if sector_close is not None and len(sector_close) == len(df):
            sector_ret_5 = sector_close.pct_change(5)
            feats['relative_str_20'] = (feats['ret_5'] - sector_ret_5)
            feats['sector_zscore'] = feats['relative_str_20'] / \
                                     (sector_ret_5.rolling(60).std() + 1e-8)
        else:
            feats['relative_str_20'] = 0.0
            feats['sector_zscore'] = 0.0
    else:
        feats['nifty_ret_1'] = 0.0
        feats['nifty_ret_5'] = 0.0
        feats['stock_vs_nifty'] = 0.0
        feats['beta_adj_ret'] = 0.0
        feats['relative_str_20'] = 0.0
        feats['sector_zscore'] = 0.0

    # ══ Z-score all non-binary features ════════════════════════════
    binary_feats = {'is_opening_30', 'is_closing_30', 'is_expiry_week',
                    'vol_regime', 'session_progress',
                    'time_sin', 'time_cos',
                    'day_of_week_sin', 'day_of_week_cos'}

    for col in feats.columns:
        if col not in binary_feats:
            feats[col] = feats[col].clip(-3, 3) / 3.0

    # Replace NaN/inf with 0
    feats = feats.replace([np.inf, -np.inf], 0.0)
    feats = feats.fillna(0.0)

    # Ensure correct column order
    feats = feats[FEATURE_NAMES]

    return feats
