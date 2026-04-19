"""
Minute-bar data pipeline: Upstox API → parquet cache → aligned DataFrames.

Caching: data/cache/{symbol}_1min.parquet
Rate limit: 0.05s between requests (25 req/s max)
"""
from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

CACHE_DIR = Path("data/cache")

# Try V3 first, fall back to V2
UPSTOX_V3 = "https://api.upstox.com/v3/historical-candle"
UPSTOX_V2 = "https://api.upstox.com/v2/historical-candle"


def download_minute_bars(symbol: str, isin: str, token: str,
                         from_date: str, to_date: str) -> pd.DataFrame:
    """
    Download 1-minute OHLCV from Upstox API.
    Paginates by MONTH (API limit).
    Filters to NSE trading hours 09:15–15:30.
    """
    cache_file = CACHE_DIR / f"{symbol}_1min.parquet"
    if cache_file.exists():
        print(f"  {symbol}: loading from cache")
        return pd.read_parquet(cache_file)

    print(f"  {symbol}: downloading minute bars...")
    all_candles = []
    start = pd.Timestamp(from_date)
    end = pd.Timestamp(to_date)

    # Determine base URL — try V3 first
    base_url = UPSTOX_V3

    while start < end:
        chunk_end = min(start + pd.DateOffset(months=1), end)
        from_str = start.strftime('%Y-%m-%d')
        to_str = chunk_end.strftime('%Y-%m-%d')

        url = f"{base_url}/{isin}/minutes/1/{to_str}/{from_str}"
        headers = {
            'Accept': 'application/json',
            'Authorization': f'Bearer {token}',
        }

        try:
            r = requests.get(url, headers=headers, timeout=30)

            # If V3 returns 404 or other error, fall back to V2
            if r.status_code in (404, 400) and base_url == UPSTOX_V3:
                print(f"    V3 failed ({r.status_code}), falling back to V2")
                base_url = UPSTOX_V2
                url = f"{base_url}/{isin}/minutes/1/{to_str}/{from_str}"
                r = requests.get(url, headers=headers, timeout=30)

            if r.status_code != 200:
                print(f"    Warning: API {r.status_code} for {symbol} "
                      f"({from_str}→{to_str}), skipping")
                start = chunk_end
                time.sleep(0.05)
                continue

            data = r.json().get('data', {})
            candles = data.get('candles', [])
            if candles:
                all_candles.extend(candles)
                if len(all_candles) % 50000 < len(candles):
                    print(f"    {from_str}→{to_str}: "
                          f"{len(candles)} bars (total: {len(all_candles)})")

        except Exception as e:
            print(f"    Error for {symbol} ({from_str}→{to_str}): {e}")

        start = chunk_end
        time.sleep(0.05)  # Rate limit: max 25 req/s

    if not all_candles:
        print(f"  ⚠️  No data for {symbol}, skipping")
        return None

    df = pd.DataFrame(all_candles,
                      columns=['timestamp', 'open', 'high', 'low',
                               'close', 'volume', 'oi'])
    df['datetime'] = pd.to_datetime(df['timestamp'])
    df = df.drop(columns=['timestamp', 'oi'])
    df = df.drop_duplicates(subset='datetime')
    df = df.sort_values('datetime').reset_index(drop=True)

    # Filter: NSE trading hours 09:15–15:30
    hour_min = df['datetime'].dt.hour * 100 + df['datetime'].dt.minute
    df = df[(hour_min >= 915) & (hour_min <= 1530)].copy()

    # Filter: remove zero-volume bars and bad data
    df = df[df['volume'] > 0].copy()
    bad_data = (df['high'] == df['low']) & (df['volume'] < 100)
    df = df[~bad_data].copy()

    df = df.reset_index(drop=True)
    print(f"    {symbol}: {len(df)} valid minute bars")

    # Cache to parquet
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache_file, index=False)

    return df


def load_all_stocks(instruments: dict, token: str,
                    from_date: str, to_date: str,
                    max_missing_pct: float = 0.10) -> dict:
    """
    Download all stocks, align to common timestamps.

    Returns: {symbol: DataFrame with [datetime, open, high, low, close, volume]}
    """
    raw_data = {}
    for symbol, isin in instruments.items():
        df = download_minute_bars(symbol, isin, token, from_date, to_date)
        if df is not None and len(df) > 100:
            raw_data[symbol] = df

    if len(raw_data) < 2:
        raise ValueError(f"Only {len(raw_data)} stocks downloaded — need at least 2")

    # Find common timestamp universe
    all_timestamps = None
    for sym, df in raw_data.items():
        ts = set(df['datetime'].values)
        if all_timestamps is None:
            all_timestamps = ts
        else:
            all_timestamps = all_timestamps.intersection(ts)

    common_ts = sorted(all_timestamps)
    print(f"\n  Common timestamps across {len(raw_data)} stocks: {len(common_ts)}")

    # Align and forward-fill
    aligned = {}
    dropped = []
    for sym, df in raw_data.items():
        df_indexed = df.set_index('datetime')
        df_aligned = df_indexed.reindex(common_ts)

        # Forward-fill up to 3 bars for minor gaps
        missing_before = df_aligned['close'].isna().sum()
        df_aligned = df_aligned.ffill(limit=3)
        missing_after = df_aligned['close'].isna().sum()

        pct_missing = missing_after / len(df_aligned)
        if pct_missing > max_missing_pct:
            print(f"  ⚠️  {sym}: {pct_missing:.1%} missing — DROPPED")
            dropped.append(sym)
            continue

        # Drop any remaining NaN rows
        df_aligned = df_aligned.dropna()
        df_aligned = df_aligned.reset_index()
        df_aligned = df_aligned.rename(columns={'index': 'datetime'})
        aligned[sym] = df_aligned

    if dropped:
        print(f"  Dropped {len(dropped)} stocks: {dropped}")
    print(f"  Final: {len(aligned)} stocks, {len(common_ts)} bars each")

    return aligned


def build_splits(all_data: dict, config) -> tuple:
    """
    Temporal splits for walk-forward validation.

    Returns: (train_data, val_data, test_data)
    Each is a dict of {symbol: DataFrame}
    """
    train_end = pd.Timestamp(config.train_end)
    val_end = pd.Timestamp(config.val_end)

    train_data = {}
    val_data = {}
    test_data = {}

    for sym, df in all_data.items():
        dt = df['datetime']
        train_data[sym] = df[dt <= train_end].copy().reset_index(drop=True)
        val_data[sym] = df[(dt > train_end) & (dt <= val_end)].copy().reset_index(drop=True)
        test_data[sym] = df[dt > val_end].copy().reset_index(drop=True)

    # Print split sizes
    sample_sym = list(train_data.keys())[0]
    print(f"  Train: {len(train_data[sample_sym])} bars "
          f"(→ {config.train_end})")
    print(f"  Val:   {len(val_data[sample_sym])} bars "
          f"({config.train_end} → {config.val_end})")
    print(f"  Test:  {len(test_data[sample_sym])} bars "
          f"({config.val_end} →)")

    return train_data, val_data, test_data
