"""
Data pipeline: Upstox API → parquet cache → features → training tensors.

Caching strategy:
- Raw OHLCV  → /content/data/{symbol}_daily.parquet
- Features   → /content/data/{symbol}_features.parquet
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import requests
import torch

from features import compute_features
from labels import compute_labels
from ic_test import ic_test

UPSTOX_BASE = "https://api.upstox.com/v2/historical-candle"

INSTRUMENTS = {
    'COALINDIA':  'NSE_EQ|INE522F01014',
    'RELIANCE':   'NSE_EQ|INE002A01018',
    'TCS':        'NSE_EQ|INE467B01029',
    'INFY':       'NSE_EQ|INE009A01021',
    'HDFCBANK':   'NSE_EQ|INE040A01034',
    'ICICIBANK':  'NSE_EQ|INE090A01021',
    'SBIN':       'NSE_EQ|INE062A01020',
    'WIPRO':      'NSE_EQ|INE075A01022',
}


def download_daily(symbol: str, instrument_key: str, token: str = None,
                   from_date: str = "2010-01-01") -> pd.DataFrame:
    """
    Download daily OHLCV from Upstox V2 API.
    Paginates in 1-year chunks (API limit per request).
    No auth needed for daily EOD data.
    """
    all_candles = []
    start = pd.Timestamp(from_date)
    end = pd.Timestamp.today()

    # Paginate in 1-year chunks
    while start < end:
        chunk_end = min(start + pd.DateOffset(years=1), end)
        from_str = start.strftime('%Y-%m-%d')
        to_str = chunk_end.strftime('%Y-%m-%d')

        url = f"{UPSTOX_BASE}/{instrument_key}/day/{to_str}/{from_str}"

        headers = {'Accept': 'application/json'}
        if token:
            headers['Authorization'] = f'Bearer {token}'

        r = requests.get(url, headers=headers)
        if r.status_code != 200:
            print(f"    Warning: API error {r.status_code} for {symbol} "
                  f"({from_str} to {to_str}), skipping chunk")
            start = chunk_end
            continue

        data = r.json().get('data', {})
        candles = data.get('candles', [])
        if candles:
            all_candles.extend(candles)
            print(f"    {from_str}→{to_str}: {len(candles)} bars")

        start = chunk_end

    if not all_candles:
        raise ValueError(f"No data downloaded for {symbol}")

    df = pd.DataFrame(all_candles,
                      columns=['timestamp', 'open', 'high', 'low',
                               'close', 'volume', 'oi'])
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.drop_duplicates(subset='timestamp')
    df = df.set_index('timestamp').sort_index()
    df = df[['open', 'high', 'low', 'close', 'volume']]
    return df


def prepare_dataset(symbols: dict, token: str = None,
                    data_dir: Path = Path("/content/data"),
                    lookahead: int = 5,
                    cost_bps: float = 3.0) -> dict:
    """
    For each symbol: download → compute features → compute labels → save.
    Returns dict of {symbol: (features_df, labels_series)}.
    """
    data_dir.mkdir(parents=True, exist_ok=True)
    dataset = {}

    for symbol, key in symbols.items():
        raw_file = data_dir / f"{symbol}_daily.parquet"
        feat_file = data_dir / f"{symbol}_features.parquet"
        label_file = data_dir / f"{symbol}_labels.parquet"

        # Load from cache if available
        if feat_file.exists() and label_file.exists():
            print(f"  {symbol}: loading from cache")
            features = pd.read_parquet(feat_file)
            labels = pd.read_parquet(label_file).squeeze()
        else:
            print(f"  {symbol}: downloading and processing...")

            # Download (or load raw cache)
            if raw_file.exists():
                raw = pd.read_parquet(raw_file)
                print(f"    {len(raw)} daily bars (cached)")
            else:
                raw = download_daily(symbol, key, token)
                raw.to_parquet(raw_file)
                print(f"    {len(raw)} daily bars (downloaded)")

            # Features
            features = compute_features(raw)

            # Labels (aligned with features)
            labels_raw = compute_labels(raw, lookahead=lookahead,
                                        cost_bps=cost_bps)
            labels = labels_raw.loc[features.index]

            # Align: drop rows where either is NaN
            valid = features.notna().all(axis=1) & labels.notna()
            features = features[valid]
            labels = labels[valid]

            # Save
            features.to_parquet(feat_file)
            labels.to_frame('label').to_parquet(label_file)
            print(f"    {len(features)} valid bars saved")

        # Run IC test (diagnostic only — keep all features)
        # With ~1100 daily bars, per-stock IC filtering is too aggressive.
        # The intersection across 8 stocks would be empty.
        print(f"  {symbol}: IC test (diagnostic)...")
        ic_test(features, labels, min_ic=0.003)

        # Drop time_sin/time_cos if constant (daily data has no intraday time)
        for col in ['time_sin', 'time_cos']:
            if col in features.columns and features[col].nunique() <= 1:
                features = features.drop(columns=[col])

        dataset[symbol] = (features, labels)

    return dataset


def build_training_batches(dataset: dict, train_frac: float = 0.7,
                           val_frac: float = 0.15):
    """
    Split each stock temporally. Stack into tensors for batched training.

    Returns:
        feat_tensor:      [n_stocks, T_train, n_features]
        lab_tensor:       [n_stocks, T_train]
        val_data:         dict of {symbol: (features_df, labels_series)}
        test_data:        dict of {symbol: (features_df, labels_series)}
        common_features:  list of feature names used
    """
    symbols = list(dataset.keys())

    train_data = {}
    val_data = {}
    test_data = {}

    for sym, (feat, lab) in dataset.items():
        T = len(feat)
        t1 = int(T * train_frac)
        t2 = int(T * (train_frac + val_frac))

        train_data[sym] = (feat.iloc[:t1], lab.iloc[:t1])
        val_data[sym] = (feat.iloc[t1:t2], lab.iloc[t1:t2])
        test_data[sym] = (feat.iloc[t2:], lab.iloc[t2:])

    # Ensure all stocks have same feature set (intersection)
    all_feat_sets = [set(v[0].columns) for v in train_data.values()]
    common_features = sorted(
        all_feat_sets[0].intersection(*all_feat_sets[1:])
    )
    print(f"  Common features across all stocks: {len(common_features)}")

    for sym in train_data:
        f, l = train_data[sym]
        train_data[sym] = (f[common_features], l)
    for sym in val_data:
        f, l = val_data[sym]
        val_data[sym] = (f[common_features], l)
    for sym in test_data:
        f, l = test_data[sym]
        test_data[sym] = (f[common_features], l)

    # Trim all stocks to same length (min length)
    min_train = min(len(v[0]) for v in train_data.values())
    for sym in train_data:
        f, l = train_data[sym]
        train_data[sym] = (f.iloc[:min_train], l.iloc[:min_train])

    # Stack into tensors
    feat_tensor = torch.tensor(
        np.stack([v[0].values for v in train_data.values()], axis=0),
        dtype=torch.float32,
    )
    lab_tensor = torch.tensor(
        np.stack([v[1].values for v in train_data.values()], axis=0),
        dtype=torch.float32,
    )

    print(f"  Training tensor: {feat_tensor.shape}")
    # feat_tensor: [n_stocks, T, n_features]
    # lab_tensor:  [n_stocks, T]

    return feat_tensor, lab_tensor, val_data, test_data, common_features
