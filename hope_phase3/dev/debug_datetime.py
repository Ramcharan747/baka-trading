#!/usr/bin/env python3
"""Diagnose the datetime type mismatch in data.py alignment."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
from data import download_minute_bars

token = os.environ.get('UPSTOX_TOKEN', '')
if not token:
    print("Set UPSTOX_TOKEN first")
    sys.exit(1)

# Download just 1 stock to inspect types
from instruments import INSTRUMENTS
isin = INSTRUMENTS['RELIANCE']
df = download_minute_bars('RELIANCE', isin, token, '2024-01-01', '2024-01-31')

print(f"\n=== DIAGNOSTIC ===")
print(f"df['datetime'] dtype: {df['datetime'].dtype}")
print(f"df['datetime'].dt.tz: {df['datetime'].dt.tz}")
print(f"Sample values:")
for i in [0, 1, 2]:
    v = df['datetime'].iloc[i]
    print(f"  [{i}] value={v!r}  type={type(v).__name__}")

print(f"\n--- As .values (numpy) ---")
vals = df['datetime'].values
print(f"dtype: {vals.dtype}")
print(f"Sample: {vals[0]!r}  type={type(vals[0]).__name__}")

print(f"\n--- pd.to_datetime(df['datetime']).values ---")
vals2 = pd.to_datetime(df['datetime']).values
print(f"dtype: {vals2.dtype}")
print(f"Sample: {vals2[0]!r}  type={type(vals2[0]).__name__}")

print(f"\n--- set() test ---")
s = set(pd.to_datetime(df['datetime']).values)
sample = list(s)[:3]
for v in sample:
    print(f"  value={v!r}  type={type(v).__name__}")

print(f"\n--- Reindex test ---")
df_idx = df.set_index(pd.to_datetime(df['datetime']))
print(f"Index dtype: {df_idx.index.dtype}")
print(f"Index tz: {df_idx.index.tz}")

common = pd.DatetimeIndex(sorted(s))
print(f"Common dtype: {common.dtype}")
print(f"Common tz: {common.tz}")

reindexed = df_idx.reindex(common)
missing = reindexed['close'].isna().sum()
print(f"After reindex: {missing}/{len(reindexed)} missing ({missing/len(reindexed):.1%})")

# Try stripping tz
print(f"\n--- Fix: strip tz ---")
df['datetime_naive'] = pd.to_datetime(df['datetime'])
if df['datetime_naive'].dt.tz is not None:
    df['datetime_naive'] = df['datetime_naive'].dt.tz_localize(None)
    print(f"Stripped tz → {df['datetime_naive'].dtype}")
else:
    print(f"Already tz-naive: {df['datetime_naive'].dtype}")

df_idx2 = df.set_index('datetime_naive')
vals_naive = df['datetime_naive'].values
s2 = set(vals_naive)
common2 = pd.DatetimeIndex(sorted(s2))
reindexed2 = df_idx2.reindex(common2)
missing2 = reindexed2['close'].isna().sum()
print(f"After reindex (tz-naive): {missing2}/{len(reindexed2)} missing ({missing2/len(reindexed2):.1%})")
