"""
Flexible loader for Kaggle-hosted NSE datasets.

Target dataset:
    https://www.kaggle.com/datasets/debashis74017/algo-trading-data-nifty-100-data-with-indicators

Actual dataset structure (verified by download):
    - One CSV per symbol, named {SYMBOL}_minute.csv
    - 6 columns: date, open, high, low, close, volume
    - 1-minute bars from 2015-02-02 to 2026-04-08
    - ~1M rows per symbol
    - **No pre-computed indicators despite the dataset name**

This module handles:
    (A) Per-symbol CSVs with _minute.csv suffix (actual layout)
    (B) Per-symbol CSVs without suffix (fallback)
    (C) One big CSV with a 'symbol' column (alternative Kaggle layouts)

Since the dataset has no indicators, compute_indicators() derives them
from raw OHLCV: RSI, MACD, EMA20, Bollinger Bands, ATR, OBV.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


# --- column-name resolution -----------------------------------------------

_DATE_COLS = ["datetime", "date", "timestamp", "time"]
_OHLCV_MAP = {
    "open": ["open", "o"],
    "high": ["high", "h"],
    "low": ["low", "l"],
    "close": ["close", "c", "adj_close", "adj close"],
    "volume": ["volume", "vol", "v"],
}


def _resolve(df: pd.DataFrame, candidates: list[str]) -> Optional[str]:
    lower = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand in lower:
            return lower[cand]
    return None


def _standardize(df: pd.DataFrame) -> pd.DataFrame:
    """Rename columns to canonical {open, high, low, close, volume} and set DatetimeIndex."""
    rename = {}
    for canonical, aliases in _OHLCV_MAP.items():
        src = _resolve(df, aliases)
        if src is not None and src != canonical:
            rename[src] = canonical
    if rename:
        df = df.rename(columns=rename)

    # Date column -> index
    date_src = _resolve(df, _DATE_COLS)
    if date_src is not None:
        df[date_src] = pd.to_datetime(df[date_src], errors="coerce")
        df = df.dropna(subset=[date_src]).set_index(date_src).sort_index()
        df.index.name = "datetime"

    return df


# --- kagglehub download ---------------------------------------------------

def download_dataset(
    kaggle_key: str | None = None,
) -> str:
    """
    Download the Nifty dataset via kagglehub and return the local path.

    Args:
        kaggle_key: Kaggle API key (KGAT_...). If None, uses env or ~/.kaggle/kaggle.json.

    Returns:
        Absolute path to the dataset directory.
    """
    import os
    if kaggle_key:
        os.environ["KAGGLE_KEY"] = kaggle_key

    try:
        import kagglehub
        path = kagglehub.dataset_download(
            "debashis74017/algo-trading-data-nifty-100-data-with-indicators"
        )
        return str(path)
    except ImportError:
        raise ImportError(
            "kagglehub not installed. Run: pip install kagglehub"
        )


# --- layout A: one file per symbol ----------------------------------------

def load_symbol_file(path: Path) -> pd.DataFrame:
    """Load a single CSV file (layout A)."""
    df = pd.read_csv(path)
    df = _standardize(df)
    # Ensure OHLCV exists — skip/raise if the file is wrong shape.
    required = {"open", "high", "low", "close", "volume"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path.name}: missing columns {missing}; got {list(df.columns)}")
    return df


def load_directory(data_dir: str | Path, symbol: str) -> pd.DataFrame:
    """
    Load a single symbol from a directory of per-symbol CSVs.

    Tries (in order):
        {symbol}_minute.csv          (actual Kaggle Nifty 500 layout)
        {symbol}.csv                 (simple layout)
        Case-insensitive variants
        Recursive search
    """
    data_dir = Path(data_dir)
    sym_upper = symbol.upper()
    sym_lower = symbol.lower()

    candidates = [
        # Actual Kaggle dataset naming: SYMBOL_minute.csv
        data_dir / f"{sym_upper}_minute.csv",
        data_dir / f"{symbol}_minute.csv",
        data_dir / f"{sym_lower}_minute.csv",
        # Plain naming
        data_dir / f"{symbol}.csv",
        data_dir / f"{sym_upper}.csv",
        data_dir / f"{sym_lower}.csv",
    ]
    for c in candidates:
        if c.exists():
            return load_symbol_file(c)

    # Fall back to recursive search.
    for pattern in [f"{sym_upper}_minute.csv", f"{symbol}_minute.csv",
                    f"{symbol}.csv", f"{sym_upper}.csv"]:
        for p in data_dir.rglob(pattern):
            return load_symbol_file(p)

    # List what's actually available to help debugging.
    available = sorted([p.name for p in data_dir.glob("*.csv")][:15])
    raise FileNotFoundError(
        f"No CSV found for symbol {symbol!r} under {data_dir}. "
        f"Available (first 15): {available}"
    )


# --- layout B: one big file -----------------------------------------------

def load_combined_file(
    path: str | Path,
    symbol: str,
    symbol_col_candidates: list[str] = ["symbol", "ticker", "name"],
) -> pd.DataFrame:
    """Slice a big multi-symbol CSV down to a single symbol."""
    df = pd.read_csv(path)
    df = _standardize(df)
    sym_col = _resolve(df, symbol_col_candidates)
    if sym_col is None:
        raise ValueError(f"No symbol column found in {path}; cols={list(df.columns)}")
    mask = df[sym_col].astype(str).str.upper() == symbol.upper()
    out = df.loc[mask].copy()
    if out.empty:
        uniq = df[sym_col].astype(str).unique()[:10]
        raise ValueError(f"Symbol {symbol!r} not found; first 10 available: {list(uniq)}")
    return out.drop(columns=[sym_col])


# --- smart wrapper --------------------------------------------------------

def load_kaggle_dataset(
    path: str | Path,
    symbol: str,
) -> pd.DataFrame:
    """
    Auto-detect layout A vs. B.

    path: either a CSV file (layout B) or a directory (layout A).
    Returns a standardized OHLCV DataFrame with DatetimeIndex and any extra
    indicator columns preserved.
    """
    p = Path(path)
    if p.is_file() and p.suffix.lower() == ".csv":
        return load_combined_file(p, symbol)
    if p.is_dir():
        # Prefer layout-A per-symbol files; fall back to scanning for a big combined file.
        try:
            return load_directory(p, symbol)
        except FileNotFoundError:
            pass
        # Scan for a plausible combined file.
        csvs = list(p.glob("*.csv"))
        for c in csvs:
            try:
                return load_combined_file(c, symbol)
            except Exception:
                continue
        raise FileNotFoundError(
            f"Could not find {symbol!r} under {p} "
            f"(tried layout A + scanning {len(csvs)} CSVs for layout B)"
        )
    raise FileNotFoundError(f"{path} is neither a CSV nor a directory")


def list_symbols(path: str | Path) -> list[str]:
    """Return the list of available symbols in a Kaggle dataset path."""
    p = Path(path)
    if p.is_dir():
        symbols = set()
        for c in p.glob("*.csv"):
            stem = c.stem
            # Strip _minute / _minute_new suffixes
            for suffix in ("_minute_new", "_minute"):
                if stem.endswith(suffix):
                    stem = stem[: -len(suffix)]
                    break
            symbols.add(stem.upper())
        return sorted(symbols)
    if p.is_file():
        df = pd.read_csv(p, nrows=100000)  # cap for big files
        df = _standardize(df)
        sym_col = _resolve(df, ["symbol", "ticker", "name"])
        if sym_col is None:
            return []
        return sorted(df[sym_col].astype(str).str.upper().unique().tolist())
    return []


# --- computed indicators --------------------------------------------------

def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute standard technical indicators from raw OHLCV data.

    The actual Kaggle dataset has NO pre-computed indicators (despite the
    dataset name), so we compute them here rather than reading them from CSV.

    Returns a DataFrame with indicator columns only (same index as input).
    """
    ind = pd.DataFrame(index=df.index)
    close = df["close"].astype(float)
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    volume = df["volume"].astype(float)

    # --- RSI (14-period) ---
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.ewm(span=14, adjust=False).mean()
    avg_loss = loss.ewm(span=14, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    ind["rsi_14"] = 100 - (100 / (1 + rs))

    # --- MACD (12, 26, 9) ---
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    ind["macd"] = ema12 - ema26
    ind["macd_signal"] = ind["macd"].ewm(span=9, adjust=False).mean()
    ind["macd_hist"] = ind["macd"] - ind["macd_signal"]

    # --- EMA 20 (distance from close as %) ---
    ema20 = close.ewm(span=20, adjust=False).mean()
    ind["ema20_pct"] = (close - ema20) / (ema20 + 1e-10)

    # --- Bollinger Bands (20, 2) ---
    sma20 = close.rolling(20).mean()
    std20 = close.rolling(20).std()
    upper = sma20 + 2 * std20
    lower = sma20 - 2 * std20
    ind["bb_position"] = (close - lower) / (upper - lower + 1e-10)
    ind["bb_width"] = (upper - lower) / (sma20 + 1e-10)

    # --- ATR (14-period) ---
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    ind["atr_14"] = tr.ewm(span=14, adjust=False).mean()
    # Normalize ATR as percentage of close
    ind["atr_14_pct"] = ind["atr_14"] / (close + 1e-10)

    # --- OBV (On-Balance Volume, normalized) ---
    obv_sign = np.sign(close.diff()).fillna(0)
    obv = (obv_sign * volume).cumsum()
    obv_ma = obv.rolling(20).mean()
    ind["obv_norm"] = (obv - obv_ma) / (obv_ma.abs() + 1e-10)

    # --- VWAP approximation (intraday) ---
    typical = (high + low + close) / 3
    cum_tp_vol = (typical * volume).cumsum()
    cum_vol = volume.cumsum()
    vwap = cum_tp_vol / (cum_vol + 1e-10)
    ind["vwap_pct"] = (close - vwap) / (vwap + 1e-10)

    return ind


# --- feature merging -------------------------------------------------------

def split_ohlcv_and_indicators(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Separate a standardized DataFrame into OHLCV + any extra columns."""
    ohlcv_cols = ["open", "high", "low", "close", "volume"]
    ohlcv = df[[c for c in ohlcv_cols if c in df.columns]].copy()
    indicator_cols = [c for c in df.columns if c not in ohlcv_cols]
    indicators = df[indicator_cols].copy() if indicator_cols else pd.DataFrame(index=df.index)
    return ohlcv, indicators


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("Usage: python kaggle_loader.py <path> <symbol>")
        print("       python kaggle_loader.py <path> --list")
        print("       python kaggle_loader.py --download")
        sys.exit(1)

    if sys.argv[1] == "--download":
        path = download_dataset()
        print(f"Dataset downloaded to: {path}")
        sys.exit(0)

    if sys.argv[2] == "--list":
        symbols = list_symbols(sys.argv[1])
        print(f"Found {len(symbols)} symbols:")
        print("\n".join(symbols[:50]))
    else:
        df = load_kaggle_dataset(sys.argv[1], sys.argv[2])
        print(f"Loaded {len(df):,} rows for {sys.argv[2]}")
        print(f"Columns: {list(df.columns)}")
        print(f"Date range: {df.index.min()} -> {df.index.max()}")
        print(df.head())

        # Show computed indicators
        print(f"\n--- Computed indicators ---")
        ind = compute_indicators(df)
        print(f"Indicator columns ({len(ind.columns)}): {list(ind.columns)}")
        print(ind.dropna().head())
