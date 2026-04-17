"""
Flexible loader for Kaggle-hosted NSE datasets.

Target dataset:
    https://www.kaggle.com/datasets/debashis74017/algo-trading-data-nifty-100-data-with-indicators

Supports two layouts commonly found in Kaggle finance datasets:
    (A) One CSV per symbol, file name = SYMBOL.csv
    (B) One big CSV with a 'symbol' / 'ticker' column

Column-name auto-detection handles these variants:
    date / datetime / timestamp / Date / Datetime / Timestamp
    open / Open / OPEN
    close / Close / CLOSE  (similarly for high/low/volume)

Indicators (RSI, MACD, etc.) already present in the CSV are preserved and
passed through to the model as extra features — on top of our own stationary
features. Use compute_features_with_indicators() to get the merged set.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

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


# --- layout A: one file per symbol ---------------------------------------

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

    Tries: {symbol}.csv, {symbol}.CSV, {symbol.upper()}.csv, nested subdirs.
    """
    data_dir = Path(data_dir)
    candidates = [
        data_dir / f"{symbol}.csv",
        data_dir / f"{symbol.upper()}.csv",
        data_dir / f"{symbol.lower()}.csv",
    ]
    for c in candidates:
        if c.exists():
            return load_symbol_file(c)

    # Fall back to recursive search.
    for p in data_dir.rglob(f"{symbol}.csv"):
        return load_symbol_file(p)
    for p in data_dir.rglob(f"{symbol.upper()}.csv"):
        return load_symbol_file(p)

    raise FileNotFoundError(
        f"No CSV found for symbol {symbol!r} under {data_dir}. "
        f"Available (first 10): {[p.name for p in list(data_dir.glob('*.csv'))[:10]]}"
    )


# --- layout B: one big file ----------------------------------------------

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
        # Layout A: file stems are the symbols.
        return sorted({c.stem.upper() for c in p.glob("*.csv")})
    if p.is_file():
        df = pd.read_csv(p, nrows=100000)  # cap for big files
        df = _standardize(df)
        sym_col = _resolve(df, ["symbol", "ticker", "name"])
        if sym_col is None:
            return []
        return sorted(df[sym_col].astype(str).str.upper().unique().tolist())
    return []


# --- feature merging ------------------------------------------------------

def split_ohlcv_and_indicators(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Separate a standardized DataFrame into OHLCV + pre-computed indicators."""
    ohlcv_cols = ["open", "high", "low", "close", "volume"]
    ohlcv = df[ohlcv_cols].copy()
    indicator_cols = [c for c in df.columns if c not in ohlcv_cols]
    indicators = df[indicator_cols].copy() if indicator_cols else pd.DataFrame(index=df.index)
    return ohlcv, indicators


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("Usage: python kaggle_loader.py <path> <symbol>")
        print("       python kaggle_loader.py <path> --list")
        sys.exit(1)

    if sys.argv[2] == "--list":
        print("\n".join(list_symbols(sys.argv[1])[:50]))
    else:
        df = load_kaggle_dataset(sys.argv[1], sys.argv[2])
        print(f"Loaded {len(df)} rows for {sys.argv[2]}")
        print(f"Columns: {list(df.columns)}")
        print(f"Date range: {df.index.min()} -> {df.index.max()}")
        print(df.head())
