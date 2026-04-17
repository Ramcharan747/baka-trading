"""
Download NSE / NIFTY data and cache to parquet.

Sources (in priority order):
    1. yfinance  — daily + limited-range intraday (free, no auth)
    2. nsepy     — NSE official daily data (free, sometimes rate-limited)
    3. Kite API  — full 1-min bars and tick data (requires Zerodha credentials)

Usage:
    python data_download.py --symbol NIFTY --start 2022-01-01 --end 2024-12-31
    python data_download.py --symbol RELIANCE --interval 1d
"""
from __future__ import annotations

import argparse
import os
from datetime import date, datetime
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------- yfinance

def _download_yf(symbol: str, start: str, end: str, interval: str) -> pd.DataFrame:
    import yfinance as yf
    # yfinance NSE symbols: "^NSEI" for NIFTY index, "RELIANCE.NS" for equities
    ticker = "^NSEI" if symbol.upper() == "NIFTY" else f"{symbol.upper()}.NS"
    df = yf.download(
        ticker,
        start=start,
        end=end,
        interval=interval,
        progress=False,
        auto_adjust=False,
    )
    if df.empty:
        raise RuntimeError(f"yfinance returned empty for {ticker} {start}->{end} {interval}")
    # yfinance 0.2.x returns a MultiIndex — flatten it.
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.columns = [c.lower().replace(" ", "_") for c in df.columns]
    return df[["open", "high", "low", "close", "volume"]].dropna()


# ---------------------------------------------------------------- nsepy

def _download_nsepy(symbol: str, start: str, end: str) -> pd.DataFrame:
    from nsepy import get_history
    start_d = datetime.strptime(start, "%Y-%m-%d").date()
    end_d = datetime.strptime(end, "%Y-%m-%d").date()
    is_index = symbol.upper() in {"NIFTY", "NIFTY 50", "BANKNIFTY", "NIFTY BANK"}
    df = get_history(
        symbol="NIFTY 50" if symbol.upper() == "NIFTY" else symbol.upper(),
        start=start_d,
        end=end_d,
        index=is_index,
    )
    if df.empty:
        raise RuntimeError(f"nsepy returned empty for {symbol} {start}->{end}")
    df.columns = [c.lower().replace(" ", "_") for c in df.columns]
    keep = ["open", "high", "low", "close", "volume"]
    return df[[c for c in keep if c in df.columns]].dropna()


# ---------------------------------------------------------------- Kite (optional)

def _download_kite(symbol: str, start: str, end: str, interval: str) -> pd.DataFrame:
    """
    Full intraday 1-min bars. Requires KITE_API_KEY and KITE_ACCESS_TOKEN env vars,
    and the instrument token for the symbol. Left as a thin wrapper — fill in
    your own token map for the symbols you care about.
    """
    from kiteconnect import KiteConnect  # type: ignore

    api_key = os.environ["KITE_API_KEY"]
    access_token = os.environ["KITE_ACCESS_TOKEN"]
    kite = KiteConnect(api_key=api_key)
    kite.set_access_token(access_token)

    token_map = {"NIFTY": 256265, "BANKNIFTY": 260105}
    instrument_token = token_map.get(symbol.upper())
    if instrument_token is None:
        raise RuntimeError(
            f"Add instrument token for {symbol} to token_map in data_download.py"
        )

    records = kite.historical_data(
        instrument_token=instrument_token,
        from_date=start,
        to_date=end,
        interval=interval,
    )
    df = pd.DataFrame(records).set_index("date")
    df.columns = [c.lower() for c in df.columns]
    return df[["open", "high", "low", "close", "volume"]].dropna()


# ---------------------------------------------------------------- driver

def download(
    symbol: str,
    start: str,
    end: str,
    interval: str = "1d",
    source: str = "auto",
) -> pd.DataFrame:
    """
    Resolve a data source and return a cleaned OHLCV DataFrame with DatetimeIndex.

    source = "auto" tries yfinance first, falls back to nsepy for daily bars.
    """
    tried: list[tuple[str, str]] = []
    order = [source] if source != "auto" else ["yfinance", "nsepy"]

    for src in order:
        try:
            if src == "yfinance":
                return _download_yf(symbol, start, end, interval)
            elif src == "nsepy" and interval == "1d":
                return _download_nsepy(symbol, start, end)
            elif src == "kite":
                return _download_kite(symbol, start, end, interval)
            else:
                tried.append((src, f"unsupported interval {interval} for {src}"))
        except Exception as e:
            tried.append((src, repr(e)))

    raise RuntimeError(f"All sources failed for {symbol}: {tried}")


def cache_path(symbol: str, interval: str) -> Path:
    return DATA_DIR / f"{symbol.upper()}_{interval}.parquet"


def load_or_download(
    symbol: str,
    start: str,
    end: str,
    interval: str = "1d",
    source: str = "auto",
    refresh: bool = False,
) -> pd.DataFrame:
    path = cache_path(symbol, interval)
    if path.exists() and not refresh:
        df = pd.read_parquet(path)
        # Slice to requested window if cache covers it.
        df = df.loc[start:end]
        if len(df) > 0:
            return df
    df = download(symbol, start, end, interval, source)
    df.to_parquet(path)
    return df


def _main():
    p = argparse.ArgumentParser()
    p.add_argument("--symbol", default="NIFTY")
    p.add_argument("--start", default="2022-01-01")
    p.add_argument("--end", default=str(date.today()))
    p.add_argument("--interval", default="1d")
    p.add_argument("--source", default="auto", choices=["auto", "yfinance", "nsepy", "kite"])
    p.add_argument("--refresh", action="store_true")
    args = p.parse_args()

    df = load_or_download(
        args.symbol, args.start, args.end, args.interval, args.source, args.refresh
    )
    path = cache_path(args.symbol, args.interval)
    print(f"Wrote {len(df):>6} rows to {path}")
    print(df.head())
    print("...")
    print(df.tail())


if __name__ == "__main__":
    _main()
