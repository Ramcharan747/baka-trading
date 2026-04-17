"""
Information Coefficient (IC), lookahead-bias, and regime-conditioned
IC diagnostics.

Acceptance thresholds (from CLAUDE.md):
    - |IC| > 0.02  AND  p < 0.05  -> keep the feature
    - |IC| > 0.30                   -> lookahead bias, reject & investigate
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


@dataclass
class ICResult:
    name: str
    ic: float
    p_value: float
    keep: bool


def ic_test(
    features: pd.DataFrame,
    labels: pd.Series,
    ic_threshold: float = 0.02,
    p_threshold: float = 0.05,
    verbose: bool = True,
) -> tuple[list[str], pd.DataFrame]:
    """
    Spearman IC of each feature vs. forward return.
    Returns (kept_feature_names, full_results_df).
    """
    rows: list[ICResult] = []
    for col in features.columns:
        s = features[col].dropna()
        y = labels.reindex(s.index).dropna()
        s = s.reindex(y.index)
        if len(s) < 30:
            rows.append(ICResult(col, np.nan, np.nan, False))
            continue
        ic, p = spearmanr(s, y)
        keep = (abs(ic) > ic_threshold) and (p < p_threshold)
        rows.append(ICResult(col, float(ic), float(p), bool(keep)))

    df = pd.DataFrame([r.__dict__ for r in rows]).sort_values("ic", ascending=False)
    if verbose:
        print(df.to_string(index=False))
        kept = df[df["keep"]]["name"].tolist()
        print(f"\nKeep {len(kept)}/{len(df)} features: {kept}")
    return df[df["keep"]]["name"].tolist(), df


def check_lookahead_bias(
    feature: pd.Series,
    future_return: pd.Series,
    panic_threshold: float = 0.30,
) -> tuple[float, float]:
    """
    If |IC| > panic_threshold the feature is almost certainly using future
    data. Legitimate intraday features on 5-bar forward returns rarely
    exceed 0.05.
    """
    f = feature.dropna()
    y = future_return.reindex(f.index).dropna()
    f = f.reindex(y.index)
    ic, p = spearmanr(f, y)
    if abs(ic) > panic_threshold:
        raise ValueError(
            f"LOOKAHEAD BIAS DETECTED on {feature.name!r}: "
            f"IC={ic:.3f}. Fix your feature pipeline."
        )
    return float(ic), float(p)


def check_all_features_for_lookahead(
    features: pd.DataFrame, labels: pd.Series
) -> None:
    """Sanity-check every feature. Raises on the first offender."""
    for col in features.columns:
        check_lookahead_bias(features[col].rename(col), labels)


def regime_ic_analysis(
    features: pd.DataFrame,
    labels: pd.Series,
    prices: pd.Series,
    feature_subset: Iterable[str] | None = None,
) -> pd.DataFrame:
    """
    Compute IC separately for bull / bear / high-volatility regimes.
    A feature that only works in one regime is not robust.
    Target: |IC| > 0.02 in ALL three regimes.
    """
    returns = prices.pct_change()
    trend = returns.rolling(60).mean()
    vol = returns.rolling(20).std()

    bull = trend > trend.quantile(0.6)
    bear = trend < trend.quantile(0.4)
    high_v = vol > vol.quantile(0.7)
    regimes = {"bull": bull, "bear": bear, "high_vol": high_v}

    cols = list(feature_subset) if feature_subset is not None else list(features.columns)
    out_rows = []
    for col in cols:
        row = {"feature": col}
        for name, mask in regimes.items():
            m = mask.reindex(features.index).fillna(False)
            f = features.loc[m, col].dropna()
            y = labels.reindex(f.index).dropna()
            f = f.reindex(y.index)
            if len(f) < 100:
                row[f"ic_{name}"] = np.nan
                row[f"p_{name}"] = np.nan
                continue
            ic, p = spearmanr(f, y)
            row[f"ic_{name}"] = float(ic)
            row[f"p_{name}"] = float(p)
        out_rows.append(row)

    out = pd.DataFrame(out_rows)
    print("\n=== Regime-conditioned IC ===")
    print(out.to_string(index=False))
    return out


def rolling_ic(
    feature: pd.Series,
    labels: pd.Series,
    window: int = 500,
) -> pd.Series:
    """Rolling Spearman IC — useful to spot IC decay over time."""
    aligned = pd.concat([feature, labels], axis=1).dropna()

    def _spear(block: np.ndarray) -> float:
        if block.shape[0] < 20:
            return np.nan
        ic, _ = spearmanr(block[:, 0], block[:, 1])
        return ic

    arr = aligned.to_numpy()
    n = arr.shape[0]
    out = np.full(n, np.nan)
    for i in range(window, n + 1):
        out[i - 1] = _spear(arr[i - window : i])
    return pd.Series(out, index=aligned.index, name=f"rolling_ic_{window}")


if __name__ == "__main__":
    from features import compute_features, make_labels

    rng = np.random.default_rng(0)
    idx = pd.date_range("2023-01-01", periods=2000, freq="1min")
    price = 100 + np.cumsum(rng.normal(0, 0.1, 2000))
    df = pd.DataFrame(
        {
            "open": price,
            "high": price + 0.05,
            "low": price - 0.05,
            "close": price,
            "volume": rng.integers(1000, 5000, 2000),
        },
        index=idx,
    )
    feats = compute_features(df)
    labels = make_labels(df, lookahead=5).reindex(feats.index).dropna()
    feats = feats.reindex(labels.index)
    ic_test(feats, labels)
