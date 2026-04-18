"""
IC test — Spearman rank correlation between each feature and the label.
Drop features below threshold before training.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def ic_test(features: pd.DataFrame, labels: pd.Series,
            min_ic: float = 0.005) -> list:
    """
    Spearman IC between each feature and the label.
    For 1-minute data with 5-bar lookahead, IC > 0.005 is already meaningful.
    Drop features below min_ic threshold.

    Returns: list of feature names that pass the IC test.
    """
    results = {}
    for col in features.columns:
        aligned = features[col].dropna().align(labels.dropna(), join='inner')
        if len(aligned[0]) < 100:
            continue
        ic, p = spearmanr(aligned[0], aligned[1])
        if np.isnan(ic):
            ic = 0.0
            p = 1.0
        results[col] = {'IC': ic, 'p': p, 'keep': abs(ic) > min_ic and p < 0.05}

    df_results = pd.DataFrame(results).T.sort_values('IC', key=abs, ascending=False)
    print(df_results[['IC', 'p', 'keep']].to_string())

    good = [k for k, v in results.items() if v['keep']]
    print(f"\nKeeping {len(good)}/{len(features.columns)} features")
    return good
