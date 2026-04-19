"""
IC diagnostic: per-feature Spearman rank correlation with label.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def ic_test(features: pd.DataFrame, labels: pd.Series,
            threshold: float = 0.05) -> pd.DataFrame:
    """
    Compute Spearman IC for each feature against the label.

    Returns DataFrame with columns [IC, p, keep] sorted by |IC| descending.
    keep = True if p < threshold.
    """
    valid = labels.notna()
    labels_valid = labels[valid]

    results = []
    for col in features.columns:
        feat = features[col][valid]
        aligned = pd.concat([feat, labels_valid], axis=1).dropna()
        if len(aligned) < 30:
            results.append({'feature': col, 'IC': 0.0, 'p': 1.0, 'keep': False})
            continue

        ic, p = spearmanr(aligned.iloc[:, 0], aligned.iloc[:, 1])
        if np.isnan(ic):
            ic, p = 0.0, 1.0
        results.append({
            'feature': col,
            'IC': round(ic, 6),
            'p': round(p, 6),
            'keep': p < threshold,
        })

    df = pd.DataFrame(results).set_index('feature')
    df = df.sort_values('IC', key=abs, ascending=False)
    return df
