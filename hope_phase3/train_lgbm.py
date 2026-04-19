"""
LightGBM training: cross-sectional approach.

Stacks all stocks' features into a flat matrix and trains LightGBM.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from models.lgbm_baseline import LGBMBaseline


def train_lgbm(train_features: dict, train_labels: dict,
               val_features: dict, val_labels: dict,
               feature_names: list) -> LGBMBaseline:
    """
    Train LightGBM on all stocks stacked cross-sectionally.

    Args:
        train_features: {symbol: DataFrame[70 cols]}
        train_labels: {symbol: Series}
        val_features: same format
        val_labels: same format
        feature_names: list of 70 feature names

    Returns: fitted LGBMBaseline
    """
    # Stack all stocks into flat arrays
    X_train_parts, y_train_parts = [], []
    for sym in sorted(train_features.keys()):
        feat = train_features[sym][feature_names].values
        lab = train_labels[sym].values
        valid = ~np.isnan(lab)
        X_train_parts.append(feat[valid])
        y_train_parts.append(lab[valid])

    X_train = np.concatenate(X_train_parts, axis=0)
    y_train = np.concatenate(y_train_parts, axis=0)

    X_val_parts, y_val_parts = [], []
    for sym in sorted(val_features.keys()):
        feat = val_features[sym][feature_names].values
        lab = val_labels[sym].values
        valid = ~np.isnan(lab)
        X_val_parts.append(feat[valid])
        y_val_parts.append(lab[valid])

    X_val = np.concatenate(X_val_parts, axis=0)
    y_val = np.concatenate(y_val_parts, axis=0)

    print(f"  LightGBM training: {X_train.shape[0]:,} samples, "
          f"val: {X_val.shape[0]:,} samples")

    model = LGBMBaseline()
    model.fit(X_train, y_train, X_val, y_val)

    return model
