"""
Cross-sectional LightGBM baseline.

Unlike HOPE/LSTM which process each stock independently,
LightGBM sees ALL stocks at once and learns which features
predict cross-sectional outperformance.
This is the standard hedge fund approach for daily alpha.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

try:
    import lightgbm as lgb
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False


class LGBMBaseline:
    def __init__(self, params=None):
        if not HAS_LGBM:
            raise ImportError("lightgbm not installed: pip install lightgbm")

        self.params = params or {
            'objective': 'regression',
            'metric': 'mse',
            'n_estimators': 500,
            'learning_rate': 0.05,
            'max_depth': 6,
            'num_leaves': 31,
            'min_child_samples': 50,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'reg_alpha': 0.1,
            'reg_lambda': 0.1,
            'verbose': -1,
        }
        self.model = None

    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
            X_val: np.ndarray, y_val: np.ndarray):
        """
        X_train: [n_samples, 70] — ALL stocks stacked
        y_train: [n_samples] — labels
        """
        dtrain = lgb.Dataset(X_train, label=y_train)
        dval = lgb.Dataset(X_val, label=y_val)
        self.model = lgb.train(
            self.params,
            dtrain,
            valid_sets=[dval],
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)],
        )

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model not fitted yet")
        return self.model.predict(X)

    def feature_importance(self, feature_names):
        imp = self.model.feature_importance(importance_type='gain')
        return pd.Series(imp, index=feature_names).sort_values(ascending=False)
