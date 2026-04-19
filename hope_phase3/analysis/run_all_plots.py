"""Run all Phase 3 analysis plots from saved results."""
from __future__ import annotations

import os
import pickle
import numpy as np

from .plot_training import plot_training_curves
from .plot_features import plot_feature_ic_heatmap, plot_feature_importance
from .plot_pnl import plot_cumulative_pnl
from .plot_memory import plot_hope_memory_trajectory
from .plot_signals import (plot_prediction_quality, plot_trade_analysis,
                           plot_signal_autocorrelation)
from .plot_comparison import plot_per_stock_sharpe, plot_regime_performance


def run_all(results_path: str = "outputs/results.pkl",
            output_dir: str = "outputs/plots"):
    """Load saved results and generate all plots."""
    os.makedirs(output_dir, exist_ok=True)

    if not os.path.exists(results_path):
        print(f"  ⚠️  Results file not found: {results_path}")
        print("  Run run_compare.py first to generate results.")
        return

    with open(results_path, 'rb') as f:
        results = pickle.load(f)

    hope_results = results.get('hope', {})
    lstm_results = results.get('lstm', {})
    lgbm_results = results.get('lgbm', {})
    hope_histories = results.get('hope_histories', [])
    lstm_histories = results.get('lstm_histories', [])

    print("Generating plots...")

    # 1. Training curves
    if hope_histories or lstm_histories:
        plot_training_curves(hope_histories, lstm_histories, output_dir)

    # 2. Cumulative PnL
    if hope_results and lstm_results:
        plot_cumulative_pnl(hope_results, lstm_results, lgbm_results, output_dir)

    # 3. Per-stock Sharpe
    if hope_results:
        plot_per_stock_sharpe(hope_results, lstm_results, lgbm_results, output_dir)

    # 4. Prediction quality
    symbols = sorted(hope_results.keys())
    if symbols:
        sym0 = symbols[0]
        h_preds = hope_results[sym0].get('preds', np.array([]))
        l_preds = lstm_results.get(sym0, {}).get('preds', np.array([]))
        g_preds = lgbm_results.get(sym0, {}).get('preds', None)
        targets = hope_results[sym0].get('targets', np.array([]))
        plot_prediction_quality(h_preds, l_preds, g_preds, targets, output_dir)

        # 5. Trade analysis
        plot_trade_analysis(h_preds, l_preds, g_preds, targets, output_dir)

        # 6. Signal autocorrelation
        plot_signal_autocorrelation(h_preds, l_preds, g_preds, output_dir)

    # 7. Regime performance
    nifty = results.get('nifty_series')
    dates = results.get('dates')
    if hope_results and lstm_results:
        plot_regime_performance(hope_results, lstm_results, nifty, dates,
                                output_dir)

    print(f"\nAll plots saved to: {output_dir}")


if __name__ == "__main__":
    run_all()
