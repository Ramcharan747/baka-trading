"""Cumulative PnL plots for HOPE vs LSTM vs LightGBM."""
from __future__ import annotations

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def plot_cumulative_pnl(hope_results: dict, lstm_results: dict,
                        lgbm_results: dict = None,
                        output_dir: str = "outputs/plots"):
    """
    Cumulative PnL for each stock + combined portfolio.
    """
    os.makedirs(output_dir, exist_ok=True)
    symbols = sorted(hope_results.keys())
    n = len(symbols)
    cols = min(5, n)
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows))
    if n == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for i, sym in enumerate(symbols):
        ax = axes[i]

        # HOPE
        h = hope_results[sym]
        h_pnl = np.tanh(h['preds'] * 10) * h['targets'][:len(h['preds'])]
        h_pnl = np.nan_to_num(h_pnl)
        ax.plot(np.cumsum(h_pnl), 'b-', linewidth=1, label='HOPE')

        # LSTM
        l = lstm_results[sym]
        l_pnl = np.tanh(l['preds'] * 10) * l['targets'][:len(l['preds'])]
        l_pnl = np.nan_to_num(l_pnl)
        ax.plot(np.cumsum(l_pnl), 'r-', linewidth=1, label='LSTM')

        # LightGBM
        if lgbm_results and sym in lgbm_results:
            g = lgbm_results[sym]
            if 'preds' in g:
                g_pnl = np.tanh(g['preds'] * 10) * g['targets'][:len(g['preds'])]
                g_pnl = np.nan_to_num(g_pnl)
                ax.plot(np.cumsum(g_pnl), 'g-', linewidth=1, label='LGBM')

        ax.set_title(sym, fontsize=9)
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.legend(fontsize=7)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle('Cumulative PnL by Stock', fontsize=14)
    plt.tight_layout()

    path = os.path.join(output_dir, 'cumulative_pnl.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")
