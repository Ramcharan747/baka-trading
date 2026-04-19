"""Per-stock Sharpe comparison and regime performance plots."""
from __future__ import annotations

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def plot_per_stock_sharpe(hope_results: dict, lstm_results: dict,
                          lgbm_results: dict = None,
                          output_dir: str = "outputs/plots"):
    """
    Grouped bar chart: one group per stock, 3 bars (HOPE/LSTM/LGBM).
    Sorted by HOPE Sharpe descending.
    """
    os.makedirs(output_dir, exist_ok=True)

    symbols = sorted(hope_results.keys())
    hope_sharpes = [hope_results[s].get('sharpe', 0) for s in symbols]
    lstm_sharpes = [lstm_results.get(s, {}).get('sharpe', 0) for s in symbols]
    lgbm_sharpes = [lgbm_results.get(s, {}).get('sharpe', 0) for s in symbols] \
        if lgbm_results else [0] * len(symbols)

    # Sort by HOPE Sharpe
    order = np.argsort(-np.array(hope_sharpes))
    symbols = [symbols[i] for i in order]
    hope_sharpes = [hope_sharpes[i] for i in order]
    lstm_sharpes = [lstm_sharpes[i] for i in order]
    lgbm_sharpes = [lgbm_sharpes[i] for i in order]

    x = np.arange(len(symbols))
    width = 0.25

    fig, ax = plt.subplots(figsize=(max(14, len(symbols) * 0.6), 8))
    bars1 = ax.bar(x - width, hope_sharpes, width, label='HOPE', color='#2196F3')
    bars2 = ax.bar(x, lstm_sharpes, width, label='LSTM', color='#FF5722')
    if lgbm_results:
        bars3 = ax.bar(x + width, lgbm_sharpes, width, label='LGBM',
                       color='#4CAF50')

    ax.axhline(0, color='gray', linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(symbols, rotation=45, ha='right', fontsize=9)
    ax.set_ylabel('Sharpe Ratio', fontsize=12)
    ax.set_title('Per-Stock Sharpe Ratio Comparison', fontsize=14)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    path = os.path.join(output_dir, 'per_stock_sharpe.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


def plot_regime_performance(hope_results: dict, lstm_results: dict,
                            nifty_series=None, dates=None,
                            output_dir: str = "outputs/plots"):
    """
    Top: NIFTY50 price with regime shading.
    Bottom: 30-day rolling Sharpe for HOPE and LSTM (portfolio level).
    """
    os.makedirs(output_dir, exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

    # Top panel: NIFTY50 price
    if nifty_series is not None and dates is not None:
        ax1.plot(dates, nifty_series, 'k-', linewidth=1)
        ax1.set_ylabel('NIFTY50', fontsize=11)
        ax1.set_title('Market Regime and Model Performance', fontsize=14)

        # Simple regime detection: rolling vol
        if len(nifty_series) > 60:
            ret = np.diff(np.log(nifty_series))
            vol = np.convolve(np.abs(ret), np.ones(20) / 20, mode='same')
            vol_med = np.median(vol)
            for i in range(len(vol)):
                if vol[i] > vol_med * 1.5:
                    ax1.axvspan(dates[i], dates[min(i + 1, len(dates) - 1)],
                                alpha=0.1, color='red')
    else:
        ax1.text(0.5, 0.5, 'NIFTY50 data not available',
                 ha='center', va='center', transform=ax1.transAxes)
    ax1.grid(True, alpha=0.3)

    # Bottom panel: rolling Sharpe
    window = 30 * 375  # 30 days × 375 bars/day

    for results, name, color in [(hope_results, 'HOPE', 'blue'),
                                  (lstm_results, 'LSTM', 'red')]:
        # Combine all stocks into portfolio PnL
        all_pnl = []
        for sym in sorted(results.keys()):
            r = results[sym]
            p = r.get('preds', np.array([]))
            t = r.get('targets', np.array([]))
            n = min(len(p), len(t))
            if n > 0:
                pnl = np.tanh(p[:n] * 10) * t[:n]
                pnl = np.nan_to_num(pnl)
                all_pnl.append(pnl)

        if all_pnl:
            min_len = min(len(p) for p in all_pnl)
            portfolio_pnl = np.mean([p[:min_len] for p in all_pnl], axis=0)

            # Rolling Sharpe
            if len(portfolio_pnl) > window:
                rolling_sharpe = []
                for i in range(window, len(portfolio_pnl)):
                    chunk = portfolio_pnl[i - window:i]
                    if chunk.std() > 1e-10:
                        s = (chunk.mean() / chunk.std()) * np.sqrt(252 * 375)
                    else:
                        s = 0
                    rolling_sharpe.append(s)
                ax2.plot(range(window, len(portfolio_pnl)), rolling_sharpe,
                         color=color, label=name, linewidth=1.5)

    ax2.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax2.set_ylabel('Rolling Sharpe', fontsize=11)
    ax2.set_xlabel('Bar', fontsize=11)
    ax2.legend(fontsize=12)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, 'regime_performance.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")
