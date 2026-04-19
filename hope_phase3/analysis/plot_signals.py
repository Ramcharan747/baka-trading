"""Signal quality, trade analysis, and autocorrelation plots."""
from __future__ import annotations

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def plot_prediction_quality(hope_preds, lstm_preds, lgbm_preds,
                            targets, output_dir: str = "outputs/plots"):
    """
    3-panel hexbin scatter: pred vs target for each model.
    """
    os.makedirs(output_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    datasets = [
        (hope_preds, "HOPE", 'Blues'),
        (lstm_preds, "LSTM", 'Reds'),
        (lgbm_preds, "LightGBM", 'Greens'),
    ]

    for ax, (preds, name, cmap) in zip(axes, datasets):
        if preds is None or len(preds) == 0:
            ax.text(0.5, 0.5, f'{name}\nNo data', ha='center', va='center')
            ax.set_title(name)
            continue

        valid = ~np.isnan(targets[:len(preds)]) & ~np.isnan(preds)
        p = preds[valid]
        t = targets[:len(preds)][valid]

        if len(p) > 0:
            ax.hexbin(p, t, gridsize=30, cmap=cmap, mincnt=1)
            # Regression line
            if len(p) > 2:
                z = np.polyfit(p, t, 1)
                x_line = np.linspace(p.min(), p.max(), 50)
                ax.plot(x_line, np.polyval(z, x_line), 'k--', linewidth=2)
                from scipy.stats import spearmanr
                ic, _ = spearmanr(p, t)
                ax.text(0.05, 0.95, f'IC={ic:.4f}', transform=ax.transAxes,
                        fontsize=11, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        ax.set_xlabel('Prediction', fontsize=11)
        ax.set_ylabel('Target', fontsize=11)
        ax.set_title(name, fontsize=13)
        ax.grid(True, alpha=0.3)

    plt.suptitle('Prediction vs Target', fontsize=14)
    plt.tight_layout()
    path = os.path.join(output_dir, 'prediction_quality.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


def plot_trade_analysis(hope_preds, lstm_preds, lgbm_preds,
                        targets, output_dir: str = "outputs/plots"):
    """
    4-panel figure:
    (a) Trade size distribution
    (b) Win rate by magnitude
    (c) Average PnL by time bin
    (d) PnL distribution
    """
    os.makedirs(output_dir, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    models = [
        (hope_preds, "HOPE", 'blue'),
        (lstm_preds, "LSTM", 'red'),
    ]
    if lgbm_preds is not None:
        models.append((lgbm_preds, "LGBM", 'green'))

    # (a) Trade size distribution
    ax = axes[0, 0]
    for preds, name, color in models:
        if preds is not None and len(preds) > 0:
            positions = np.tanh(preds * 10)
            ax.hist(positions, bins=50, alpha=0.5, label=name, color=color)
    ax.set_title('Position Size Distribution', fontsize=12)
    ax.set_xlabel('tanh(pred × 10)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # (b) Win rate by magnitude
    ax = axes[0, 1]
    for preds, name, color in models:
        if preds is not None and len(preds) > 0:
            valid = ~np.isnan(targets[:len(preds)])
            p = np.abs(preds[valid])
            t = targets[:len(preds)][valid]
            pos = np.tanh(preds[valid] * 10)
            pnl = pos * t

            bins = np.percentile(p, np.linspace(0, 100, 11))
            bins = np.unique(bins)
            if len(bins) > 1:
                indices = np.digitize(p, bins) - 1
                wr = []
                centers = []
                for bi in range(len(bins) - 1):
                    mask = indices == bi
                    if mask.sum() > 0:
                        wr.append((pnl[mask] > 0).mean())
                        centers.append((bins[bi] + bins[bi + 1]) / 2)
                ax.plot(centers, wr, 'o-', color=color, label=name)

    ax.axhline(0.5, color='gray', linestyle='--', alpha=0.5)
    ax.set_title('Win Rate by Prediction Magnitude', fontsize=12)
    ax.set_xlabel('|Prediction|')
    ax.set_ylabel('Win Rate')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # (c) PnL distribution
    ax = axes[1, 0]
    for preds, name, color in models:
        if preds is not None and len(preds) > 0:
            valid = ~np.isnan(targets[:len(preds)])
            pos = np.tanh(preds[valid] * 10)
            pnl = pos * targets[:len(preds)][valid]
            ax.hist(pnl, bins=50, alpha=0.5, label=name, color=color)

    ax.set_title('PnL Distribution', fontsize=12)
    ax.set_xlabel('PnL per bar')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # (d) Cumulative PnL
    ax = axes[1, 1]
    for preds, name, color in models:
        if preds is not None and len(preds) > 0:
            valid = ~np.isnan(targets[:len(preds)])
            pos = np.tanh(preds[valid] * 10)
            pnl = pos * targets[:len(preds)][valid]
            ax.plot(np.cumsum(pnl), color=color, label=name, linewidth=1.5)

    ax.set_title('Cumulative PnL', fontsize=12)
    ax.set_xlabel('Bar')
    ax.set_ylabel('Cumulative PnL')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.suptitle('Trade Analysis', fontsize=14)
    plt.tight_layout()
    path = os.path.join(output_dir, 'trade_analysis.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


def plot_signal_autocorrelation(hope_preds, lstm_preds, lgbm_preds,
                                output_dir: str = "outputs/plots",
                                max_lag: int = 60):
    """
    Autocorrelation of predictions for each model.
    HOPE should decay faster (less stale signals).
    """
    os.makedirs(output_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, 6))

    models = [
        (hope_preds, "HOPE", 'blue'),
        (lstm_preds, "LSTM", 'red'),
    ]
    if lgbm_preds is not None:
        models.append((lgbm_preds, "LGBM", 'green'))

    for preds, name, color in models:
        if preds is None or len(preds) < max_lag + 10:
            continue
        p = preds[~np.isnan(preds)]
        if len(p) < max_lag + 10:
            continue

        acf = []
        p_centered = p - p.mean()
        var = np.var(p_centered)
        if var < 1e-12:
            continue
        for lag in range(1, max_lag + 1):
            c = np.mean(p_centered[:-lag] * p_centered[lag:]) / var
            acf.append(c)

        ax.plot(range(1, max_lag + 1), acf, '-', color=color,
                label=name, linewidth=2)

    ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Lag (bars)', fontsize=12)
    ax.set_ylabel('Autocorrelation', fontsize=12)
    ax.set_title('Signal Autocorrelation', fontsize=14)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, 'signal_autocorrelation.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")
