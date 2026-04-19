"""Feature IC heatmap and importance comparison."""
from __future__ import annotations

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def plot_feature_ic_heatmap(feat_tensor, lab_tensor, feature_names,
                            output_dir: str = "outputs/plots",
                            window: int = 20):
    """
    Rolling IC heatmap: X=time, Y=feature, Color=IC.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Average across stocks
    n_stocks, T, n_feat = feat_tensor.shape
    feat_avg = feat_tensor.mean(dim=0).cpu().numpy()  # [T, n_feat]
    lab_avg = lab_tensor.mean(dim=0).cpu().numpy()     # [T]

    # Rolling IC
    n_windows = T // window
    ic_matrix = np.zeros((n_feat, n_windows))

    for wi in range(n_windows):
        s = wi * window
        e = s + window
        for fi in range(n_feat):
            from scipy.stats import spearmanr
            ic, _ = spearmanr(feat_avg[s:e, fi], lab_avg[s:e])
            ic_matrix[fi, wi] = ic if not np.isnan(ic) else 0

    # Sort by mean IC
    mean_ics = np.abs(ic_matrix).mean(axis=1)
    order = np.argsort(-mean_ics)

    fig, ax = plt.subplots(figsize=(16, 10))
    im = ax.imshow(ic_matrix[order], aspect='auto', cmap='RdBu_r',
                   vmin=-0.3, vmax=0.3)
    ax.set_yticks(range(n_feat))
    ax.set_yticklabels([feature_names[i] for i in order], fontsize=6)
    ax.set_xlabel('Time Window', fontsize=12)
    ax.set_title('Feature IC Heatmap (sorted by |IC|)', fontsize=14)
    plt.colorbar(im, label='Spearman IC')
    plt.tight_layout()

    path = os.path.join(output_dir, 'feature_ic_heatmap.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


def plot_feature_importance(lgbm_importance, feature_names,
                            output_dir: str = "outputs/plots"):
    """LightGBM feature importance bar chart (top 20)."""
    os.makedirs(output_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(12, 8))
    top20 = lgbm_importance.head(20)
    ax.barh(range(len(top20)), top20.values, color='forestgreen', alpha=0.8)
    ax.set_yticks(range(len(top20)))
    ax.set_yticklabels(top20.index, fontsize=10)
    ax.set_xlabel('Gain', fontsize=12)
    ax.set_title('LightGBM Feature Importance (Top 20)', fontsize=14)
    ax.invert_yaxis()
    plt.tight_layout()

    path = os.path.join(output_dir, 'feature_importance.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")
