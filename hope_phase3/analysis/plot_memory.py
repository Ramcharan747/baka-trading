"""HOPE memory trajectory and regime plots."""
from __future__ import annotations

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def plot_hope_memory_trajectory(w_norm_history: dict,
                                output_dir: str = "outputs/plots"):
    """
    W_norm over training steps for each CMS level.
    w_norm_history: {cms_level: [norm_per_step]}
    """
    os.makedirs(output_dir, exist_ok=True)

    n_levels = len(w_norm_history)
    fig, axes = plt.subplots(n_levels, 1, figsize=(14, 4 * n_levels))
    if n_levels == 1:
        axes = [axes]

    for i, (level, norms) in enumerate(sorted(w_norm_history.items())):
        axes[i].plot(norms, linewidth=1)
        axes[i].set_ylabel('W_norm', fontsize=11)
        axes[i].set_title(f'CMS Level {level} Memory Norm', fontsize=12)
        axes[i].grid(True, alpha=0.3)

    axes[-1].set_xlabel('Training Step', fontsize=11)
    plt.suptitle('HOPE Memory Trajectory', fontsize=14)
    plt.tight_layout()

    path = os.path.join(output_dir, 'memory_trajectory.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


def plot_memory_state_at_regimes(hope_states, price_series, dates,
                                 output_dir: str = "outputs/plots"):
    """
    Top: price with regime labels. Bottom: memory norm at regime transitions.
    """
    os.makedirs(output_dir, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    ax1.plot(dates, price_series, 'k-', linewidth=1)
    ax1.set_ylabel('Price', fontsize=11)
    ax1.set_title('Price and Memory Response', fontsize=14)
    ax1.grid(True, alpha=0.3)

    # Extract memory norms from states
    if hope_states:
        norms = []
        for state in hope_states:
            if isinstance(state, list) and len(state) > 0:
                layer_norms = []
                for layer_state in state:
                    for k, v in layer_state.items():
                        if hasattr(v, 'norm'):
                            layer_norms.append(v.norm().item())
                norms.append(np.mean(layer_norms) if layer_norms else 0)
            else:
                norms.append(0)
        ax2.plot(dates[:len(norms)], norms, 'b-', linewidth=1)
    ax2.set_ylabel('Memory Norm', fontsize=11)
    ax2.set_xlabel('Date', fontsize=11)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, 'memory_regimes.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")
