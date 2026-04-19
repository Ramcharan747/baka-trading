"""Training curve plots: loss and Sharpe/IC over epochs for HOPE vs LSTM."""
from __future__ import annotations

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def plot_training_curves(hope_histories: list, lstm_histories: list,
                         output_dir: str = "outputs/plots"):
    """
    Plot mean ± std training curves for HOPE and LSTM.

    hope_histories: list of dicts per seed, each {"epoch": [], "loss": []}
    lstm_histories: same format
    """
    os.makedirs(output_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 1, figsize=(14, 8))
    plt.style.use('seaborn-v0_8')

    # HOPE
    if hope_histories:
        epochs = hope_histories[0].get('epoch', list(range(len(hope_histories[0]['loss']))))
        losses = np.array([h['loss'] for h in hope_histories])
        mean_loss = losses.mean(axis=0)
        std_loss = losses.std(axis=0) if len(losses) > 1 else np.zeros_like(mean_loss)

        axes.plot(epochs, mean_loss, 'b-', label='HOPE', linewidth=2)
        axes.fill_between(epochs, mean_loss - std_loss, mean_loss + std_loss,
                          alpha=0.2, color='blue')

    # LSTM
    if lstm_histories:
        epochs = lstm_histories[0].get('epoch', list(range(len(lstm_histories[0]['loss']))))
        losses = np.array([h['loss'] for h in lstm_histories])
        mean_loss = losses.mean(axis=0)
        std_loss = losses.std(axis=0) if len(losses) > 1 else np.zeros_like(mean_loss)

        axes.plot(epochs, mean_loss, 'r-', label='LSTM', linewidth=2)
        axes.fill_between(epochs, mean_loss - std_loss, mean_loss + std_loss,
                          alpha=0.2, color='red')

    axes.set_xlabel('Epoch', fontsize=12)
    axes.set_ylabel('Loss (neg Sharpe)', fontsize=12)
    axes.set_title('Training Curves: HOPE vs LSTM', fontsize=14)
    axes.legend(fontsize=12)
    axes.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, 'training_curves.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")
