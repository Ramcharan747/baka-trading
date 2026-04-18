"""
Walk-forward evaluation for HOPE Phase 2.
Measures if model generalizes to unseen future data.
"""
from __future__ import annotations

import numpy as np
import torch
from scipy.stats import spearmanr

from train import detach_states


def walk_forward_evaluation(model, val_data, common_features,
                            chunk_size, device, n_windows=2,
                            eval_chunk_size=16):
    """
    Walk-forward: warm up state on first K bars, test on K+1 window.

    Args:
        model:           MiniHOPE instance
        val_data:        dict of {symbol: (features_df, labels_series)}
        common_features: list of feature names to use
        chunk_size:      Titans chunk size (used for warm-up)
        device:          cuda or cpu
        n_windows:       number of walk-forward windows (default 2)
        eval_chunk_size: chunk size for test window prediction (default 16)

    Returns: dict of {symbol: {mean_IC, positive_windows, n_windows}}
    """
    results = {}

    for sym, (feat_df, lab_series) in val_data.items():
        feat = feat_df[common_features].values
        labs = lab_series.values
        T = len(feat)

        # Adapt n_windows if data is too small
        effective_windows = n_windows
        if T < effective_windows * eval_chunk_size * 2:
            effective_windows = 1

        window = T // max(effective_windows, 1)

        if window < eval_chunk_size:
            results[sym] = {'mean_IC': 0.0, 'positive_windows': 0, 'n_windows': 0}
            continue

        ics = []
        for w in range(effective_windows - 1):
            train_end = (w + 1) * window
            test_end = min((w + 2) * window, T)

            # Convert to tensor [1, T, n_features]
            x_train = torch.tensor(
                feat[:train_end], dtype=torch.float32, device=device,
            ).unsqueeze(0)
            x_test = torch.tensor(
                feat[train_end:test_end], dtype=torch.float32, device=device,
            ).unsqueeze(0)
            y_test = labs[train_end:test_end]

            # Warm up state on training portion (use full chunk_size)
            states = model.init_state(1, torch.device(device))
            model.eval()
            with torch.no_grad():
                n_warmup = len(x_train[0]) // chunk_size
                for ci in range(n_warmup):
                    s = ci * chunk_size
                    e = s + chunk_size
                    _, states = model(x_train[:, s:e, :], states, step=s)
                    states = detach_states(states)

                # Test with end-of-warmup state (use smaller eval_chunk_size)
                all_preds = []
                n_test = len(x_test[0]) // eval_chunk_size
                for ci in range(n_test):
                    s = ci * eval_chunk_size
                    e = s + eval_chunk_size
                    pred, states = model(x_test[:, s:e, :], states, step=s)
                    pred_np = pred.squeeze().cpu().numpy()
                    if pred_np.ndim == 0:
                        pred_np = pred_np.reshape(1)
                    all_preds.extend(pred_np.tolist())
                    states = detach_states(states)

            pred_arr = np.array(all_preds)
            true_arr = y_test[:len(pred_arr)]
            if len(pred_arr) > 10 and np.std(pred_arr) > 1e-10:
                ic, p = spearmanr(pred_arr, true_arr)
                if not np.isnan(ic):
                    ics.append(ic)

        # If only 1 window, use full val set as test (no warm-up split)
        if effective_windows == 1 and not ics:
            x_full = torch.tensor(
                feat, dtype=torch.float32, device=device,
            ).unsqueeze(0)

            states = model.init_state(1, torch.device(device))
            model.eval()
            with torch.no_grad():
                all_preds = []
                n_chunks = T // eval_chunk_size
                for ci in range(n_chunks):
                    s = ci * eval_chunk_size
                    e = s + eval_chunk_size
                    pred, states = model(x_full[:, s:e, :], states, step=s)
                    pred_np = pred.squeeze().cpu().numpy()
                    if pred_np.ndim == 0:
                        pred_np = pred_np.reshape(1)
                    all_preds.extend(pred_np.tolist())
                    states = detach_states(states)

            pred_arr = np.array(all_preds)
            true_arr = labs[:len(pred_arr)]
            if len(pred_arr) > 10 and np.std(pred_arr) > 1e-10:
                ic, _ = spearmanr(pred_arr, true_arr)
                if not np.isnan(ic):
                    ics.append(ic)

        results[sym] = {
            'mean_IC': np.mean(ics) if ics else 0.0,
            'positive_windows': sum(ic > 0 for ic in ics),
            'n_windows': len(ics),
        }

    return results
