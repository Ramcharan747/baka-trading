"""
Walk-forward evaluation for HOPE and LSTM.

Flow:
  1. Warmup: run model on feat_tensor (training data) to build memory state
  2. Predict: for each stock in test data, run with warmed-up state
  3. Compute: IC (Spearman), realized Sharpe
  4. Return raw predictions and targets for plotting
"""
from __future__ import annotations

import numpy as np
import torch
from scipy.stats import spearmanr

from train import detach_states


def walk_forward_evaluation(model, feat_tensor: torch.Tensor,
                            test_data: dict,
                            common_features: list,
                            config, device,
                            model_type: str = "hope") -> dict:
    """
    Evaluate model on test data with warmed-up state.

    Args:
        model: MiniHOPE or LSTMBaseline
        feat_tensor: [n_stocks, T_train, n_features] — exact training tensor
        test_data: {symbol: (features_df, labels_series)}
        common_features: list of feature names
        config: PhaseConfig
        device: torch device string
        model_type: "hope" or "lstm"

    Returns: {symbol: {"IC": float, "sharpe": float,
                        "preds": ndarray, "targets": ndarray}}
    """
    model.eval()
    n_stocks, T_train, n_features = feat_tensor.shape
    chunk_size = config.chunk_size

    # ── Step 1: Warmup on training tensor ─────────────────────────
    feat_train = feat_tensor.to(device)

    if model_type == "hope":
        states = model.init_state(
            batch_size=n_stocks, device=torch.device(device))
        with torch.no_grad():
            for ci in range(T_train // chunk_size):
                s = ci * chunk_size
                e = s + chunk_size
                x = feat_train[:, s:e, :]
                _, states = model(x, states, step=s)
                states = detach_states(states, "hope")
    else:
        h = torch.zeros(model.n_layers, n_stocks,
                        model.hidden_size, device=device)
        c = torch.zeros_like(h)
        lstm_state = (h, c)
        with torch.no_grad():
            for ci in range(T_train // chunk_size):
                s = ci * chunk_size
                e = s + chunk_size
                x = feat_train[:, s:e, :]
                _, lstm_state = model(x, lstm_state)
                lstm_state = (lstm_state[0].detach(),
                              lstm_state[1].detach())

    # ── Step 2: Predict on each stock's test data ─────────────────
    symbols = sorted(test_data.keys())
    results = {}

    for stock_idx, sym in enumerate(symbols):
        feat_df, lab_series = test_data[sym]
        feat_arr = feat_df[common_features].values
        lab_arr = lab_series.values
        T_test = len(feat_arr)

        x_test = torch.tensor(
            feat_arr, dtype=torch.float32, device=device
        ).unsqueeze(0)  # [1, T_test, n_features]

        all_preds = []

        if model_type == "hope":
            # Slice this stock's warmed-up state
            stock_states = []
            for layer_state in states:
                s_single = {}
                for k, v in layer_state.items():
                    if isinstance(v, torch.Tensor) and v.dim() >= 2:
                        s_single[k] = v[stock_idx:stock_idx + 1]
                    else:
                        s_single[k] = v
                stock_states.append(s_single)

            with torch.no_grad():
                for ci in range(T_test // chunk_size):
                    s = ci * chunk_size
                    e = s + chunk_size
                    x = x_test[:, s:e, :]
                    pred, stock_states = model(x, stock_states, step=s)
                    stock_states = detach_states(stock_states, "hope")
                    all_preds.extend(
                        pred.squeeze().cpu().numpy().tolist())
        else:
            h_single = lstm_state[0][:, stock_idx:stock_idx + 1, :].contiguous()
            c_single = lstm_state[1][:, stock_idx:stock_idx + 1, :].contiguous()
            stock_lstm_state = (h_single, c_single)

            with torch.no_grad():
                for ci in range(T_test // chunk_size):
                    s = ci * chunk_size
                    e = s + chunk_size
                    x = x_test[:, s:e, :]
                    pred, stock_lstm_state = model(x, stock_lstm_state)
                    stock_lstm_state = (
                        stock_lstm_state[0].detach(),
                        stock_lstm_state[1].detach())
                    all_preds.extend(
                        pred.squeeze().cpu().numpy().tolist())

        preds_arr = np.array(all_preds)
        true_arr = lab_arr[:len(preds_arr)]

        # Remove NaN labels
        valid = ~np.isnan(true_arr)
        preds_valid = preds_arr[valid]
        true_valid = true_arr[valid]

        # IC (Spearman)
        if len(preds_valid) > 10:
            ic, _ = spearmanr(preds_valid, true_valid)
            if np.isnan(ic):
                ic = 0.0
        else:
            ic = 0.0

        # Realized Sharpe from positions
        positions = np.tanh(preds_valid * 10)
        pnl = positions * true_valid
        if pnl.std() > 1e-10:
            sharpe = (pnl.mean() / pnl.std()) * np.sqrt(252 * 375)
        else:
            sharpe = 0.0

        results[sym] = {
            'IC': ic,
            'sharpe': sharpe,
            'preds': preds_arr,
            'targets': true_arr,
        }

    return results
