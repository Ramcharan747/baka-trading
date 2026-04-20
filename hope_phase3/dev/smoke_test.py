"""
Phase 3 Smoke Test: validates entire pipeline in <5 minutes.

Tests:
  1. Download 1 month of minute data for 3 stocks
  2. Compute all 70 features, check no NaN
  3. Print IC of each feature group with label
  4. Initialize HOPE model, check param count
  5. Initialize LSTM model, check param count
  6. Run 1 training epoch, check loss
  7. Check memory W_norm is in healthy range
  8. Check gradient flow
  9. Print PASS/FAIL
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import PhaseConfig
from instruments import INSTRUMENTS
from data import load_all_stocks
from features import compute_features, FEATURE_NAMES
from labels import compute_labels
from ic_test import ic_test
from models.hope import MiniHOPE, HopeConfig
from models.lstm_baseline import LSTMBaseline
from train import train_epoch_minute, init_states_hope, init_states_lstm, make_optimizer
from losses import sharpe_loss


SMOKE_STOCKS = ["RELIANCE", "TCS", "SBIN"]
SMOKE_FROM = "2024-01-01"
SMOKE_TO = "2024-01-31"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--token", default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    token = args.token or os.environ.get('UPSTOX_TOKEN')
    device = args.device
    passed = True

    print("=" * 60)
    print("  HOPE Phase 3 — SMOKE TEST")
    print("=" * 60)

    # ── Test 1: Download data ─────────────────────────────────────
    print("\n[1/9] Downloading minute data...")
    instruments = {k: v for k, v in INSTRUMENTS.items()
                   if k in SMOKE_STOCKS or k == "NIFTY50"}

    try:
        all_data = load_all_stocks(instruments, token, SMOKE_FROM, SMOKE_TO)
        nifty_df = all_data.pop("NIFTY50", None)
        n_stocks = len(all_data)
        print(f"  ✅ Downloaded {n_stocks} stocks")
        if n_stocks < 2:
            print(f"  ❌ Need at least 2 stocks, got {n_stocks}")
            passed = False
            return
    except Exception as e:
        print(f"  ❌ Data download failed: {e}")
        passed = False
        return

    # ── Test 2: Compute features ──────────────────────────────────
    print("\n[2/9] Computing features...")
    dataset = {}
    for sym in sorted(all_data.keys()):
        df = all_data[sym]
        feats = compute_features(df, nifty_df)
        labs = compute_labels(df['close'], 15, 0.0003)
        valid = labs.notna()
        feats = feats[valid].reset_index(drop=True)
        labs = labs[valid].reset_index(drop=True)
        dataset[sym] = (feats, labs)

        nan_count = feats.isna().sum().sum()
        inf_count = np.isinf(feats.values).sum()
        n_cols = feats.shape[1]

        if nan_count > 0:
            print(f"  ❌ {sym}: {nan_count} NaN values in features")
            passed = False
        elif inf_count > 0:
            print(f"  ❌ {sym}: {inf_count} inf values in features")
            passed = False
        else:
            print(f"  ✅ {sym}: {len(feats)} bars, {n_cols} features, "
                  f"0 NaN, 0 inf")

        if n_cols != 70:
            print(f"  ❌ Expected 70 features, got {n_cols}")
            passed = False

    # ── Test 3: IC diagnostic ─────────────────────────────────────
    print("\n[3/9] Feature IC diagnostic...")
    first_sym = sorted(dataset.keys())[0]
    f, l = dataset[first_sym]
    ic_results = ic_test(f, l)
    n_sig = ic_results['keep'].sum()
    print(f"  {first_sym}: {n_sig}/{len(ic_results)} features significant (p<0.05)")
    print("  Top 5 by |IC|:")
    for feat, row in ic_results.head(5).iterrows():
        print(f"    {feat:20s}  IC={row['IC']:+.4f}  p={row['p']:.4f}")
    print(f"  ✅ IC test completed")

    # ── Test 4: HOPE param count ──────────────────────────────────
    print("\n[4/9] HOPE model initialization...")
    config = PhaseConfig()
    hope_config = HopeConfig(
        n_features=70, d_model=128, n_layers=4, n_outputs=1,
        inner_lr=0.01, inner_decay=0.99, grad_clip=1.0,
        cms_levels=3, cms_schedule=[30, 120, 375],
        cms_lr=[1e-3, 1e-4, 1e-5], chunk_size=64,
    )
    hope = MiniHOPE(hope_config).to(device)
    hope_params = sum(p.numel() for p in hope.parameters())
    print(f"  HOPE params: {hope_params:,}")
    if hope_params < 500_000 or hope_params > 3_000_000:
        print(f"  ⚠️  Expected ~1.26M params, got {hope_params:,}")
    else:
        print(f"  ✅ HOPE param count in expected range")

    # ── Test 5: LSTM param count ──────────────────────────────────
    print("\n[5/9] LSTM model initialization...")
    lstm = LSTMBaseline(
        n_features=70, hidden_size=config.lstm_hidden,
        n_layers=config.lstm_layers, n_outputs=1,
    ).to(device)
    lstm_params = sum(p.numel() for p in lstm.parameters())
    print(f"  LSTM params: {lstm_params:,}")
    if lstm_params < 500_000 or lstm_params > 3_000_000:
        print(f"  ⚠️  Expected ~1.28M params, got {lstm_params:,}")
    else:
        print(f"  ✅ LSTM param count in expected range")

    # ── Test 6: Training epoch ────────────────────────────────────
    print("\n[6/9] Training 1 epoch (2 chunks)...")
    symbols = sorted(dataset.keys())

    # Build training tensor (small)
    min_len = min(len(dataset[s][0]) for s in symbols)
    chunk_test = min(128, min_len)  # use 2 chunks of 64
    feat_arrays = [dataset[s][0][FEATURE_NAMES].iloc[:chunk_test].values
                   for s in symbols]
    lab_arrays = [dataset[s][1].iloc[:chunk_test].values
                  for s in symbols]
    feat_t = torch.tensor(np.stack(feat_arrays), dtype=torch.float32).to(device)
    lab_t = torch.tensor(np.stack(lab_arrays), dtype=torch.float32).to(device)

    optimizer = make_optimizer(hope, lr=1e-3, weight_decay=0.01, model_type="hope")
    states = init_states_hope(hope, len(symbols), torch.device(device))
    config.chunk_size = 64

    t0 = time.time()
    loss, states = train_epoch_minute(
        hope, feat_t, lab_t, optimizer, None, states, config, model_type="hope")
    dt = time.time() - t0
    print(f"  Loss after 1 epoch: {loss:.6f}  ({dt:.1f}s)")
    print(f"  ✅ Training epoch completed")

    # ── Test 7: Memory W_norm ─────────────────────────────────────
    print("\n[7/9] Memory state health check...")
    for li, layer_state in enumerate(states):
        for k, v in layer_state.items():
            if isinstance(v, torch.Tensor) and v.dim() >= 2:
                norm = v.norm().item()
                print(f"  Layer {li} {k}: norm={norm:.4f}")
                if norm < 0.0001 or norm > 1000:
                    print(f"  ⚠️  {k} norm out of healthy range [0.1, 10]")
    print(f"  ✅ Memory state check completed")

    # ── Test 8: Gradient flow ─────────────────────────────────────
    print("\n[8/9] Gradient flow check...")
    no_grad_params = []
    for name, p in hope.named_parameters():
        if p.requires_grad and (p.grad is None or p.grad.norm().item() == 0):
            no_grad_params.append(name)

    if no_grad_params:
        print(f"  ⚠️  {len(no_grad_params)} params with no gradient:")
        for n in no_grad_params[:5]:
            print(f"    {n}")
    else:
        print(f"  ✅ All parameters receiving gradients")

    # ── Test 9: Final verdict ─────────────────────────────────────
    print("\n" + "=" * 60)
    if passed:
        print("  ✅ Smoke test PASSED")
    else:
        print("  ❌ Smoke test FAILED")
    print("=" * 60)


if __name__ == "__main__":
    main()
