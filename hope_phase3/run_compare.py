"""
Phase 3: HOPE vs LSTM vs LightGBM comparison.

Loads HOPE from HF checkpoints (5 seeds), trains LSTM from scratch (5 seeds),
trains LightGBM (1 run), prints 3-way comparison table.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, ".")

from config import PhaseConfig
from instruments import INSTRUMENTS, PREDICT_STOCKS, SECTORS, STOCK_SECTOR
from data import load_all_stocks
from features import compute_features, FEATURE_NAMES
from labels import compute_labels
from models.hope import MiniHOPE, HopeConfig
from models.lstm_baseline import LSTMBaseline
from train import train_epoch_minute, init_states_hope, init_states_lstm, detach_states
from evaluate import walk_forward_evaluation
from backtest import compute_backtest_metrics
from checkpoint import load_checkpoint
from losses import sharpe_loss


def make_hope_config(config: PhaseConfig):
    return HopeConfig(
        n_features=config.n_features,
        d_model=config.d_model,
        n_layers=config.n_layers,
        n_outputs=1,
        inner_lr=config.inner_lr,
        inner_decay=config.inner_decay,
        grad_clip=config.grad_clip_inner,
        cms_levels=config.cms_levels,
        cms_schedule=config.cms_schedule,
        cms_lr=config.cms_lr,
        chunk_size=config.chunk_size,
    )


def main():
    parser = argparse.ArgumentParser(description="HOPE vs LSTM vs LightGBM")
    parser.add_argument("--mode",       default="dev", choices=["dev", "prod"])
    parser.add_argument("--hope_seeds", default="0")
    parser.add_argument("--lstm_seeds", default="0")
    parser.add_argument("--device",     default="cuda")
    parser.add_argument("--epochs",     type=int, default=None)
    parser.add_argument("--token",      default=None)
    parser.add_argument("--hf_token",   default=None)
    parser.add_argument("--hf_repo",    default="Baka7/hope-finance-phase3")
    args = parser.parse_args()

    config = PhaseConfig(mode=args.mode)
    device = args.device

    hope_seeds = [int(s) for s in args.hope_seeds.split(",")]
    lstm_seeds = [int(s) for s in args.lstm_seeds.split(",")]

    if args.mode == "dev":
        epochs = args.epochs or config.dev_epochs
        stock_list = config.dev_stocks
        from_date = config.dev_from_date
        to_date = config.dev_to_date
    else:
        epochs = args.epochs or config.epochs
        stock_list = None
        from_date = config.from_date
        to_date = config.to_date

    upstox_token = args.token or os.environ.get('UPSTOX_TOKEN')
    hf_token = args.hf_token or os.environ.get('HF_TOKEN')

    if hf_token:
        try:
            from huggingface_hub import login
            login(token=hf_token)
        except Exception:
            pass

    print("=" * 100)
    print(f"  HOPE vs LSTM vs LightGBM — Phase 3 Comparison")
    print(f"  Mode: {args.mode}  HOPE seeds: {hope_seeds}  LSTM seeds: {lstm_seeds}")
    print("=" * 100)

    # ── [1/5] Load data ───────────────────────────────────────────
    print("\n[1/5] Loading data...")
    instruments = INSTRUMENTS.copy()
    if stock_list:
        instruments = {k: v for k, v in instruments.items()
                       if k in stock_list or k == "NIFTY50"}

    all_data = load_all_stocks(instruments, upstox_token, from_date, to_date)
    nifty_df = all_data.pop("NIFTY50", None)

    # Compute features + labels
    sector_closes = {}
    for sector, members in SECTORS.items():
        available = [s for s in members if s in all_data]
        if available:
            sector_closes[sector] = sum(
                all_data[s]['close'] for s in available
            ) / len(available)

    dataset = {}
    for sym in sorted(all_data.keys()):
        df = all_data[sym]
        sector = STOCK_SECTOR.get(sym)
        sector_close = sector_closes.get(sector)

        feats = compute_features(df, nifty_df, sector_close)
        labs = compute_labels(df['close'], config.label_horizon,
                              config.transaction_cost)
        valid = labs.notna()
        feats = feats[valid].reset_index(drop=True)
        labs = labs[valid].reset_index(drop=True)
        dataset[sym] = (feats, labs)

    symbols = sorted(dataset.keys())
    n_stocks = len(symbols)

    # Split data
    if args.mode == "dev":
        train_feats, train_labs = {}, {}
        val_feats, val_labs = {}, {}
        test_feats, test_labs = {}, {}
        for sym in symbols:
            f, l = dataset[sym]
            T = len(f)
            t1 = int(T * 0.7)
            t2 = int(T * 0.85)
            train_feats[sym] = f.iloc[:t1]
            train_labs[sym] = l.iloc[:t1]
            val_feats[sym] = f.iloc[t1:t2]
            val_labs[sym] = l.iloc[t1:t2]
            test_feats[sym] = f.iloc[t2:]
            test_labs[sym] = l.iloc[t2:]
    else:
        import pandas as pd
        train_end = pd.Timestamp(config.train_end)
        val_end = pd.Timestamp(config.val_end)
        train_feats, train_labs = {}, {}
        val_feats, val_labs = {}, {}
        test_feats, test_labs = {}, {}
        for sym in symbols:
            f, l = dataset[sym]
            df_orig = all_data[sym]
            dt = pd.to_datetime(df_orig['datetime'])
            dt_valid = dt[l.index].reset_index(drop=True)
            train_mask = dt_valid <= train_end
            val_mask = (dt_valid > train_end) & (dt_valid <= val_end)
            test_mask = dt_valid > val_end
            train_feats[sym] = f[train_mask].reset_index(drop=True)
            train_labs[sym] = l[train_mask].reset_index(drop=True)
            val_feats[sym] = f[val_mask].reset_index(drop=True)
            val_labs[sym] = l[val_mask].reset_index(drop=True)
            test_feats[sym] = f[test_mask].reset_index(drop=True)
            test_labs[sym] = l[test_mask].reset_index(drop=True)

    # Stack training tensors
    min_train = min(len(train_feats[s]) for s in symbols)
    feat_arrays = [train_feats[s][FEATURE_NAMES].iloc[:min_train].values
                   for s in symbols]
    lab_arrays = [train_labs[s].iloc[:min_train].values
                  for s in symbols]
    feat_tensor = torch.tensor(
        np.stack(feat_arrays, axis=0), dtype=torch.float32).to(device)
    lab_tensor = torch.tensor(
        np.stack(lab_arrays, axis=0), dtype=torch.float32).to(device)

    test_data = {sym: (test_feats[sym], test_labs[sym]) for sym in symbols}
    print(f"  {n_stocks} stocks, training tensor: {feat_tensor.shape}")

    # ── [2/5] Load HOPE from checkpoints ──────────────────────────
    print(f"\n[2/5] Loading HOPE from checkpoints ({len(hope_seeds)} seeds)...")
    all_hope_results = {}  # {sym: [result_seed0, ...]}

    for seed in hope_seeds:
        print(f"\n  ── HOPE seed {seed} ──")
        torch.manual_seed(seed)
        hope_config = make_hope_config(config)
        hope = MiniHOPE(hope_config).to(device)
        optimizer_dummy = torch.optim.AdamW(hope.parameters(), lr=config.lr)
        hope, _, _, start_epoch, _ = load_checkpoint(
            hope, optimizer_dummy, device=device)
        print(f"  Loaded from epoch {max(0, start_epoch - 1)}")

        results = walk_forward_evaluation(
            hope, feat_tensor, test_data, FEATURE_NAMES,
            config, device, model_type="hope")

        for sym in symbols:
            if sym not in all_hope_results:
                all_hope_results[sym] = []
            all_hope_results[sym].append(results.get(sym, {'IC': 0, 'sharpe': 0}))

    hope_params = sum(p.numel() for p in hope.parameters())

    # ── [3/5] Train LSTM from scratch ─────────────────────────────
    print(f"\n[3/5] Training LSTM ({len(lstm_seeds)} seeds)...")
    all_lstm_results = {}

    for seed_idx, seed in enumerate(lstm_seeds):
        print(f"\n  ── LSTM seed {seed} ({seed_idx+1}/{len(lstm_seeds)}) ──")
        torch.manual_seed(seed)
        np.random.seed(seed)

        lstm = LSTMBaseline(
            n_features=config.n_features,
            hidden_size=config.lstm_hidden,
            n_layers=config.lstm_layers,
            n_outputs=1,
        ).to(device)

        if seed_idx == 0:
            lstm_params = sum(p.numel() for p in lstm.parameters())
            print(f"  LSTM params: {lstm_params:,}")

        optimizer = torch.optim.AdamW(
            lstm.parameters(), lr=config.lr, weight_decay=config.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs, eta_min=config.min_lr)

        states = init_states_lstm(lstm, n_stocks, torch.device(device))

        for epoch in range(epochs):
            avg_loss, states = train_epoch_minute(
                lstm, feat_tensor, lab_tensor,
                optimizer, scheduler, states, config,
                model_type="lstm", _epoch=epoch)
            if epoch == 0 or (epoch + 1) % 5 == 0:
                print(f"    Epoch {epoch}: loss={avg_loss:.6f}")

        results = walk_forward_evaluation(
            lstm, feat_tensor, test_data, FEATURE_NAMES,
            config, device, model_type="lstm")

        for sym in symbols:
            if sym not in all_lstm_results:
                all_lstm_results[sym] = []
            all_lstm_results[sym].append(results.get(sym, {'IC': 0, 'sharpe': 0}))

    # ── [4/5] Train LightGBM ─────────────────────────────────────
    print(f"\n[4/5] Training LightGBM...")
    try:
        from train_lgbm import train_lgbm
        lgbm = train_lgbm(train_feats, train_labs, val_feats, val_labs,
                          FEATURE_NAMES)

        # LightGBM predictions on test
        lgbm_results = {}
        for sym in symbols:
            X_test = test_feats[sym][FEATURE_NAMES].values
            y_test = test_labs[sym].values
            valid = ~np.isnan(y_test)
            X_v = X_test[valid]
            y_v = y_test[valid]
            if len(X_v) > 10:
                preds = lgbm.predict(X_v)
                from scipy.stats import spearmanr
                ic, _ = spearmanr(preds, y_v)
                if np.isnan(ic):
                    ic = 0.0
                positions = np.tanh(preds * 10)
                pnl = positions * y_v
                if pnl.std() > 1e-10:
                    sh = (pnl.mean() / pnl.std()) * np.sqrt(252 * 375)
                else:
                    sh = 0.0
                lgbm_results[sym] = {'IC': ic, 'sharpe': sh}
            else:
                lgbm_results[sym] = {'IC': 0, 'sharpe': 0}

        has_lgbm = True
    except Exception as e:
        print(f"  ⚠️  LightGBM failed: {e}")
        lgbm_results = {sym: {'IC': 0, 'sharpe': 0} for sym in symbols}
        has_lgbm = False

    # ── [5/5] Print comparison table ──────────────────────────────
    print(f"\n{'='*110}")
    print(f"  HOPE vs LSTM vs LightGBM — FINAL COMPARISON")
    print(f"  HOPE: {hope_params:,} params  |  LSTM: {lstm_params:,} params  |  "
          f"LightGBM: cross-sectional")
    print(f"{'='*110}")

    header = (f"  {'Stock':12s}  "
              f"{'HOPE IC':>12}  {'LSTM IC':>12}  {'LGBM IC':>8}  "
              f"{'H Shrp':>7}  {'L Shrp':>7}  {'G Shrp':>7}  {'Winner':>8}")
    print(header)
    print(f"  {'-'*100}")

    hope_ics, lstm_ics, lgbm_ics = [], [], []

    for sym in sorted(symbols):
        # HOPE: mean ± std across seeds
        h_ics = [r['IC'] for r in all_hope_results.get(sym, [{'IC': 0}])]
        h_mean = np.mean(h_ics)
        h_std = np.std(h_ics) if len(h_ics) > 1 else 0
        h_sharpes = [r['sharpe'] for r in all_hope_results.get(sym, [{'sharpe': 0}])]
        h_sharpe = np.mean(h_sharpes)

        # LSTM: mean ± std across seeds
        l_ics = [r['IC'] for r in all_lstm_results.get(sym, [{'IC': 0}])]
        l_mean = np.mean(l_ics)
        l_std = np.std(l_ics) if len(l_ics) > 1 else 0
        l_sharpes = [r['sharpe'] for r in all_lstm_results.get(sym, [{'sharpe': 0}])]
        l_sharpe = np.mean(l_sharpes)

        # LGBM
        g_ic = lgbm_results.get(sym, {}).get('IC', 0)
        g_sharpe = lgbm_results.get(sym, {}).get('sharpe', 0)

        hope_ics.append(h_mean)
        lstm_ics.append(l_mean)
        lgbm_ics.append(g_ic)

        # Determine winner
        best = max(h_mean, l_mean, g_ic)
        if best == h_mean:
            winner = "HOPE✅"
        elif best == l_mean:
            winner = "LSTM✅"
        else:
            winner = "LGBM✅"

        h_str = f"{h_mean:+.4f}" if len(h_ics) <= 1 else f"{h_mean:+.4f}±{h_std:.3f}"
        l_str = f"{l_mean:+.4f}" if len(l_ics) <= 1 else f"{l_mean:+.4f}±{l_std:.3f}"

        print(f"  {sym:12s}  {h_str:>12}  {l_str:>12}  {g_ic:+8.4f}  "
              f"{h_sharpe:+7.2f}  {l_sharpe:+7.2f}  {g_sharpe:+7.2f}  {winner:>8}")

    # Summary
    print(f"\n  {'MEAN':12s}  {np.mean(hope_ics):+12.4f}  "
          f"{np.mean(lstm_ics):+12.4f}  {np.mean(lgbm_ics):+8.4f}")

    hope_wins = sum(h > l and h > g
                    for h, l, g in zip(hope_ics, lstm_ics, lgbm_ics))
    lstm_wins = sum(l > h and l > g
                    for h, l, g in zip(hope_ics, lstm_ics, lgbm_ics))
    lgbm_wins = sum(g > h and g > l
                    for h, l, g in zip(hope_ics, lstm_ics, lgbm_ics))

    print(f"\n  Wins: HOPE {hope_wins}/{n_stocks}  |  LSTM {lstm_wins}/{n_stocks}  |  "
          f"LGBM {lgbm_wins}/{n_stocks}")
    print(f"{'='*110}")


if __name__ == "__main__":
    main()
