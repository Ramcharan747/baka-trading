"""
Phase 3: Main HOPE training script.

Usage:
  python run_phase3.py --seed 0 --mode dev --device cuda --token $UPSTOX_TOKEN --hf_token $HF_TOKEN
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
from data import load_all_stocks, build_splits
from features import compute_features, FEATURE_NAMES
from labels import compute_labels
from ic_test import ic_test
from models.hope import MiniHOPE, HopeConfig
from train import train_epoch_minute, init_states_hope, detach_states
from evaluate import walk_forward_evaluation
from checkpoint import save_checkpoint, load_checkpoint


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
    parser = argparse.ArgumentParser(description="HOPE Phase 3 Training")
    parser.add_argument("--seed",     type=int, default=0)
    parser.add_argument("--mode",     default="dev", choices=["dev", "prod"])
    parser.add_argument("--device",   default="cuda")
    parser.add_argument("--epochs",   type=int, default=None)
    parser.add_argument("--token",    default=None)
    parser.add_argument("--hf_token", default=None)
    parser.add_argument("--hf_repo",  default="Baka7/hope-finance-phase3")
    args = parser.parse_args()

    config = PhaseConfig(mode=args.mode)
    device = args.device
    seed = args.seed

    # Override config based on mode
    if args.mode == "dev":
        from_date = config.dev_from_date
        to_date = config.dev_to_date
        epochs = args.epochs or config.dev_epochs
        stock_list = config.dev_stocks
    else:
        from_date = config.from_date
        to_date = config.to_date
        epochs = args.epochs or config.epochs
        stock_list = None  # use all

    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    upstox_token = args.token or os.environ.get('UPSTOX_TOKEN')
    hf_token = args.hf_token or os.environ.get('HF_TOKEN')

    if hf_token:
        try:
            from huggingface_hub import login
            login(token=hf_token)
        except Exception:
            pass

    print("=" * 70)
    print(f"  HOPE Phase 3 Training")
    print(f"  Mode: {args.mode}  Seed: {seed}  Device: {device}  Epochs: {epochs}")
    print("=" * 70)

    # ── [1/5] Load and validate data ──────────────────────────────
    print("\n[1/5] Loading data...")
    instruments = INSTRUMENTS.copy()
    if stock_list:
        instruments = {k: v for k, v in instruments.items()
                       if k in stock_list or k == "NIFTY50"}

    all_data = load_all_stocks(instruments, upstox_token, from_date, to_date)

    # Separate NIFTY50 index
    nifty_df = all_data.pop("NIFTY50", None)
    if nifty_df is not None:
        print(f"  NIFTY50 index: {len(nifty_df)} bars")
    else:
        print("  ⚠️  NIFTY50 index not available, cross-asset features = 0")

    # ── [2/5] Compute features ────────────────────────────────────
    print("\n[2/5] Computing features...")
    dataset = {}  # {sym: (features_df, labels_series)}

    # Compute sector mean closes
    sector_closes = {}
    for sector, members in SECTORS.items():
        available = [s for s in members if s in all_data]
        if available:
            sector_closes[sector] = sum(
                all_data[s]['close'] for s in available
            ) / len(available)

    for sym in sorted(all_data.keys()):
        df = all_data[sym]
        sector = STOCK_SECTOR.get(sym)
        sector_close = sector_closes.get(sector)

        feats = compute_features(df, nifty_df, sector_close)
        labs = compute_labels(df['close'], config.label_horizon,
                              config.transaction_cost)

        # Drop rows where label is NaN (last horizon rows)
        valid = labs.notna()
        feats = feats[valid].reset_index(drop=True)
        labs = labs[valid].reset_index(drop=True)

        dataset[sym] = (feats, labs)
        print(f"  {sym}: {len(feats)} bars, {feats.shape[1]} features")

    n_stocks = len(dataset)
    print(f"\n  Total: {n_stocks} stocks with {config.n_features} features")

    # ── [3/5] IC diagnostic ───────────────────────────────────────
    print("\n[3/5] Feature IC diagnostic (first stock)...")
    first_sym = sorted(dataset.keys())[0]
    first_feats, first_labs = dataset[first_sym]
    ic_results = ic_test(first_feats, first_labs)
    n_kept = ic_results['keep'].sum()
    print(ic_results.to_string())
    print(f"\n  {first_sym}: {n_kept}/{len(ic_results)} features significant")

    # ── Build training tensors ────────────────────────────────────
    # Split temporally
    symbols = sorted(dataset.keys())

    # Use train_end/val_end from config (for prod) or split proportionally (for dev)
    if args.mode == "dev":
        # For dev: use 70/15/15 split
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
        # For prod: use date-based splits from config
        import pandas as pd
        train_end = pd.Timestamp(config.train_end)
        val_end = pd.Timestamp(config.val_end)
        # Need datetime — get from all_data
        train_feats, train_labs = {}, {}
        val_feats, val_labs = {}, {}
        test_feats, test_labs = {}, {}

        for sym in symbols:
            f, l = dataset[sym]
            df_orig = all_data[sym]
            dt = pd.to_datetime(df_orig['datetime'])
            # Trim dt to valid range (NaN labels dropped)
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

    # Stack training data into tensors (trim to min length)
    min_train = min(len(train_feats[s]) for s in symbols)
    feat_arrays = []
    lab_arrays = []
    for sym in symbols:
        feat_arrays.append(
            train_feats[sym][FEATURE_NAMES].iloc[:min_train].values)
        lab_arrays.append(
            train_labs[sym].iloc[:min_train].values)

    feat_tensor = torch.tensor(
        np.stack(feat_arrays, axis=0), dtype=torch.float32).to(device)
    lab_tensor = torch.tensor(
        np.stack(lab_arrays, axis=0), dtype=torch.float32).to(device)

    print(f"  Training tensor: {feat_tensor.shape}")

    # Build test_data dict for evaluation
    test_data = {}
    for sym in symbols:
        test_data[sym] = (test_feats[sym], test_labs[sym])

    # ── [4/5] Train HOPE ──────────────────────────────────────────
    print(f"\n[4/5] Training HOPE for {epochs} epochs...")
    hope_config = make_hope_config(config)
    model = MiniHOPE(hope_config).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  HOPE params: {n_params:,}")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=config.min_lr)

    # Try to resume from checkpoint
    model, optimizer, states, start_epoch, step = load_checkpoint(
        model, optimizer, device=device)

    if states is None:
        states = init_states_hope(model, n_stocks, torch.device(device))
        start_epoch = 0

    # Memory health check before training
    print(f"\n  Memory state health check (pre-training):")
    for li, layer_state in enumerate(states):
        for k, v in layer_state.items():
            if isinstance(v, torch.Tensor):
                print(f"    Layer {li} {k}: norm={v.norm().item():.4f}, "
                      f"shape={list(v.shape)}")

    for epoch in range(start_epoch, epochs):
        t0 = time.time()
        avg_loss, states = train_epoch_minute(
            model, feat_tensor, lab_tensor,
            optimizer, scheduler, states, config,
            model_type="hope")
        dt = time.time() - t0

        # Gradient flow check (first epoch only)
        if epoch == start_epoch:
            print(f"\n  Gradient flow check:")
            for name, p in model.named_parameters():
                if p.grad is not None:
                    print(f"    {name}: grad_norm={p.grad.norm().item():.6f}")
                elif p.requires_grad:
                    print(f"    {name}: NO GRADIENT ⚠️")

        print(f"  Epoch {epoch}: loss={avg_loss:.6f}  ({dt:.1f}s)  "
              f"lr={optimizer.param_groups[0]['lr']:.6f}")

        # Save checkpoint
        metrics = {'loss': avg_loss}
        save_checkpoint(
            model, optimizer, states, epoch,
            step=epoch * (min_train // config.chunk_size),
            metrics=metrics, hf_token=hf_token)

    # ── [5/5] Evaluate on test set ────────────────────────────────
    print(f"\n[5/5] Evaluating on test set...")
    results = walk_forward_evaluation(
        model, feat_tensor, test_data, FEATURE_NAMES,
        config, device, model_type="hope")

    print(f"\n  {'Stock':12s}  {'IC':>8}  {'Sharpe':>8}")
    print(f"  {'-'*32}")
    ics = []
    for sym in sorted(results.keys()):
        r = results[sym]
        ics.append(r['IC'])
        print(f"  {sym:12s}  {r['IC']:+8.4f}  {r['sharpe']:+8.2f}")

    print(f"\n  Mean IC: {np.mean(ics):+.4f}")
    print("=" * 70)
    print("  Phase 3 training complete.")
    print("=" * 70)


if __name__ == "__main__":
    main()
