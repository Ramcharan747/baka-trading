"""
Learning rate sweep — find the peak and base of model performance
across a range of training learning rates.

For each lr, trains the model and measures walk-forward IC on validation.
Prints a clear table showing where IC peaks and where it degrades.

Usage:
    python lr_sweep.py \
        --kaggle-path /path/to/dataset \
        --symbol COALINDIA --start 2024-01-01 --end 2024-06-30 \
        --model baka --use-kaggle-indicators \
        --window 60 --batch-size 128 --epochs 5 \
        --device cuda
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr

import kaggle_loader
from kaggle_loader import compute_indicators
from features import compute_features, make_labels, align_features_labels
from models import BAKAFinanceConfig, MiniBAKAFinance, LSTMBaseline
from train import TrainConfig, train_one_window, predict


ARTIFACT_DIR = Path(__file__).parent / "artifacts"
ARTIFACT_DIR.mkdir(exist_ok=True)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--symbol", default="COALINDIA")
    p.add_argument("--start", default="2024-01-01")
    p.add_argument("--end", default="2024-06-30")
    p.add_argument("--kaggle-path", required=True)
    p.add_argument("--use-kaggle-indicators", action="store_true")
    p.add_argument("--model", default="baka", choices=["lstm", "baka"])
    p.add_argument("--cms-ablate", type=int, default=-1)
    p.add_argument("--window", type=int, default=60)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--loss", default="ic", choices=["ic", "sharpe"])
    p.add_argument("--train-frac", type=float, default=0.6)
    p.add_argument("--val-frac", type=float, default=0.1)
    p.add_argument("--cost-bps", type=float, default=3.0)
    p.add_argument("--lookahead", type=int, default=5)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def make_model(args, n_features: int) -> torch.nn.Module:
    if args.model == "lstm":
        return LSTMBaseline(n_features=n_features, hidden_dim=32, n_layers=1,
                            n_outputs=1, dropout=0.1)
    cfg = BAKAFinanceConfig(n_features=n_features, n_outputs=1)
    model = MiniBAKAFinance(cfg)
    if args.cms_ablate >= 0:
        model.ablate_cms_level(args.cms_ablate)
    return model


def main():
    args = parse_args()
    print(f"\n{'='*70}")
    print(f"  Learning Rate Sweep: {args.model.upper()} on {args.symbol}")
    print(f"{'='*70}")

    # --- Load data ---
    print("\n[1/3] Loading data...")
    raw = kaggle_loader.load_kaggle_dataset(args.kaggle_path, args.symbol)
    raw = raw.loc[args.start:args.end]
    ohlcv, _ = kaggle_loader.split_ohlcv_and_indicators(raw)
    df = ohlcv

    if args.use_kaggle_indicators:
        extra_feats = compute_indicators(raw)
    else:
        extra_feats = None

    feats = compute_features(df)
    if extra_feats is not None and not extra_feats.empty:
        extra_feats = extra_feats.apply(pd.to_numeric, errors="coerce")
        keep_cols = [c for c in extra_feats.columns
                     if extra_feats[c].notna().sum() > len(extra_feats) * 0.5]
        extra_feats = extra_feats[keep_cols]
        feats = feats.join(extra_feats, how="left").ffill().dropna()

    labels = make_labels(df, lookahead=args.lookahead, cost_bps=args.cost_bps)
    X, y = align_features_labels(feats, labels)
    print(f"  Features: {X.shape}, Labels: {y.shape}")

    n_features = X.shape[1]

    # Split: train / val / test
    train_end = int(len(X) * args.train_frac)
    val_end = int(len(X) * (args.train_frac + args.val_frac))

    Xtrain = X.iloc[:train_end].to_numpy()
    ytrain = y.iloc[:train_end].to_numpy()
    Xval = X.iloc[train_end:val_end].to_numpy()
    yval = y.iloc[train_end:val_end].to_numpy()
    Xtest = X.iloc[val_end:].to_numpy()
    ytest = y.iloc[val_end:].to_numpy()

    print(f"  Train: {len(Xtrain):,}  Val: {len(Xval):,}  Test: {len(Xtest):,}")

    # --- Sweep ---
    lr_values = [1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2]

    print(f"\n[2/3] Sweeping {len(lr_values)} learning rates...")
    print(f"  Each trains for {args.epochs} epochs\n")

    rows = []
    for lr in lr_values:
        print(f"  --- lr = {lr:.0e} ---")
        cfg = TrainConfig(
            window=args.window, batch_size=args.batch_size,
            epochs=args.epochs, lr=lr, loss=args.loss,
            device=args.device, log_every=9999,  # suppress per-step logs
        )

        model = make_model(args, n_features)
        model = train_one_window(model, Xtrain, ytrain, cfg)

        # Validate
        val_preds, val_aligned = predict(model, Xval, yval, cfg)
        if len(val_preds) >= 20:
            val_ic, val_p = spearmanr(val_preds, val_aligned)
        else:
            val_ic, val_p = 0.0, 1.0

        # Test (held out)
        test_preds, test_aligned = predict(model, Xtest, ytest, cfg)
        if len(test_preds) >= 20:
            test_ic, test_p = spearmanr(test_preds, test_aligned)
        else:
            test_ic, test_p = 0.0, 1.0

        # Prediction stats
        pred_std = float(test_preds.std()) if len(test_preds) > 0 else 0.0
        pred_range = float(test_preds.max() - test_preds.min()) if len(test_preds) > 0 else 0.0
        pct_pos = float((test_preds > test_preds.mean()).sum() / max(1, len(test_preds)) * 100)

        rows.append({
            "lr": lr,
            "val_ic": float(val_ic),
            "val_p": float(val_p),
            "test_ic": float(test_ic),
            "test_p": float(test_p),
            "pred_std": pred_std,
            "pred_range": pred_range,
            "sign_diversity": pct_pos,
        })
        print(f"    val IC={val_ic:+.4f} (p={val_p:.4f})  "
              f"test IC={test_ic:+.4f} (p={test_p:.4f})  "
              f"pred_std={pred_std:.4f}")

    # --- Results ---
    df_results = pd.DataFrame(rows)

    print(f"\n{'='*70}")
    print(f"  LEARNING RATE SWEEP RESULTS — {args.model.upper()}")
    print(f"{'='*70}")
    print(f"\n{'lr':>10s}  {'val_IC':>8s}  {'val_p':>8s}  {'test_IC':>8s}  "
          f"{'test_p':>8s}  {'pred_std':>9s}  {'sign_div':>8s}")
    print("-" * 75)

    best_val_ic = df_results["val_ic"].abs().max()
    for _, r in df_results.iterrows():
        marker = " <-- PEAK" if abs(r["val_ic"]) == best_val_ic else ""
        print(f"  {r['lr']:>8.0e}  {r['val_ic']:>+7.4f}  {r['val_p']:>8.4f}  "
              f"{r['test_ic']:>+7.4f}  {r['test_p']:>8.4f}  "
              f"{r['pred_std']:>9.4f}  {r['sign_diversity']:>6.1f}%{marker}")

    # Find peak
    best_idx = df_results["val_ic"].abs().idxmax()
    best = df_results.loc[best_idx]
    print(f"\n  PEAK lr = {best['lr']:.0e}")
    print(f"    Val  IC = {best['val_ic']:+.4f}")
    print(f"    Test IC = {best['test_ic']:+.4f}")

    # Check for overfitting: val_ic >> test_ic
    if abs(best["val_ic"]) > 2 * abs(best["test_ic"]) and abs(best["val_ic"]) > 0.02:
        print(f"    ⚠️  Possible overfitting: val IC much higher than test IC")

    # Identify where it degrades (the "base")
    lowest_idx = df_results["val_ic"].abs().idxmin()
    lowest = df_results.loc[lowest_idx]
    print(f"\n  BASE lr = {lowest['lr']:.0e}")
    print(f"    Val  IC = {lowest['val_ic']:+.4f}")
    print(f"    Test IC = {lowest['test_ic']:+.4f}")

    # Save
    out_path = ARTIFACT_DIR / f"lr_sweep_{args.symbol}_{args.model}.json"
    df_results.to_json(out_path, orient="records", indent=2)
    print(f"\nSaved to: {out_path}")

    print(f"\n{'='*70}")
    print(f"  RECOMMENDATION")
    print(f"{'='*70}")
    print(f"  Use --lr {best['lr']:.0e} for {args.model.upper()} on {args.symbol}")
    print(f"  Then run threshold_analysis.py with that lr to tune paper trading")
    print(f"{'='*70}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted.")
        sys.exit(130)
