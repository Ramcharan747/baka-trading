"""
Threshold + stop-loss analysis with z-score normalized predictions.

Trains a model, z-score normalizes the raw predictions, then does a 2D
sweep over (signal_threshold, stop_loss_pct) to find the optimal
operating point.

The key insight: raw model outputs are uncalibrated scores (mean ~0.7,
std ~1.0 for BAKA; mean ~0.85, std ~0.06 for LSTM). After z-scoring,
threshold=0.5 means "trade only when prediction is 0.5σ from mean",
which has proper statistical meaning.

Usage (Colab):
    python threshold_analysis.py \
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

import kaggle_loader
from kaggle_loader import compute_indicators
from features import compute_features, make_labels, align_features_labels
from models import BAKAFinanceConfig, MiniBAKAFinance, LSTMBaseline
from train import TrainConfig, train_one_window, predict
from paper_trading import PaperTradingSimulator, SimConfig, run_simulation


ARTIFACT_DIR = Path(__file__).parent / "artifacts"
ARTIFACT_DIR.mkdir(exist_ok=True)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--symbol", default="COALINDIA")
    p.add_argument("--start", default="2024-01-01")
    p.add_argument("--end", default="2024-06-30")
    p.add_argument("--source", default="kaggle")
    p.add_argument("--kaggle-path", required=True)
    p.add_argument("--use-kaggle-indicators", action="store_true")
    p.add_argument("--model", default="baka", choices=["lstm", "baka"])
    p.add_argument("--cms-ablate", type=int, default=-1)
    p.add_argument("--cms-schedule", default="minute",
                   help="'minute' [16,256,4096,65536] or 'daily' [5,21,63,252]")
    p.add_argument("--window", type=int, default=60)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--loss", default="ic", choices=["ic", "sharpe"])
    p.add_argument("--train-frac", type=float, default=0.6)
    p.add_argument("--val-frac", type=float, default=0.1)
    p.add_argument("--cost-bps", type=float, default=3.0)
    p.add_argument("--lookahead", type=int, default=5)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def _parse_cms_schedule(s):
    PRESETS = {
        "minute": ((16, 256, 4096, 65536), (1e-3, 1e-4, 1e-5, 1e-6)),
        "daily":  ((5, 21, 63, 252),       (1e-2, 5e-3, 1e-3, 5e-4)),
    }
    if s in PRESETS: return PRESETS[s]
    vals = tuple(int(x) for x in s.split(","))
    return vals, (1e-2, 5e-3, 1e-3, 5e-4)[:len(vals)]


def make_model(args, n_features: int) -> torch.nn.Module:
    if args.model == "lstm":
        return LSTMBaseline(n_features=n_features, hidden_dim=32, n_layers=1,
                            n_outputs=1, dropout=0.1)
    cms_schedule, cms_lr = _parse_cms_schedule(args.cms_schedule)
    cfg = BAKAFinanceConfig(n_features=n_features, n_outputs=1,
                            cms_levels=len(cms_schedule),
                            cms_schedule=cms_schedule, cms_lr=cms_lr)
    model = MiniBAKAFinance(cfg)
    if args.cms_ablate >= 0:
        model.ablate_cms_level(args.cms_ablate)
    return model


def main():
    args = parse_args()
    print(f"\n{'='*70}")
    print(f"  Threshold Analysis: {args.model.upper()} on {args.symbol}")
    print(f"  (with z-score normalization)")
    print(f"{'='*70}")

    # --- Load data ---
    print("\n[1/4] Loading data...")
    raw = kaggle_loader.load_kaggle_dataset(args.kaggle_path, args.symbol)
    raw = raw.loc[args.start:args.end]
    ohlcv, _ = kaggle_loader.split_ohlcv_and_indicators(raw)
    df = ohlcv

    if args.use_kaggle_indicators:
        extra_feats = compute_indicators(raw)
        print(f"  {len(df):,} rows, {len(extra_feats.columns)} computed indicators")
    else:
        extra_feats = None
        print(f"  {len(df):,} rows, OHLCV only")

    # --- Features + labels ---
    print("\n[2/4] Computing features...")
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

    # --- Train + predict on held-out tail ---
    print("\n[3/4] Training and predicting on held-out tail...")
    cfg = TrainConfig(window=args.window, batch_size=args.batch_size,
                      epochs=args.epochs, lr=args.lr, loss=args.loss,
                      device=args.device)
    n_features = X.shape[1]

    holdout_start = int(len(X) * (args.train_frac + args.val_frac))
    Xtrain = X.iloc[:holdout_start].to_numpy()
    ytrain = y.iloc[:holdout_start].to_numpy()
    Xtail = X.iloc[holdout_start:].to_numpy()
    ytail = y.iloc[holdout_start:].to_numpy()

    model = make_model(args, n_features)
    train_one_window(model, Xtrain, ytrain, cfg)
    preds_raw, _ = predict(model, Xtail, ytail, cfg)

    # --- Z-score normalize predictions ---
    raw_mean = preds_raw.mean()
    raw_std = preds_raw.std() + 1e-8
    preds = (preds_raw - raw_mean) / raw_std

    # Align predictions back to timestamps
    pred_idx = X.iloc[holdout_start + cfg.window - 1:
                       holdout_start + cfg.window - 1 + len(preds)].index
    preds_s = pd.Series(preds, index=pred_idx)
    prices_s = df["close"].reindex(pred_idx)

    # --- Distribution analysis ---
    print(f"\n{'='*70}")
    print(f"  RAW PREDICTION DISTRIBUTION")
    print(f"{'='*70}")
    print(f"  Count:  {len(preds_raw):,}")
    print(f"  Mean:   {raw_mean:+.6f}")
    print(f"  Std:    {raw_std:.6f}")
    print(f"  Min:    {preds_raw.min():+.6f}")
    print(f"  Max:    {preds_raw.max():+.6f}")
    pct_positive = (preds_raw > 0).sum() / len(preds_raw) * 100
    print(f"  %% positive: {pct_positive:.1f}%")

    print(f"\n{'='*70}")
    print(f"  Z-SCORED PREDICTION DISTRIBUTION")
    print(f"{'='*70}")
    abs_z = np.abs(preds)
    print(f"  Mean:   {preds.mean():+.6f}")
    print(f"  Std:    {preds.std():.6f}")
    print(f"  Min:    {preds.min():+.6f}")
    print(f"  Max:    {preds.max():+.6f}")
    pct_positive_z = (preds > 0).sum() / len(preds) * 100
    print(f"  %% positive (long signals): {pct_positive_z:.1f}%")
    print(f"  %% negative (short signals): {100-pct_positive_z:.1f}%")
    print(f"\n  |z-score| percentiles:")
    for p in [10, 25, 50, 75, 90, 95, 99]:
        val = np.percentile(abs_z, p)
        print(f"    P{p:>2d}: {val:.4f}")

    # --- 2D sweep: signal_threshold × stop_loss_pct ---
    print(f"\n{'='*70}")
    print(f"  2D SWEEP: signal_threshold × stop_loss")
    print(f"{'='*70}")

    thresholds = [0.0, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]
    stop_losses = [0.005, 0.01, 0.02, 0.03, 0.05, 0.10]

    rows = []
    for thresh in thresholds:
        for sl in stop_losses:
            sim_cfg = SimConfig(
                cost_bps=args.cost_bps,
                signal_threshold=thresh,
                stop_loss_pct=sl,
            )
            _, metrics = run_simulation(preds_s.copy(), prices_s.copy(), sim_cfg)
            rows.append({
                "threshold": thresh,
                "stop_loss": sl,
                "trades": metrics["trades"],
                "return_pct": metrics["total_return"] * 100,
                "sharpe": metrics["sharpe"],
                "max_dd_pct": metrics["max_dd"] * 100,
                "win_rate": metrics["win_rate"] * 100,
                "avg_win": metrics["avg_win"],
                "avg_loss": metrics["avg_loss"],
            })

    df_sweep = pd.DataFrame(rows)

    # Print as a grid: threshold on rows, stop_loss on columns, value = Sharpe
    print(f"\n  SHARPE RATIO GRID (threshold × stop_loss):")
    print(f"  {'thresh':>8s}", end="")
    for sl in stop_losses:
        print(f"  SL={sl*100:.1f}%", end="")
    print()
    print("  " + "-" * (10 + 10 * len(stop_losses)))
    best_sharpe = df_sweep["sharpe"].max()
    for thresh in thresholds:
        print(f"  {thresh:>8.2f}", end="")
        for sl in stop_losses:
            row = df_sweep[(df_sweep["threshold"] == thresh) &
                           (df_sweep["stop_loss"] == sl)].iloc[0]
            marker = " *" if row["sharpe"] == best_sharpe else "  "
            print(f"  {row['sharpe']:>+5.2f}{marker}", end="")
        print()

    # Print trade count grid
    print(f"\n  TRADE COUNT GRID (threshold × stop_loss):")
    print(f"  {'thresh':>8s}", end="")
    for sl in stop_losses:
        print(f"  SL={sl*100:.1f}%", end="")
    print()
    print("  " + "-" * (10 + 10 * len(stop_losses)))
    for thresh in thresholds:
        print(f"  {thresh:>8.2f}", end="")
        for sl in stop_losses:
            row = df_sweep[(df_sweep["threshold"] == thresh) &
                           (df_sweep["stop_loss"] == sl)].iloc[0]
            print(f"  {row['trades']:>7.0f}", end="")
        print()

    # Print return grid
    print(f"\n  RETURN %% GRID (threshold × stop_loss):")
    print(f"  {'thresh':>8s}", end="")
    for sl in stop_losses:
        print(f"  SL={sl*100:.1f}%", end="")
    print()
    print("  " + "-" * (10 + 10 * len(stop_losses)))
    for thresh in thresholds:
        print(f"  {thresh:>8.2f}", end="")
        for sl in stop_losses:
            row = df_sweep[(df_sweep["threshold"] == thresh) &
                           (df_sweep["stop_loss"] == sl)].iloc[0]
            print(f"  {row['return_pct']:>+6.2f}%", end="")
        print()

    # Find best
    best_idx = df_sweep["sharpe"].idxmax()
    best = df_sweep.loc[best_idx]
    print(f"\n{'='*70}")
    print(f"  BEST OPERATING POINT")
    print(f"{'='*70}")
    print(f"  Signal threshold : {best['threshold']:.2f} σ")
    print(f"  Stop loss        : {best['stop_loss']*100:.1f}%")
    print(f"  Sharpe           : {best['sharpe']:+.2f}")
    print(f"  Return           : {best['return_pct']:+.2f}%")
    print(f"  Max Drawdown     : {best['max_dd_pct']:+.2f}%")
    print(f"  Trades           : {best['trades']:.0f}")
    print(f"  Win Rate         : {best['win_rate']:.1f}%")
    print(f"  Avg Win / Loss   : {best['avg_win']:+.2f} / {best['avg_loss']:+.2f}")
    print(f"{'='*70}")

    # Save
    sweep_path = ARTIFACT_DIR / f"sweep_{args.symbol}_{args.model}.json"
    df_sweep.to_json(sweep_path, orient="records", indent=2)

    stats = {
        "model": args.model,
        "symbol": args.symbol,
        "n_predictions": len(preds),
        "raw_mean": float(raw_mean),
        "raw_std": float(raw_std),
        "z_pct_positive": float(pct_positive_z),
        "z_pct_negative": float(100 - pct_positive_z),
        "best_threshold": float(best["threshold"]),
        "best_stop_loss": float(best["stop_loss"]),
        "best_sharpe": float(best["sharpe"]),
        "best_return_pct": float(best["return_pct"]),
        "best_trades": int(best["trades"]),
    }
    stats_path = ARTIFACT_DIR / f"pred_stats_{args.symbol}_{args.model}.json"
    stats_path.write_text(json.dumps(stats, indent=2))
    print(f"\nSaved to: {sweep_path}")
    print(f"Saved to: {stats_path}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted.")
        sys.exit(130)
