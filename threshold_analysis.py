"""
Threshold analysis and paper-trading sweep.

Trains a model, captures raw predictions on the held-out tail, then
sweeps signal_threshold to find the optimal operating point.

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
    print(f"\n{'='*60}")
    print(f"  Threshold Analysis: {args.model.upper()} on {args.symbol}")
    print(f"{'='*60}")

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
                      epochs=args.epochs, loss=args.loss, device=args.device)
    n_features = X.shape[1]

    holdout_start = int(len(X) * (args.train_frac + args.val_frac))
    Xtrain = X.iloc[:holdout_start].to_numpy()
    ytrain = y.iloc[:holdout_start].to_numpy()
    Xtail = X.iloc[holdout_start:].to_numpy()
    ytail = y.iloc[holdout_start:].to_numpy()

    model = make_model(args, n_features)
    train_one_window(model, Xtrain, ytrain, cfg)
    preds, _ = predict(model, Xtail, ytail, cfg)

    # Align predictions back to timestamps
    pred_idx = X.iloc[holdout_start + cfg.window - 1:
                       holdout_start + cfg.window - 1 + len(preds)].index
    preds_s = pd.Series(preds, index=pred_idx)
    prices_s = df["close"].reindex(pred_idx)

    # --- Prediction distribution analysis ---
    print(f"\n{'='*60}")
    print(f"  PREDICTION DISTRIBUTION ANALYSIS")
    print(f"{'='*60}")
    abs_preds = np.abs(preds)
    percentiles = [10, 25, 50, 75, 90, 95, 99]
    print(f"\n  Total predictions: {len(preds):,}")
    print(f"  Mean:   {preds.mean():+.6f}")
    print(f"  Std:    {preds.std():.6f}")
    print(f"  Min:    {preds.min():+.6f}")
    print(f"  Max:    {preds.max():+.6f}")
    print(f"\n  |prediction| percentiles:")
    for p in percentiles:
        val = np.percentile(abs_preds, p)
        print(f"    P{p:>2d}: {val:.6f}")

    # --- Threshold sweep ---
    print(f"\n{'='*60}")
    print(f"  THRESHOLD SWEEP")
    print(f"{'='*60}")

    # Build threshold candidates from prediction distribution
    thresholds = sorted(set([
        0.0001, 0.0002, 0.0005,
        0.001, 0.002, 0.003, 0.005,
        0.01, 0.02, 0.03, 0.05,
        np.percentile(abs_preds, 50),
        np.percentile(abs_preds, 60),
        np.percentile(abs_preds, 70),
        np.percentile(abs_preds, 75),
        np.percentile(abs_preds, 80),
        np.percentile(abs_preds, 85),
        np.percentile(abs_preds, 90),
        np.percentile(abs_preds, 95),
    ]))

    rows = []
    for thresh in thresholds:
        sim_cfg = SimConfig(
            cost_bps=args.cost_bps,
            signal_threshold=thresh,
        )
        _, metrics = run_simulation(preds_s, prices_s, sim_cfg)
        would_trade = int(np.sum(abs_preds > thresh))
        rows.append({
            "threshold": thresh,
            "trades": metrics["trades"],
            "signals_above": would_trade,
            "return_pct": metrics["total_return"] * 100,
            "sharpe": metrics["sharpe"],
            "max_dd_pct": metrics["max_dd"] * 100,
            "win_rate": metrics["win_rate"] * 100,
        })

    df_sweep = pd.DataFrame(rows)
    print(f"\n{'threshold':>12s}  {'trades':>6s}  {'signals':>7s}  {'return%':>8s}  "
          f"{'sharpe':>7s}  {'maxDD%':>7s}  {'winrate':>7s}")
    print("-" * 70)
    for _, r in df_sweep.iterrows():
        marker = " <-- best" if r["sharpe"] == df_sweep["sharpe"].max() else ""
        print(f"  {r['threshold']:>10.6f}  {r['trades']:>6.0f}  {r['signals_above']:>7.0f}  "
              f"{r['return_pct']:>+7.2f}%  {r['sharpe']:>+6.2f}  "
              f"{r['max_dd_pct']:>+6.2f}%  {r['win_rate']:>5.1f}%{marker}")

    # Find best threshold
    best_idx = df_sweep["sharpe"].idxmax()
    best = df_sweep.loc[best_idx]
    print(f"\n{'='*60}")
    print(f"  BEST THRESHOLD: {best['threshold']:.6f}")
    print(f"  Sharpe={best['sharpe']:+.2f}  Return={best['return_pct']:+.2f}%  "
          f"MaxDD={best['max_dd_pct']:+.2f}%  Trades={best['trades']:.0f}  "
          f"WinRate={best['win_rate']:.1f}%")
    print(f"{'='*60}")

    # Save sweep results
    sweep_path = ARTIFACT_DIR / f"sweep_{args.symbol}_{args.model}.json"
    df_sweep.to_json(sweep_path, orient="records", indent=2)
    print(f"\nSaved sweep to: {sweep_path}")

    # Save prediction stats
    stats_path = ARTIFACT_DIR / f"pred_stats_{args.symbol}_{args.model}.json"
    stats = {
        "model": args.model,
        "symbol": args.symbol,
        "n_predictions": len(preds),
        "pred_mean": float(preds.mean()),
        "pred_std": float(preds.std()),
        "pred_min": float(preds.min()),
        "pred_max": float(preds.max()),
        "abs_percentiles": {str(p): float(np.percentile(abs_preds, p))
                           for p in percentiles},
        "best_threshold": float(best["threshold"]),
        "best_sharpe": float(best["sharpe"]),
    }
    stats_path.write_text(json.dumps(stats, indent=2))
    print(f"Saved prediction stats to: {stats_path}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted.")
        sys.exit(130)
