"""
Master script: download -> features -> IC test -> train -> walk-forward -> paper trade.

Usage:
    python run_experiment.py --symbol NIFTY --start 2022-01-01 --end 2024-12-31 \
        --interval 1d --model lstm

    python run_experiment.py --symbol COALINDIA --model baka --cms-ablate 3

    # Auto-download from Kaggle and train:
    python run_experiment.py --source kaggle --kaggle-download \
        --symbol COALINDIA --start 2024-01-01 --end 2024-12-31 \
        --model baka --use-kaggle-indicators

The script is intentionally long but linear — read top to bottom.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import data_download
import kaggle_loader
from kaggle_loader import compute_indicators
from features import compute_features, make_labels, align_features_labels
from ic_test import (
    ic_test,
    check_all_features_for_lookahead,
    regime_ic_analysis,
)
from models import BAKAFinanceConfig, MiniBAKAFinance, LSTMBaseline
from train import TrainConfig, train_one_window, predict, walk_forward_evaluation
from paper_trading import PaperTradingSimulator, SimConfig, run_simulation


ARTIFACT_DIR = Path(__file__).parent / "artifacts"
ARTIFACT_DIR.mkdir(exist_ok=True)


# --------------------------------------------------------------- parse

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--symbol", default="NIFTY")
    p.add_argument("--start", default="2022-01-01")
    p.add_argument("--end", default="2024-12-31")
    p.add_argument("--interval", default="1d",
                   help="1d for daily, 1m/5m for intraday (yfinance limits apply)")
    p.add_argument("--source", default="auto",
                   choices=["auto", "yfinance", "nsepy", "kite", "kaggle"])
    p.add_argument("--kaggle-path", default=None,
                   help="Path to a Kaggle dataset directory or CSV (e.g. /kaggle/input/...)")
    p.add_argument("--kaggle-download", action="store_true",
                   help="Auto-download the Nifty 500 dataset via kagglehub")
    p.add_argument("--kaggle-key", default=None,
                   help="Kaggle API key (KGAT_...) for download")
    p.add_argument("--use-kaggle-indicators", action="store_true",
                   help="Compute and merge technical indicators (RSI, MACD, etc.) as extra features")
    p.add_argument("--lookahead", type=int, default=5)
    p.add_argument("--cost-bps", type=float, default=3.0)
    p.add_argument("--model", default="lstm", choices=["lstm", "baka"])
    p.add_argument("--cms-ablate", type=int, default=-1,
                   help="(BAKA only) Zero out CMS level k and evaluate; -1 = no ablation")
    p.add_argument("--cms-schedule", default="minute",
                   help="CMS update schedule: 'minute' [16,256,4096,65536] or "
                        "'daily' [5,21,63,252] (week/month/quarter/year)")
    p.add_argument("--window", type=int, default=32)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=1e-3,
                   help="Training learning rate (default 1e-3)")
    p.add_argument("--loss", default="ic", choices=["ic", "sharpe"])
    p.add_argument("--train-frac", type=float, default=0.6)
    p.add_argument("--val-frac", type=float, default=0.1)
    p.add_argument("--skip-paper-trading", action="store_true")
    p.add_argument("--signal-threshold", type=float, default=None,
                   help="Override signal_threshold for paper trading (applied AFTER z-score normalization)")
    p.add_argument("--stop-loss", type=float, default=None,
                   help="Override stop_loss_pct for paper trading (e.g. 0.02 = 2%%)")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


# --------------------------------------------------------------- factories

def _parse_cms_schedule(schedule_arg: str):
    """Parse CMS schedule: preset name or custom comma-separated values."""
    PRESETS = {
        "minute": ((16, 256, 4096, 65536), (1e-3, 1e-4, 1e-5, 1e-6)),
        "daily":  ((5, 21, 63, 252),       (1e-2, 5e-3, 1e-3, 5e-4)),
    }
    if schedule_arg in PRESETS:
        return PRESETS[schedule_arg]
    # Custom: "5,21,63,252"
    vals = tuple(int(x) for x in schedule_arg.split(","))
    # Use default LRs scaled: fastest=1e-2, each 3-5× slower
    default_lrs = (1e-2, 5e-3, 1e-3, 5e-4)[:len(vals)]
    return vals, default_lrs


def make_model(args, n_features: int) -> torch.nn.Module:
    if args.model == "lstm":
        return LSTMBaseline(
            n_features=n_features,
            hidden_dim=32,
            n_layers=1,
            n_outputs=1,
            dropout=0.1,
        )
    cms_schedule, cms_lr = _parse_cms_schedule(args.cms_schedule)
    cfg = BAKAFinanceConfig(
        n_features=n_features, n_outputs=1,
        cms_levels=len(cms_schedule),
        cms_schedule=cms_schedule,
        cms_lr=cms_lr,
    )
    model = MiniBAKAFinance(cfg)
    if args.cms_ablate >= 0:
        print(f"[BAKA] CMS schedule: {cms_schedule}  LRs: {cms_lr}")
        print(f"[BAKA] Ablating CMS level {args.cms_ablate}")
        model.ablate_cms_level(args.cms_ablate)
    else:
        print(f"[BAKA] CMS schedule: {cms_schedule}  LRs: {cms_lr}")
    return model


# --------------------------------------------------------------- main

def main():
    args = parse_args()
    print(f"\n================ BAKA Trading Experiment ================")
    print(f"Symbol: {args.symbol}  range: {args.start} -> {args.end}  interval: {args.interval}")
    print(f"Model : {args.model}  device: {args.device}")
    print("=" * 60)

    # --- Step 1: data ---
    print("\n[1/6] Loading data...")
    if args.source == "kaggle" or args.kaggle_path or args.kaggle_download:
        # Resolve the dataset path
        kaggle_path = args.kaggle_path
        if not kaggle_path:
            if args.kaggle_download:
                print("  Downloading dataset via kagglehub...")
                kaggle_path = kaggle_loader.download_dataset(args.kaggle_key)
                print(f"  Dataset at: {kaggle_path}")
            else:
                raise ValueError(
                    "--source kaggle requires --kaggle-path or --kaggle-download"
                )
        raw = kaggle_loader.load_kaggle_dataset(kaggle_path, args.symbol)
        raw = raw.loc[args.start:args.end]
        ohlcv, _ = kaggle_loader.split_ohlcv_and_indicators(raw)
        df = ohlcv
        # Dataset has NO pre-computed indicators — compute them if requested
        if args.use_kaggle_indicators:
            extra_feats = compute_indicators(raw)
            print(f"  Kaggle: {len(df):,} rows, OHLCV + {len(extra_feats.columns)} computed indicators")
        else:
            extra_feats = None
            print(f"  Kaggle: {len(df):,} rows, OHLCV only")
    else:
        df = data_download.load_or_download(
            args.symbol, args.start, args.end, args.interval, args.source
        )
        extra_feats = None
        print(f"  {len(df)} rows, columns: {list(df.columns)}")

    # --- Step 2: features + labels ---
    print("\n[2/6] Computing features and labels...")
    feats = compute_features(df)
    if extra_feats is not None and not extra_feats.empty:
        # Coerce non-numeric columns and align.
        extra_feats = extra_feats.apply(pd.to_numeric, errors="coerce")
        # Drop columns that are mostly NaN after coercion.
        keep_cols = [c for c in extra_feats.columns if extra_feats[c].notna().sum() > len(extra_feats) * 0.5]
        extra_feats = extra_feats[keep_cols]
        feats = feats.join(extra_feats, how="left").ffill().dropna()
        print(f"  Merged {len(extra_feats.columns)} Kaggle indicators -> total features: {feats.shape[1]}")
    labels = make_labels(df, lookahead=args.lookahead, cost_bps=args.cost_bps)
    X, y = align_features_labels(feats, labels)
    print(f"  features: {X.shape}  labels: {y.shape}")
    print(f"  feature cols: {list(X.columns)}")

    # --- Step 3: lookahead-bias + IC test ---
    print("\n[3/6] Lookahead-bias check (raises if any feature has |IC| > 0.3)...")
    check_all_features_for_lookahead(X, y)
    print("  OK — no suspicious IC values.")

    print("\n[4/6] IC test per feature (keep |IC|>0.02 and p<0.05)...")
    kept, ic_df = ic_test(X, y)
    if not kept:
        print("  WARNING: zero features passed the IC threshold.")
        print("  Consider: longer history, different lookahead, or richer features.")
    ic_df.to_csv(ARTIFACT_DIR / f"ic_{args.symbol}_{args.interval}.csv", index=False)

    # Regime analysis on the full feature set (not just kept).
    try:
        regime_df = regime_ic_analysis(X, y, df["close"].reindex(X.index))
        regime_df.to_csv(ARTIFACT_DIR / f"regime_{args.symbol}_{args.interval}.csv", index=False)
    except Exception as e:
        print(f"  (regime analysis skipped: {e})")

    # If anything passed, restrict to kept features for training.
    if kept:
        X = X[kept]
        print(f"  Training on kept features: {kept}")

    # --- Step 4: walk-forward evaluation ---
    print("\n[5/6] Walk-forward evaluation...")
    cfg = TrainConfig(
        window=args.window,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        loss=args.loss,
        device=args.device,
    )
    n_features = X.shape[1]
    wf = walk_forward_evaluation(
        model_factory=lambda: make_model(args, n_features),
        features=X,
        labels=y,
        train_frac=args.train_frac,
        val_frac=args.val_frac,
        cfg=cfg,
    )
    wf_records = [asdict(r) for r in wf]
    wf_path = ARTIFACT_DIR / f"wf_{args.symbol}_{args.model}_{args.interval}.json"
    wf_path.write_text(json.dumps(wf_records, indent=2))
    print(f"  Wrote {wf_path}")

    # --- Step 5: paper trading on held-out tail ---
    if args.skip_paper_trading:
        return
    print("\n[6/6] Paper trading on held-out tail...")
    holdout_start = int(len(X) * (args.train_frac + args.val_frac))
    if holdout_start >= len(X) - args.window - 1:
        print("  (not enough tail data for paper trading; skipping)")
        return

    # Train on everything up to holdout_start, predict the tail.
    Xtrain = X.iloc[:holdout_start].to_numpy()
    ytrain = y.iloc[:holdout_start].to_numpy()
    Xtail = X.iloc[holdout_start:].to_numpy()
    ytail = y.iloc[holdout_start:].to_numpy()

    model = make_model(args, n_features)
    train_one_window(model, Xtrain, ytrain, cfg)
    preds, _ = predict(model, Xtail, ytail, cfg)

    # Z-score normalize predictions so paper trading thresholds work properly.
    # Raw model outputs are uncalibrated scores, not return predictions.
    preds_mean = preds.mean()
    preds_std = preds.std() + 1e-8
    preds_z = (preds - preds_mean) / preds_std
    print(f"  Raw  preds: mean={preds_mean:+.4f} std={preds_std:.4f} range=[{preds.min():+.4f}, {preds.max():+.4f}]")
    print(f"  Z-scored:   mean={preds_z.mean():+.4f} std={preds_z.std():.4f} range=[{preds_z.min():+.4f}, {preds_z.max():+.4f}]")

    # Align predictions back to timestamps (preds start at index window-1 in the tail)
    pred_idx = X.iloc[holdout_start + cfg.window - 1 : holdout_start + cfg.window - 1 + len(preds)].index
    preds_s = pd.Series(preds_z, index=pred_idx)
    prices_s = df["close"].reindex(pred_idx)

    sim_cfg = SimConfig(
        cost_bps=args.cost_bps,
        signal_threshold=args.signal_threshold if args.signal_threshold is not None else 0.5,
        stop_loss_pct=args.stop_loss if args.stop_loss is not None else 0.02,
    )
    print(f"  Paper trading with: threshold={sim_cfg.signal_threshold}, stop_loss={sim_cfg.stop_loss_pct*100:.1f}%")
    sim, metrics = run_simulation(preds_s, prices_s, sim_cfg)
    m_path = ARTIFACT_DIR / f"paper_{args.symbol}_{args.model}_{args.interval}.json"
    m_path.write_text(json.dumps(metrics, indent=2))

    eq = sim.equity_series()
    if len(eq) > 0:
        eq.to_csv(ARTIFACT_DIR / f"equity_{args.symbol}_{args.model}_{args.interval}.csv")

    print("\n================ DONE ================")
    print(f"Mean walk-forward IC: {np.mean([r.ic for r in wf]):+.4f}" if wf else "")
    print(f"Paper Sharpe (ann.) : {metrics['sharpe']:+.2f}")
    print(f"Artifacts in: {ARTIFACT_DIR}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted.")
        sys.exit(130)
