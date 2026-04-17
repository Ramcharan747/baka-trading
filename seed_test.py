"""
Seed stability test — validate that the best configuration is robust
to random initialization before scaling up to Stage 2.

Runs the ablate-3 BAKA config (CMS=[5,21,63], no yearly memory)
across N seeds and reports mean ± std of Sharpe ratio.

Gate: mean Sharpe > 1.0 AND std < 0.8  →  proceed to Stage 2
      Otherwise                         →  signal is noise, don't scale

Usage:
    python seed_test.py \
        --kaggle-path /path/to/dataset \
        --symbol COALINDIA --start 2020-01-01 --end 2024-12-31 \
        --lr 1e-05 --signal-threshold 1.50 --stop-loss 0.005 \
        --window 60 --batch-size 128 --epochs 5 \
        --device cuda
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ARTIFACT_DIR = Path(__file__).parent / "artifacts"
ARTIFACT_DIR.mkdir(exist_ok=True)

SEEDS = [42, 123, 456, 789, 2024]

# Gate thresholds for Stage 2 readiness
MEAN_SHARPE_GATE = 1.0
STD_SHARPE_GATE  = 0.8


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--symbol", default="COALINDIA")
    p.add_argument("--start", default="2020-01-01")
    p.add_argument("--end", default="2024-12-31")
    p.add_argument("--kaggle-path", required=True)
    p.add_argument("--use-kaggle-indicators", action="store_true", default=True)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--signal-threshold", type=float, default=1.5)
    p.add_argument("--stop-loss", type=float, default=0.005)
    p.add_argument("--window", type=int, default=60)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def run_one_seed(args, seed: int) -> dict:
    """Run run_experiment.py with the given seed and return paper trading metrics."""

    # Build CLI
    cmd = [
        sys.executable, "run_experiment.py",
        "--source", "kaggle",
        "--kaggle-path", args.kaggle_path,
        "--symbol", args.symbol,
        "--start", args.start,
        "--end", args.end,
        "--model", "baka",
        "--cms-schedule", "daily",
        "--cms-ablate", "3",         # ablate yearly — our best config
        "--use-kaggle-indicators",
        "--lr", str(args.lr),
        "--signal-threshold", str(args.signal_threshold),
        "--stop-loss", str(args.stop_loss),
        "--window", str(args.window),
        "--batch-size", str(args.batch_size),
        "--epochs", str(args.epochs),
        "--device", args.device,
        "--seed", str(seed),
    ]

    print(f"\n  Running seed={seed}...")
    print(f"  {' '.join(cmd[-20:])}")  # last few args for readability

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=Path(__file__).parent,
    )

    if result.returncode != 0:
        print(f"  ❌ seed={seed} FAILED:")
        print(result.stderr[-2000:])
        return {"seed": seed, "sharpe": float("nan"), "return": float("nan"),
                "max_dd": float("nan"), "trades": 0, "win_rate": float("nan"),
                "error": True}

    # Parse paper metrics from artifact JSON
    paper_path = ARTIFACT_DIR / f"paper_{args.symbol}_baka_1d.json"
    try:
        m = json.loads(paper_path.read_text())
        sharpe   = m.get("sharpe", float("nan"))
        ret      = m.get("total_return", float("nan"))
        max_dd   = m.get("max_dd", float("nan"))
        trades   = m.get("trades", 0)
        win_rate = m.get("win_rate", float("nan"))
    except Exception as e:
        print(f"  ⚠️  Could not read artifact: {e}")
        sharpe = ret = max_dd = win_rate = float("nan")
        trades = 0

    return {
        "seed": seed,
        "sharpe": sharpe,
        "return": ret,
        "max_dd": max_dd,
        "trades": trades,
        "win_rate": win_rate,
        "error": False,
    }


def main():
    args = parse_args()

    print(f"\n{'='*70}")
    print(f"  SEED STABILITY TEST — BAKA ablate-3 (CMS=[5,21,63])")
    print(f"  Symbol: {args.symbol}  {args.start} → {args.end}")
    print(f"  LR: {args.lr:.0e}  threshold: {args.signal_threshold:.2f}σ  "
          f"stop_loss: {args.stop_loss*100:.1f}%")
    print(f"  Seeds: {args.seeds}")
    print(f"  Gate: mean Sharpe > {MEAN_SHARPE_GATE:.1f} AND std < {STD_SHARPE_GATE:.1f}")
    print(f"{'='*70}")

    rows = []
    for seed in args.seeds:
        row = run_one_seed(args, seed)
        rows.append(row)
        status = "✅" if not row.get("error") else "❌"
        print(f"\n  {status} seed={seed:4d}:  "
              f"Sharpe={row['sharpe']:+.2f}  "
              f"ret={row['return']*100:+.2f}%  "
              f"maxDD={row['max_dd']*100:+.2f}%  "
              f"trades={row['trades']:3d}  "
              f"wr={row['win_rate']*100:.1f}%")

    valid = [r for r in rows if not r.get("error") and not np.isnan(r["sharpe"])]
    sharpes = [r["sharpe"] for r in valid]

    if len(sharpes) == 0:
        print("\n❌ All seeds failed. Cannot evaluate stability.")
        sys.exit(1)

    mean_sharpe = float(np.mean(sharpes))
    std_sharpe  = float(np.std(sharpes))
    min_sharpe  = float(np.min(sharpes))
    max_sharpe  = float(np.max(sharpes))

    print(f"\n{'='*70}")
    print(f"  STABILITY REPORT ({len(sharpes)}/{len(args.seeds)} seeds succeeded)")
    print(f"{'='*70}")
    print(f"  Mean Sharpe : {mean_sharpe:+.3f}")
    print(f"  Std  Sharpe : {std_sharpe:.3f}")
    print(f"  Min  Sharpe : {min_sharpe:+.3f}")
    print(f"  Max  Sharpe : {max_sharpe:+.3f}")
    print(f"  Range       : {max_sharpe - min_sharpe:.3f}")
    print()

    gate_mean = mean_sharpe >= MEAN_SHARPE_GATE
    gate_std  = std_sharpe  <= STD_SHARPE_GATE

    if gate_mean and gate_std:
        verdict = "✅ PASS — PROCEED TO STAGE 2"
        detail  = (f"Signal is robust across seeds. "
                   f"Scaling to d_model=64, n_layers=3 is justified.")
    elif gate_mean and not gate_std:
        verdict = "⚠️  UNSTABLE — Mean is good but variance is too high"
        detail  = (f"Mean Sharpe {mean_sharpe:.2f} > {MEAN_SHARPE_GATE:.1f} ✅  "
                   f"but std {std_sharpe:.2f} > {STD_SHARPE_GATE:.1f} ❌\n"
                   f"  Recommendation: train for more epochs (--epochs 10) or "
                   f"adjust threshold before scaling.")
    elif not gate_mean and std_sharpe < 0.3:
        verdict = "❌ CONSISTENTLY WEAK — Not noise, but too weak to scale"
        detail  = (f"Mean Sharpe {mean_sharpe:.2f} < {MEAN_SHARPE_GATE:.1f} ❌  "
                   f"but std {std_sharpe:.2f} is low. "
                   f"Try different threshold/LR before scaling.")
    else:
        verdict = "❌ FAIL — Result is noise, do NOT scale to Stage 2"
        detail  = (f"Mean Sharpe {mean_sharpe:.2f} < {MEAN_SHARPE_GATE:.1f} AND "
                   f"std {std_sharpe:.2f} > {STD_SHARPE_GATE:.1f}. "
                   f"Model is not learning a stable signal.")

    print(f"  VERDICT: {verdict}")
    print(f"  {detail}")
    print(f"{'='*70}")

    # Save results
    out = {
        "config": {
            "symbol": args.symbol,
            "start": args.start,
            "end": args.end,
            "model": "baka",
            "cms_schedule": "daily_3level",
            "cms_ablate": 3,
            "lr": args.lr,
            "signal_threshold": args.signal_threshold,
            "stop_loss": args.stop_loss,
        },
        "seeds": rows,
        "summary": {
            "mean_sharpe": mean_sharpe,
            "std_sharpe": std_sharpe,
            "min_sharpe": min_sharpe,
            "max_sharpe": max_sharpe,
            "n_valid": len(sharpes),
            "gate_pass": gate_mean and gate_std,
        },
        "verdict": verdict,
    }
    out_path = ARTIFACT_DIR / f"seed_test_{args.symbol}_baka.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nSaved to: {out_path}")

    return 0 if (gate_mean and gate_std) else 1


if __name__ == "__main__":
    sys.exit(main())
