"""
Phase 1: Synthetic frequency-shifting sine test.

Runs 4 conditions × 5 seeds:
1. BAKA persistent (correct — state flows continuously)
2. BAKA reset every 60 steps (wrong — what the previous code did)
3. LSTM persistent
4. LSTM reset every 60 steps

Gate criteria for passing Phase 1 (proceed to Phase 2):
- BAKA persistent IC > LSTM persistent IC (by at least 0.01)
- BAKA persistent IC > BAKA reset IC (persistent state helps)
- Std across seeds < 0.3 × mean
- All above hold across majority of seeds

If FAIL: prints W_current norms to diagnose whether DGD is updating.

Usage:
    python run_phase1.py [--device cuda] [--epochs 10]
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

from synthetic_data import generate_frequency_shifting_sine
from mini_baka import MiniBAKA, MiniBAKAConfig
from mini_lstm import MiniLSTM
from train_streaming import StreamingTrainer, detach_state


SEEDS = [42, 123, 456, 789, 2024]

# Data config
N_STEPS      = 100_000
TRAIN_FRAC   = 0.7
VAL_FRAC     = 0.15
# Test = remaining 0.15

# Training config
CHUNK_SIZE   = 64
LR           = 1e-3
EPOCHS       = 10

# Gate thresholds
IC_ADVANTAGE = 0.01   # BAKA persistent must beat LSTM persistent by this
STD_RATIO    = 0.3    # std/mean must be below this


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--epochs", type=int, default=EPOCHS)
    p.add_argument("--n-steps", type=int, default=N_STEPS)
    p.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    return p.parse_args()


def make_baka():
    return MiniBAKA(MiniBAKAConfig(
        n_features=1, n_outputs=1,
        d_model=16, n_heads=2, d_ffn=32, dropout=0.0,
        titans_chunk=16, titans_lr=0.01,
        cms_levels=3, cms_schedule=(16, 256, 4096), cms_lr=(1e-2, 1e-3, 1e-4),
    ))


def make_lstm():
    return MiniLSTM(n_features=1, hidden_size=32, num_layers=1, n_outputs=1)


def run_one_condition(
    model_factory,
    x_train, y_train,
    x_val, y_val,
    x_test, y_test,
    epochs: int,
    device: str,
    reset_interval: int = 0,
    label: str = "",
    seed: int = 42,
) -> dict:
    """
    Train and evaluate one model.

    reset_interval=0: persistent state (correct)
    reset_interval=N: reset state every N steps (simulates old wrong approach)
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = model_factory()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    trainer = StreamingTrainer(
        model, optimizer,
        chunk_size=CHUNK_SIZE,
        loss_fn="mse",
        device=device,
        log_every=99999,  # suppress per-chunk logs
    )

    # Train
    t0 = time.time()
    for epoch in range(epochs):
        loss = trainer.train_epoch(x_train, y_train, reset_state=(epoch == 0))
    train_time = time.time() - t0

    # Evaluate: persistent state (continues from training)
    result_persist = trainer.evaluate(x_test, y_test, reset_state=False)

    # Evaluate: reset state (to measure memory contribution)
    result_reset = trainer.evaluate(x_test, y_test, reset_state=True)

    # Diagnostic: W_current norm (BAKA only)
    w_norm = float("nan")
    if hasattr(model, "titans") and trainer.state is not None:
        w_norm = float(trainer.state["titans_W"].norm().item())

    return {
        "label": label,
        "seed": seed,
        "IC_persistent": result_persist["IC"],
        "IC_reset": result_reset["IC"],
        "MSE_persistent": result_persist["MSE"],
        "MSE_reset": result_reset["MSE"],
        "memory_delta": result_persist["IC"] - result_reset["IC"],
        "W_current_norm": w_norm,
        "train_time": train_time,
    }


def main():
    args = parse_args()

    print(f"\n{'='*70}")
    print(f"  PHASE 1: Synthetic Frequency-Shifting Sine Test")
    print(f"  Device: {args.device}  Epochs: {args.epochs}  Seeds: {args.seeds}")
    print(f"{'='*70}")

    # Generate data ONCE (same data for all models/seeds)
    x, y, freq = generate_frequency_shifting_sine(n_steps=args.n_steps, seed=0)

    n = len(x)
    train_end = int(n * TRAIN_FRAC)
    val_end = int(n * (TRAIN_FRAC + VAL_FRAC))

    x_train, y_train = x[:train_end], y[:train_end]
    x_val, y_val = x[train_end:val_end], y[train_end:val_end]
    x_test, y_test = x[val_end:], y[val_end:]

    print(f"\n  Data: {n:,} steps → train {len(x_train):,} / val {len(x_val):,} / test {len(x_test):,}")
    print(f"  Freq jumps: {(np.abs(np.diff(freq)) > 0.01).sum()}")

    # Print model sizes
    baka_tmp = make_baka()
    lstm_tmp = make_lstm()
    print(f"  BAKA params: {baka_tmp.param_count():,}")
    print(f"  LSTM params: {lstm_tmp.param_count():,}")
    del baka_tmp, lstm_tmp

    # Run all conditions
    conditions = [
        ("BAKA-persistent", make_baka),
        ("LSTM-persistent", make_lstm),
    ]

    all_results = {name: [] for name, _ in conditions}

    for seed in args.seeds:
        print(f"\n  ── Seed {seed} ──")
        for name, factory in conditions:
            r = run_one_condition(
                factory, x_train, y_train, x_val, y_val, x_test, y_test,
                epochs=args.epochs, device=args.device, label=name, seed=seed,
            )
            all_results[name].append(r)
            mem_str = ""
            if not np.isnan(r["W_current_norm"]):
                mem_str = f"  W_norm={r['W_current_norm']:.4f}"
            status = "✅" if r["memory_delta"] > 0 else "  "
            print(f"    {name:20s}  IC_persist={r['IC_persistent']:+.4f}  "
                  f"IC_reset={r['IC_reset']:+.4f}  "
                  f"delta={r['memory_delta']:+.4f} {status}"
                  f"  MSE={r['MSE_persistent']:.6f}{mem_str}")

    # ================================================================ Report
    print(f"\n{'='*70}")
    print(f"  RESULTS SUMMARY")
    print(f"{'='*70}")

    print(f"\n{'Model':>20s}  {'mean IC_p':>10s}  {'std IC_p':>10s}  "
          f"{'mean IC_r':>10s}  {'mean Δ':>10s}  {'mem helps':>10s}")
    print("-" * 80)

    summary = {}
    for name in ["BAKA-persistent", "LSTM-persistent"]:
        results = all_results[name]
        ics_p = [r["IC_persistent"] for r in results]
        ics_r = [r["IC_reset"] for r in results]
        deltas = [r["memory_delta"] for r in results]
        mem_helps = sum(1 for d in deltas if d > 0)

        mean_p = np.mean(ics_p)
        std_p = np.std(ics_p)
        mean_r = np.mean(ics_r)
        mean_d = np.mean(deltas)

        summary[name] = {
            "mean_ic_p": mean_p, "std_ic_p": std_p,
            "mean_ic_r": mean_r, "mean_delta": mean_d,
            "mem_helps": mem_helps, "ics_p": ics_p,
        }

        print(f"  {name:18s}  {mean_p:>+10.4f}  {std_p:>10.4f}  "
              f"{mean_r:>+10.4f}  {mean_d:>+10.4f}  "
              f"{mem_helps}/{len(results)}")

    # ================================================================ Gate Check
    print(f"\n{'='*70}")
    print(f"  GATE CHECK — Phase 1")
    print(f"{'='*70}")

    baka = summary["BAKA-persistent"]
    lstm = summary["LSTM-persistent"]

    # Gate 1: BAKA persistent > LSTM persistent
    gate1 = baka["mean_ic_p"] > lstm["mean_ic_p"]
    g1_margin = baka["mean_ic_p"] - lstm["mean_ic_p"]
    g1_str = f"✅ PASS (+{g1_margin:.4f})" if gate1 else f"❌ FAIL ({g1_margin:+.4f})"
    print(f"  Gate 1: BAKA IC > LSTM IC?         {g1_str}")

    # Gate 2: BAKA persistent > BAKA reset (memory helps)
    gate2 = baka["mean_delta"] > 0
    g2_str = f"✅ PASS (delta={baka['mean_delta']:+.4f})" if gate2 else \
             f"❌ FAIL (delta={baka['mean_delta']:+.4f})"
    print(f"  Gate 2: BAKA persistent > reset?   {g2_str}")

    # Gate 3: Low variance across seeds
    if abs(baka["mean_ic_p"]) > 0.001:
        ratio = baka["std_ic_p"] / abs(baka["mean_ic_p"])
    else:
        ratio = float("inf")
    gate3 = ratio < STD_RATIO
    g3_str = f"✅ PASS (ratio={ratio:.2f})" if gate3 else \
             f"❌ FAIL (ratio={ratio:.2f}, need <{STD_RATIO:.1f})"
    print(f"  Gate 3: Std/Mean < {STD_RATIO:.1f}?             {g3_str}")

    # Gate 4: Memory helps in majority of seeds
    gate4 = baka["mem_helps"] >= 3
    g4_str = f"✅ PASS ({baka['mem_helps']}/5)" if gate4 else \
             f"❌ FAIL ({baka['mem_helps']}/5)"
    print(f"  Gate 4: Memory helps ≥3/5 seeds?   {g4_str}")

    all_pass = gate1 and gate2 and gate3 and gate4

    print(f"\n  {'='*60}")
    if all_pass:
        print(f"  ✅ PHASE 1 PASSED — Proceed to Phase 2 (Financial Data)")
    else:
        print(f"  ❌ PHASE 1 FAILED — Fix architecture before Phase 2")
        # Print diagnostics
        print(f"\n  DIAGNOSTICS:")
        if not gate2:
            print(f"  → W_current norms across seeds:")
            for r in all_results["BAKA-persistent"]:
                print(f"    seed={r['seed']}: W_norm={r['W_current_norm']:.4f}")
            print(f"  → If W_norm stays near 0, DGD is not updating W_current")
            print(f"  → If W_norm explodes, increase gradient clipping")
    print(f"  {'='*60}")

    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
