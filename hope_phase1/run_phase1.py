"""
HOPE Phase 1: Complete experiment — HOPE vs LSTM on frequency-shifting sine.

SUCCESS CRITERIA (must ALL pass to proceed to Phase 2):
  Gate 1: HOPE IC_persistent > LSTM IC_persistent
  Gate 2: HOPE IC_persistent > HOPE IC_reset  (memory helps)
  Gate 3: HOPE IC std < 0.3 × |mean|           (stable across seeds)
  Gate 4: HOPE memory helps in ≥ 3/5 seeds
  Gate 5: W_norm changes across seeds          (DGD actually firing)

Usage:
  python run_phase1.py --device cuda --epochs 10
"""
from __future__ import annotations

import argparse
import sys
import time

import numpy as np
import torch

# Add parent to path for imports
sys.path.insert(0, ".")

from synthetic_data import generate_frequency_shifting_sine
from models.hope import MiniHOPE, HopeConfig
from models.lstm_baseline import LSTMBaseline
from train import (
    train_epoch_hope,
    train_epoch_lstm,
    evaluate_hope,
    evaluate_lstm,
)
from diagnostics import diagnose_hope


# ─── Configuration ──────────────────────────────────────────────────

N_STEPS = 20000
SEEDS = [42, 123, 456, 789, 2024]
CHUNK_SIZE = 64
LR = 1e-3
EPOCHS = 10


def parse_args():
    p = argparse.ArgumentParser(description="HOPE Phase 1 Experiment")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--epochs", type=int, default=EPOCHS)
    p.add_argument("--n-steps", type=int, default=N_STEPS)
    p.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    return p.parse_args()


def make_hope_config() -> HopeConfig:
    """HOPE config matching LSTM param count (~5K params)."""
    return HopeConfig(
        n_features=1,
        d_model=24,
        n_layers=1,       # single layer for Phase 1
        n_outputs=1,
        inner_lr=0.01,
        inner_decay=0.99,
        grad_clip=1.0,
        cms_levels=3,
        cms_schedule=[16, 64, 256],
        cms_lr=[1e-3, 1e-4, 1e-5],
        chunk_size=CHUNK_SIZE,
    )


def make_lstm():
    """LSTM baseline with ~5K params."""
    return LSTMBaseline(n_features=1, hidden_size=24, n_layers=2, n_outputs=1)


def run_single_seed(
    seed: int,
    config: HopeConfig,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    epochs: int,
    device: str,
    run_diagnostic: bool = False,
) -> dict:
    """Run one HOPE + LSTM experiment for a single seed."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    # ─── HOPE ───
    hope = MiniHOPE(config).to(device)

    if run_diagnostic:
        print()
        diagnose_hope(hope, device)
        # Re-create model with same seed for clean training
        torch.manual_seed(seed)
        np.random.seed(seed)
        hope = MiniHOPE(config).to(device)

    optimizer = torch.optim.AdamW(hope.parameters(), lr=LR, weight_decay=0.01)

    # Cosine LR schedule
    n_chunks_per_epoch = len(x_train) // config.chunk_size
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs * n_chunks_per_epoch, eta_min=1e-5
    )

    # Train — states persist across epochs (Fix 2)
    states = hope.init_state(1, torch.device(device))
    for epoch in range(epochs):
        loss, states = train_epoch_hope(
            hope, x_train, y_train, optimizer,
            chunk_size=config.chunk_size, device=device,
            loss_fn="mse", scheduler=scheduler,
            states=states,
        )
        if epoch == 0 or (epoch + 1) % 5 == 0 or epoch == epochs - 1:
            print(f"    epoch {epoch}: loss={loss:.6f}")

    # Evaluate with END-OF-TRAINING state directly (no fast_forward)
    r_persist = evaluate_hope(
        hope, x_test, y_test, states,
        chunk_size=config.chunk_size, device=device,
        reset_state=False,
    )

    # Evaluate with reset state
    r_reset = evaluate_hope(
        hope, x_test, y_test, states,
        chunk_size=config.chunk_size, device=device,
        reset_state=True,
    )

    delta = r_persist["IC"] - r_reset["IC"]
    mem_icon = "✅" if delta > 0 else ""
    print(
        f"    HOPE-persistent       IC_persist={r_persist['IC']:+.4f}  "
        f"IC_reset={r_reset['IC']:+.4f}  delta={delta:+.4f} {mem_icon}  "
        f"MSE={r_persist['MSE']:.6f}  W_norm={r_persist['W_norm']:.4f}"
    )

    # ─── LSTM ───
    torch.manual_seed(seed)
    np.random.seed(seed)

    lstm = make_lstm().to(device)
    lstm_opt = torch.optim.AdamW(lstm.parameters(), lr=LR, weight_decay=0.01)
    lstm_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        lstm_opt, T_max=epochs * n_chunks_per_epoch, eta_min=1e-5
    )

    for epoch in range(epochs):
        train_epoch_lstm(
            lstm, x_train, y_train, lstm_opt,
            chunk_size=config.chunk_size, device=device,
            loss_fn="mse", scheduler=lstm_sched,
        )

    r_lstm = evaluate_lstm(
        lstm, x_test, y_test,
        device=device, chunk_size=config.chunk_size,
    )
    print(
        f"    LSTM-persistent       IC_persist={r_lstm['IC']:+.4f}  "
        f"MSE={r_lstm['MSE']:.6f}"
    )

    return {
        "HOPE": {
            "IC_persist": r_persist["IC"],
            "IC_reset": r_reset["IC"],
            "delta": delta,
            "W_norm": r_persist["W_norm"],
            "MSE": r_persist["MSE"],
        },
        "LSTM": {
            "IC_persist": r_lstm["IC"],
            "MSE": r_lstm["MSE"],
        },
    }


def main():
    args = parse_args()
    device = args.device
    epochs = args.epochs

    print("=" * 70)
    print("  PHASE 1: Synthetic Frequency-Shifting Sine Test (HOPE)")
    print(f"  Device: {device}  Epochs: {epochs}  Seeds: {args.seeds}")
    print("=" * 70)

    # Generate data (same for all seeds — only model seed varies)
    x, y, freq = generate_frequency_shifting_sine(n_steps=args.n_steps, seed=42)
    T = len(x)
    train_end = int(T * 0.70)
    val_end = int(T * 0.85)

    x_train, y_train = x[:train_end], y[:train_end]
    x_test, y_test = x[val_end:], y[val_end:]

    # Count freq jumps
    freq_diffs = np.abs(np.diff(freq[:train_end]))
    n_jumps = (freq_diffs > 0.02).sum()

    config = make_hope_config()

    # Count params
    hope_tmp = MiniHOPE(config)
    lstm_tmp = make_lstm()
    hope_params = sum(p.numel() for p in hope_tmp.parameters())
    lstm_params = sum(p.numel() for p in lstm_tmp.parameters())

    print(f"\n  Data: {T:,} steps → train {train_end:,} / val {val_end-train_end:,} / test {T-val_end:,}")
    print(f"  Freq jumps: {n_jumps}")
    print(f"  HOPE params: {hope_params:,}")
    print(f"  LSTM params: {lstm_params:,}")

    # Run experiments
    all_results = []
    for i, seed in enumerate(args.seeds):
        print(f"\n  ── Seed {seed} ──")
        result = run_single_seed(
            seed=seed,
            config=config,
            x_train=x_train,
            y_train=y_train,
            x_test=x_test,
            y_test=y_test,
            epochs=epochs,
            device=device,
            run_diagnostic=(i == 0),  # diagnostic on first seed only
        )
        all_results.append(result)

    # ─── Results Summary ───
    hope_ics = [r["HOPE"]["IC_persist"] for r in all_results]
    lstm_ics = [r["LSTM"]["IC_persist"] for r in all_results]
    deltas = [r["HOPE"]["delta"] for r in all_results]
    w_norms = [r["HOPE"]["W_norm"] for r in all_results]
    mses = [r["HOPE"]["MSE"] for r in all_results]

    mean_hope = np.mean(hope_ics)
    std_hope = np.std(hope_ics)
    mean_lstm = np.mean(lstm_ics)
    mean_delta = np.mean(deltas)
    mem_wins = sum(d > 0 for d in deltas)
    w_unique = len(set(round(w, 2) for w in w_norms)) > 1

    print(f"\n{'='*70}")
    print(f"  RESULTS SUMMARY")
    print(f"{'='*70}")
    print(f"\n{'Model':>20s}   {'mean IC_p':>10s}  {'std IC_p':>10s}  "
          f"{'mean IC_r':>10s}  {'mean Δ':>10s}   {'mem helps':>10s}")
    print("-" * 80)
    print(f"  {'HOPE-persistent':>18s}   {mean_hope:>+10.4f}  {std_hope:>10.4f}  "
          f"  {np.mean([r['HOPE']['IC_reset'] for r in all_results]):>+10.4f}  "
          f"  {mean_delta:>+10.4f}  {mem_wins}/{len(args.seeds)}")
    print(f"  {'LSTM-persistent':>18s}   {mean_lstm:>+10.4f}  {np.std(lstm_ics):>10.4f}")

    # ─── Gate Check ───
    print(f"\n{'='*70}")
    print(f"  GATE CHECK — Phase 1")
    print(f"{'='*70}")

    gate1 = mean_hope > mean_lstm
    gate2 = mean_delta > 0
    gate3 = (std_hope / (abs(mean_hope) + 1e-8)) < 0.3
    gate4 = mem_wins >= 3
    gate5 = w_unique

    checks = [
        (gate1, "HOPE IC > LSTM IC?",
         f"{'✅ PASS' if gate1 else '❌ FAIL'} ({mean_hope-mean_lstm:+.4f})"),
        (gate2, "HOPE persistent > reset?",
         f"{'✅ PASS' if gate2 else '❌ FAIL'} (delta={mean_delta:+.4f})"),
        (gate3, "Std/|Mean| < 0.3?",
         f"{'✅ PASS' if gate3 else '❌ FAIL'} "
         f"(ratio={'inf' if abs(mean_hope) < 1e-8 else f'{std_hope/abs(mean_hope):.2f}'}"
         f", need <0.3)"),
        (gate4, "Memory helps >=3/5 seeds?",
         f"{'✅ PASS' if gate4 else '❌ FAIL'} ({mem_wins}/5)"),
        (gate5, "W_norm varies across seeds?",
         f"{'✅ PASS' if gate5 else '❌ FAIL'} ({[round(w, 2) for w in w_norms]})"),
    ]

    for _, name, result in checks:
        print(f"  Gate: {name:35s} {result}")

    passed = all(g for g, _, _ in checks)
    print(f"\n  {'='*60}")
    if passed:
        print(f"  ✅ PHASE 1 PASSED — proceed to Phase 2")
    else:
        print(f"  ❌ PHASE 1 FAILED — Fix architecture before Phase 2")
        print(f"\n  DIAGNOSTICS:")
        print(f"  → W_current norms across seeds:")
        for i, seed in enumerate(args.seeds):
            print(
                f"    seed={seed}: W_norm={w_norms[i]:.4f}  MSE={mses[i]:.4f}"
            )
        if not gate1:
            print(f"  → HOPE IC ({mean_hope:.4f}) < LSTM IC ({mean_lstm:.4f})")
        if not gate2:
            print(f"  → Memory doesn't help (mean delta = {mean_delta:+.4f})")
    print(f"  {'='*60}")


if __name__ == "__main__":
    main()
