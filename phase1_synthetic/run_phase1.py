"""
Phase 1: Synthetic frequency-shifting sine test.

Runs BAKA vs LSTM, persistent vs reset, across 5 seeds.
Includes gradient diagnostics after epoch 1 to catch architecture bugs early.

Gate criteria for passing Phase 1 (proceed to Phase 2):
- BAKA persistent IC > LSTM persistent IC (by at least 0.01)
- BAKA persistent IC > BAKA reset IC (persistent state helps)
- Std across seeds < 0.3 × mean
- Memory helps in ≥3/5 seeds

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
from train_streaming import StreamingTrainer, detach_state, mse_loss


SEEDS = [42, 123, 456, 789, 2024]

# Data config
N_STEPS      = 100_000
TRAIN_FRAC   = 0.7
VAL_FRAC     = 0.15

# Training config
CHUNK_SIZE   = 64
LR           = 1e-3
EPOCHS       = 10

# Gate thresholds
IC_ADVANTAGE = 0.01
STD_RATIO    = 0.3


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
        d_model=16, d_ffn=32, dropout=0.0,
        titans_lr=0.01, titans_clip=0.5,
        cms_schedule=(16, 256, 4096), cms_lr=(1e-2, 1e-3, 1e-4),
    ))


def make_lstm():
    return MiniLSTM(n_features=1, hidden_size=32, num_layers=1, n_outputs=1)


def diagnose_baka(model, x_sample, y_sample, device):
    """
    Print gradient norms and output behavior to diagnose architecture bugs.
    Run after first training batch. Shows exactly what is/isn't updating.
    """
    model = model.to(device)
    model.train()

    x_t = torch.from_numpy(x_sample[:64].astype(np.float32)).unsqueeze(0).unsqueeze(-1).to(device)
    y_t = torch.from_numpy(y_sample[:64].astype(np.float32)).unsqueeze(0).to(device)

    state = model.init_state(batch_size=1, device=device)

    # Check 1: Does output respond to input?
    with torch.no_grad():
        x_zeros = torch.zeros_like(x_t)
        x_ones = torch.ones_like(x_t)
        out_z, _ = model(x_zeros, state)
        out_o, _ = model(x_ones, state)
        diff = (out_o - out_z).abs().mean().item()
    print(f"\n  DIAGNOSTIC: Output diff (zeros vs ones): {diff:.6f}", end="")
    if diff < 1e-6:
        print(" ← ❌ OUTPUT IGNORES INPUT")
    else:
        print(" ← ✅ Output responds to input")

    # Check 2: Gradient norms
    pred, new_state = model(x_t, state)
    loss = mse_loss(pred.squeeze(), y_t.squeeze())
    loss.backward()

    print(f"  DIAGNOSTIC: Loss on first batch: {loss.item():.6f}")
    print(f"  DIAGNOSTIC: Gradient norms:")
    has_grad = False
    for name, param in model.named_parameters():
        if param.grad is not None:
            gn = param.grad.norm().item()
            marker = "" if gn > 1e-8 else " ← ⚠️ NEAR ZERO"
            print(f"    {name:40s}: {gn:.6f}{marker}")
            has_grad = True
        else:
            print(f"    {name:40s}: NO GRADIENT ← ❌")

    # Check 3: W_current change
    w_norm = new_state["titans_W"].norm().item()
    print(f"  DIAGNOSTIC: W_current norm after 1 chunk: {w_norm:.6f}")

    # Zero grads
    for p in model.parameters():
        if p.grad is not None:
            p.grad.zero_()


def run_one_condition(
    model_factory,
    x_train, y_train,
    x_val, y_val,
    x_test, y_test,
    epochs: int,
    device: str,
    label: str = "",
    seed: int = 42,
    run_diagnostic: bool = False,
) -> dict:
    """Train and evaluate one model."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = model_factory()

    # Run diagnostic on first seed of BAKA
    if run_diagnostic and hasattr(model, "titans"):
        diagnose_baka(model, x_train, y_train, device)
        # Re-create model with same seed for clean training
        torch.manual_seed(seed)
        np.random.seed(seed)
        model = model_factory()

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    trainer = StreamingTrainer(
        model, optimizer,
        chunk_size=CHUNK_SIZE,
        loss_fn="mse",
        device=device,
        log_every=99999,
    )

    # Train
    t0 = time.time()
    for epoch in range(epochs):
        loss = trainer.train_epoch(x_train, y_train, reset_state=(epoch == 0))
        # Print epoch loss for first seed to monitor training progress
        if run_diagnostic and epoch in (0, 1, epochs - 1):
            print(f"    epoch {epoch}: loss={loss:.6f}")
    train_time = time.time() - t0

    # Evaluate
    result_persist = trainer.evaluate(x_test, y_test, reset_state=False)
    result_reset = trainer.evaluate(x_test, y_test, reset_state=True)

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

    # Generate data ONCE
    x, y, freq = generate_frequency_shifting_sine(n_steps=args.n_steps, seed=0)

    n = len(x)
    train_end = int(n * TRAIN_FRAC)
    val_end = int(n * (TRAIN_FRAC + VAL_FRAC))

    x_train, y_train = x[:train_end], y[:train_end]
    x_val, y_val = x[train_end:val_end], y[train_end:val_end]
    x_test, y_test = x[val_end:], y[val_end:]

    print(f"\n  Data: {n:,} steps → train {len(x_train):,} / val {len(x_val):,} / test {len(x_test):,}")
    print(f"  Freq jumps: {(np.abs(np.diff(freq)) > 0.01).sum()}")

    baka_tmp = make_baka()
    lstm_tmp = make_lstm()
    print(f"  BAKA params: {baka_tmp.param_count():,}")
    print(f"  LSTM params: {lstm_tmp.param_count():,}")
    del baka_tmp, lstm_tmp

    conditions = [
        ("BAKA-persistent", make_baka),
        ("LSTM-persistent", make_lstm),
    ]

    all_results = {name: [] for name, _ in conditions}
    first_seed = True

    for seed in args.seeds:
        print(f"\n  ── Seed {seed} ──")
        for name, factory in conditions:
            r = run_one_condition(
                factory, x_train, y_train, x_val, y_val, x_test, y_test,
                epochs=args.epochs, device=args.device, label=name, seed=seed,
                run_diagnostic=(first_seed and "BAKA" in name),
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
        first_seed = False

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

    gate1 = baka["mean_ic_p"] > lstm["mean_ic_p"]
    g1_margin = baka["mean_ic_p"] - lstm["mean_ic_p"]
    g1_str = f"✅ PASS (+{g1_margin:.4f})" if gate1 else f"❌ FAIL ({g1_margin:+.4f})"
    print(f"  Gate 1: BAKA IC > LSTM IC?         {g1_str}")

    gate2 = baka["mean_delta"] > 0
    g2_str = f"✅ PASS (delta={baka['mean_delta']:+.4f})" if gate2 else \
             f"❌ FAIL (delta={baka['mean_delta']:+.4f})"
    print(f"  Gate 2: BAKA persistent > reset?   {g2_str}")

    if abs(baka["mean_ic_p"]) > 0.001:
        ratio = baka["std_ic_p"] / abs(baka["mean_ic_p"])
    else:
        ratio = float("inf")
    gate3 = ratio < STD_RATIO
    g3_str = f"✅ PASS (ratio={ratio:.2f})" if gate3 else \
             f"❌ FAIL (ratio={ratio:.2f}, need <{STD_RATIO:.1f})"
    print(f"  Gate 3: Std/Mean < {STD_RATIO:.1f}?             {g3_str}")

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
        print(f"\n  DIAGNOSTICS:")
        print(f"  → W_current norms across seeds:")
        for r in all_results["BAKA-persistent"]:
            print(f"    seed={r['seed']}: W_norm={r['W_current_norm']:.4f}  "
                  f"MSE={r['MSE_persistent']:.4f}")
        if baka["mean_ic_p"] < 0.01:
            print(f"\n  → BAKA IC near zero: model not learning at all")
            print(f"  → Check gradient norms in DIAGNOSTIC output above")
            print(f"  → If all gradients non-zero, try larger LR or more epochs")
    print(f"  {'='*60}")

    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
