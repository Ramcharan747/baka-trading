"""
HOPE Phase 2: Financial data training with HuggingFace checkpointing.
Runs on Colab T4. Checkpoints every epoch.
Can resume from any checkpoint.

Usage:
    python run_phase2.py --device cuda --epochs 10
    python run_phase2.py --device cuda --epochs 10 --resume
"""
from __future__ import annotations

import argparse
import sys
import time

import numpy as np
import torch

sys.path.insert(0, ".")

from hope_phase1.models.hope import HopeConfig
from hope_phase1.models.lstm_baseline import LSTMBaseline
from data import INSTRUMENTS, prepare_dataset, build_training_batches
from train import train_epoch_finance, detach_states
from evaluate import walk_forward_evaluation
from checkpoint import save_checkpoint, load_checkpoint


# ── HYPERPARAMETERS ──────────────────────────────────────────────────────────
LR = 0.005             # Same as HOPE
CHUNK_SIZE = 64        # Same sequence length


def make_config(n_features: int) -> HopeConfig:
    """LSTM Baseline config for daily financial data."""
    return HopeConfig(
        n_features=n_features,
        d_model=24,
        n_layers=2,           # 2 layers
        n_outputs=1,          # signed return prediction
        chunk_size=CHUNK_SIZE,
    )


def parse_args():
    p = argparse.ArgumentParser(description="LSTM Baseline Phase 2: Financial Data")
    p.add_argument("--token", default=None, help="Upstox API token")
    p.add_argument("--hf_token", default=None, help="HuggingFace write token")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    p.add_argument("--resume", action="store_true", help="Resume from local latest.json")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)

    # Read tokens from args or environment
    import os
    upstox_token = args.token or os.environ.get('UPSTOX_TOKEN')
    hf_token = args.hf_token or os.environ.get('HF_TOKEN')

    # Setup HuggingFace (non-fatal — local checkpoints always work)
    if hf_token:
        try:
            from huggingface_hub import login
            login(token=hf_token)
            print("  ✅ HuggingFace login OK")
        except Exception as e:
            print(f"  ⚠️  HF login failed ({e}), checkpoints will save locally only")

    print("=" * 75)
    print("  LSTM Baseline: Financial Data Training")
    print(f"  Device: {device}  Epochs: {args.epochs}")
    print("=" * 75)

    # ── STEP 1: Data ─────────────────────────────────────────────────
    print("\n[1/4] Loading data...")
    t0 = time.time()
    dataset = prepare_dataset(INSTRUMENTS, token=upstox_token, cost_bps=3.0)

    feat_tensor, lab_tensor, val_data, test_data, common_features = \
        build_training_batches(dataset)
    
    n_stocks, T, n_features = feat_tensor.shape
    print(f"  Stocks: {n_stocks}, Bars: {T}, Features: {n_features}")
    print(f"  Data loaded in {time.time() - t0:.1f}s")

    # ── STEP 2: Model ─────────────────────────────────────────────────
    print("\n[2/4] Initializing model...")
    config = make_config(n_features)
    model = LSTMBaseline(
        n_features=config.n_features,
        hidden_size=config.d_model,
        n_layers=config.n_layers,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {total_params:,}")
    print(f"  Config: d_model={config.d_model} n_layers={config.n_layers}")

    # Resume from checkpoint if requested
    start_epoch = 0
    all_stock_states = None
    if args.resume:
        print("\n[2b] Resuming from checkpoint...")
        model, optimizer, restored_states, start_epoch, _ = \
            load_checkpoint(model, optimizer, device=device)
        # restored_states is the per-layer state; we need per-stock states
        # For now, re-initialize per-stock states (warm-up will recover)
        all_stock_states = None

    # ── STEP 3: Train ─────────────────────────────────────────────────
    print(f"\n[3/4] Training epochs {start_epoch}–{args.epochs - 1}...")
    n_chunks_per_epoch = T // CHUNK_SIZE
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs * n_chunks_per_epoch,
        eta_min=1e-5,
    )
    # Fast-forward scheduler if resuming
    for _ in range(start_epoch * n_chunks_per_epoch):
        scheduler.step()

    best_val_ic = -np.inf

    for epoch in range(start_epoch, args.epochs):
        t_epoch = time.time()

        # Train
        loss, all_stock_states = train_epoch_finance(
            model, feat_tensor, lab_tensor,
            optimizer, chunk_size=CHUNK_SIZE,
            device=device, all_stock_states=all_stock_states,
        )

        # Validate every epoch
        val_results = walk_forward_evaluation(
            model, val_data, common_features,
            chunk_size=CHUNK_SIZE, device=device,
        )
        mean_val_ic = np.mean([v['mean_IC'] for v in val_results.values()])

        elapsed = time.time() - t_epoch
        print(f"\n  Epoch {epoch:2d}: loss={loss:.6f}  val_IC={mean_val_ic:+.4f}  "
              f"({elapsed:.1f}s)")
        for sym, res in sorted(val_results.items()):
            print(f"    {sym:12s}: IC={res['mean_IC']:+.4f}  "
                  f"PnL={res['total_pnl']*100:+.2f}%  "
                  f"WR={res['win_rate']*100:.1f}% ({res['n_trades']}t)")

        # Save checkpoint every epoch
        # Use first stock's state as representative for checkpoint
        metrics = {'loss': loss, 'val_IC': mean_val_ic}
        if all_stock_states and len(all_stock_states) > 0:
            save_checkpoint(
                model, optimizer,
                all_stock_states[0],  # save first stock's state
                epoch,
                epoch * n_chunks_per_epoch,
                metrics,
            )

        if mean_val_ic > best_val_ic:
            best_val_ic = mean_val_ic
            print(f"  ✅ New best val IC: {best_val_ic:+.4f}")

    # ── STEP 4: Final evaluation ───────────────────────────────────────
    print(f"\n[4/4] Final test evaluation...")
    test_results = walk_forward_evaluation(
        model, test_data, common_features,
        chunk_size=CHUNK_SIZE, device=device,
    )

    print(f"\n{'=' * 75}")
    print(f"  PHASE 2 FINAL RESULTS (LSTM BASELINE)")
    print(f"{'=' * 75}")
    for sym, res in sorted(test_results.items()):
        print(f"  {sym:12s}: test_IC={res['mean_IC']:+.4f}  "
              f"PnL={res['total_pnl']*100:+.2f}%  "
              f"WR={res['win_rate']*100:.1f}% ({res['n_trades']}t)  "
              f"pos={res['positive_windows']}/{res['n_windows']}")

    mean_test_ic = np.mean([v['mean_IC'] for v in test_results.values()])
    mean_test_pnl = np.mean([v['total_pnl'] for v in test_results.values()])
    total_test_trades = sum([v['n_trades'] for v in test_results.values()])
    
    pos_stocks = sum(
        v['positive_windows'] > v['n_windows'] // 2
        for v in test_results.values()
    )

    print(f"\n  Mean test IC:           {mean_test_ic:+.4f}")
    print(f"  Mean test PnL:          {mean_test_pnl*100:+.2f}%")
    print(f"  Total test trades:      {total_test_trades}")
    print(f"  Stocks with IC > 0:     {pos_stocks}/{len(test_results)}")

    gate1 = mean_test_ic > 0.01
    gate2 = pos_stocks >= len(test_results) // 2
    gate3 = best_val_ic > 0.005

    print(f"\n  Gate 1 (mean IC > 0.01):      {'✅' if gate1 else '❌'}")
    print(f"  Gate 2 (>50% stocks IC>0):    {'✅' if gate2 else '❌'}")
    print(f"  Gate 3 (best val IC > 0.005): {'✅' if gate3 else '❌'}")

    if all([gate1, gate2, gate3]):
        print("\n  ✅ PHASE 2 PASSED — proceed to Phase 3 "
              "(minute bars + live paper trading)")
    else:
        print("\n  ❌ PHASE 2 FAILED — check feature IC and model convergence")
    print(f"  {'=' * 58}")


if __name__ == "__main__":
    main()
