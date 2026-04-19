"""
Phase 2b: HOPE vs LSTM comparison on financial data.
Trains LSTM from scratch on same data as HOPE.
Loads HOPE from local/HF checkpoint (no retraining).
Compares IC, trade count, win rate side by side.
"""
from __future__ import annotations

import argparse
import numpy as np
import torch
import sys
sys.path.insert(0, ".")

from models.hope import MiniHOPE, HopeConfig
from models.lstm_baseline import LSTMBaseline
from data import INSTRUMENTS, prepare_dataset, build_training_batches
from train import train_epoch_finance, detach_states
from evaluate import walk_forward_evaluation
from backtest import run_backtest
from checkpoint import load_checkpoint
from losses import ic_loss

CHUNK_SIZE = 64
LR = 1e-3
EPOCHS = 20


def make_hope_config(n_features):
    return HopeConfig(
        n_features=n_features, d_model=24, n_layers=2, n_outputs=1,
        inner_lr=0.01, inner_decay=0.99, grad_clip=1.0,
        cms_levels=3, cms_schedule=[5, 21, 63],
        cms_lr=[1e-3, 1e-4, 1e-5], chunk_size=CHUNK_SIZE,
    )


def train_lstm(lstm, feat_tensor, lab_tensor, device, epochs=EPOCHS):
    """Train LSTM with same streaming paradigm as HOPE."""
    optimizer = torch.optim.AdamW(
        lstm.parameters(), lr=LR, weight_decay=0.01)
    n_stocks, T, _ = feat_tensor.shape
    n_chunks = T // CHUNK_SIZE
    feat_tensor = feat_tensor.to(device)
    lab_tensor = lab_tensor.to(device)

    for epoch in range(epochs):
        lstm.train()
        # Reset LSTM hidden state at start of each epoch
        h = torch.zeros(lstm.lstm.num_layers, n_stocks,
                        lstm.lstm.hidden_size, device=device)
        c = torch.zeros_like(h)
        state = (h, c)
        total_loss = 0.0

        for ci in range(n_chunks):
            s, e = ci * CHUNK_SIZE, (ci + 1) * CHUNK_SIZE
            x = feat_tensor[:, s:e, :]   # [n_stocks, chunk, n_feat]
            y = lab_tensor[:, s:e]        # [n_stocks, chunk]

            pred, state = lstm(x, state)
            state = (state[0].detach(), state[1].detach())
            pred = pred.squeeze(-1)       # [n_stocks, chunk]

            losses = []
            for si in range(n_stocks):
                losses.append(ic_loss(pred[si], y[si]))
            loss = torch.stack(losses).mean()

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                lstm.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / max(n_chunks, 1)
        if epoch == 0 or (epoch + 1) % 5 == 0:
            print(f"    LSTM epoch {epoch}: loss={avg_loss:.6f}")

    return lstm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--token",    default=None)
    parser.add_argument("--hf_token", default=None)
    parser.add_argument("--device",   default="cuda")
    parser.add_argument("--epochs",   type=int, default=EPOCHS)
    args = parser.parse_args()
    device = args.device

    import os
    upstox_token = args.token or os.environ.get('UPSTOX_TOKEN')

    if args.hf_token:
        try:
            from huggingface_hub import login
            login(token=args.hf_token)
        except Exception:
            pass

    # ── Data (same as Phase 2) ────────────────────────────────────
    print("[1/4] Loading data...")
    dataset = prepare_dataset(INSTRUMENTS, token=upstox_token)
    feat_tensor, lab_tensor, val_data, test_data, common_features = \
        build_training_batches(dataset)
    n_stocks, T, n_features = feat_tensor.shape
    print(f"  {n_stocks} stocks, {T} bars, {n_features} features")

    # ── HOPE: load from checkpoint, no retraining ─────────────────
    print("\n[2/4] Loading HOPE from checkpoint...")
    hope_config = make_hope_config(n_features)
    hope = MiniHOPE(hope_config).to(device)
    optimizer_dummy = torch.optim.AdamW(hope.parameters(), lr=LR)
    hope, _, _, start_epoch, _ = load_checkpoint(
        hope, optimizer_dummy, device=device)
    print(f"  Loaded HOPE from epoch {start_epoch - 1}")

    # ── LSTM: train from scratch ───────────────────────────────────
    print("\n[3/4] Training LSTM from scratch...")
    lstm = LSTMBaseline(
        n_features=n_features, hidden_size=24,
        n_layers=2, n_outputs=1,
    ).to(device)
    lstm_params = sum(p.numel() for p in lstm.parameters())
    print(f"  LSTM params: {lstm_params:,}")
    lstm = train_lstm(lstm, feat_tensor, lab_tensor,
                      device, epochs=args.epochs)

    # ── Evaluate both ─────────────────────────────────────────────
    print("\n[4/4] Comparing HOPE vs LSTM...")
    hope_val = walk_forward_evaluation(
        hope, val_data, common_features, CHUNK_SIZE, device)
    lstm_val = walk_forward_evaluation(
        lstm, val_data, common_features, CHUNK_SIZE, device,
        model_type="lstm")

    hope_test = walk_forward_evaluation(
        hope, test_data, common_features, CHUNK_SIZE, device)
    lstm_test = walk_forward_evaluation(
        lstm, test_data, common_features, CHUNK_SIZE, device,
        model_type="lstm")

    # Print comparison table
    print(f"\n{'='*70}")
    print(f"  HOPE vs LSTM — FINAL COMPARISON")
    print(f"{'='*70}")
    print(f"  {'Stock':12s}  {'HOPE val':>9} {'LSTM val':>9}  "
          f"{'HOPE test':>10} {'LSTM test':>10}")
    print(f"  {'-'*60}")

    hope_test_ics, lstm_test_ics = [], []
    for sym in sorted(hope_test.keys()):
        hv = hope_val[sym]['mean_IC']
        lv = lstm_val.get(sym, {}).get('mean_IC', 0)
        ht = hope_test[sym]['mean_IC']
        lt = lstm_test.get(sym, {}).get('mean_IC', 0)
        hope_test_ics.append(ht)
        lstm_test_ics.append(lt)
        winner = "HOPE✅" if ht > lt else "LSTM✅"
        print(f"  {sym:12s}  {hv:+9.4f} {lv:+9.4f}  "
              f"{ht:+10.4f} {lt:+10.4f}  {winner}")

    hope_mean = np.mean(hope_test_ics)
    lstm_mean = np.mean(lstm_test_ics)
    print(f"\n  {'MEAN':12s}  {'':9s} {'':9s}  "
          f"{hope_mean:+10.4f} {lstm_mean:+10.4f}  "
          f"{'HOPE✅' if hope_mean > lstm_mean else 'LSTM✅'}")

    hope_wins = sum(h > l for h, l in
                    zip(hope_test_ics, lstm_test_ics))
    print(f"\n  HOPE beats LSTM: {hope_wins}/{len(hope_test_ics)} stocks")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
