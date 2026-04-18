"""
Diagnostics for HOPE — run these FIRST to verify architecture correctness.

Three checks:
  1. Output diff (zeros vs ones): if 0, input is being ignored
  2. Gradient norms: if any are zero/NO GRADIENT, path is broken
  3. W_norm change: if M_mem norm doesn't change, DGD isn't firing
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .losses import ic_loss


def diagnose_hope(model: nn.Module, device: str = "cuda"):
    """
    Run after initialization. Identifies architecture bugs immediately.

    Checks:
    1. Does output respond to different inputs? (zeros vs ones)
    2. Do all parameters receive gradients?
    3. Does DGD actually modify M_mem?
    """
    model = model.to(device)
    chunk_size = 16

    # ─── Check 1: Output responds to input ───
    x_zeros = torch.zeros(1, chunk_size, 1, device=device)
    x_ones = torch.ones(1, chunk_size, 1, device=device)

    states_z = model.init_state(1, torch.device(device))
    states_o = model.init_state(1, torch.device(device))

    with torch.no_grad():
        out_z, _ = model(x_zeros, states_z)
        out_o, _ = model(x_ones, states_o)

    diff = (out_o - out_z).abs().mean().item()
    status = "✅ Output responds to input" if diff > 0.01 else "❌ OUTPUT IGNORES INPUT"
    print(f"  DIAGNOSTIC: Output diff (zeros vs ones): {diff:.6f} ← {status}")

    # ─── Check 2: Gradient norms ───
    x_chunk = torch.randn(1, chunk_size, 1, device=device)
    y_chunk = torch.randn(1, chunk_size, device=device)
    states = model.init_state(1, torch.device(device))

    model.train()
    pred, _ = model(x_chunk, states)
    loss = ic_loss(pred.squeeze(), y_chunk.squeeze())
    print(f"  DIAGNOSTIC: Loss on first batch: {loss.item():.6f}")

    loss.backward()

    print("  DIAGNOSTIC: Gradient norms:")
    for name, param in model.named_parameters():
        if param.grad is not None:
            norm = param.grad.norm().item()
            status = ""
            if norm < 1e-6:
                status = "← ⚠️ NEAR ZERO"
            print(f"    {name:50s}: {norm:.6f} {status}")
        else:
            print(f"    {name:50s}: NO GRADIENT ← ❌")

    model.zero_grad()

    # ─── Check 3: W_norm changes after DGD ───
    states = model.init_state(1, torch.device(device))
    initial_norms = [
        state["M_mem"][0].norm().item()
        for state in states
        if "M_mem" in state
    ]

    x_seq = torch.randn(1, 64, 1, device=device)
    with torch.no_grad():
        _, states_after = model(x_seq, states)

    final_norms = [
        state["M_mem"][0].norm().item()
        for state in states_after
        if "M_mem" in state
    ]

    for i, (init_n, final_n) in enumerate(zip(initial_norms, final_norms)):
        changed = abs(final_n - init_n) > 1e-4
        status = "✅ DGD firing" if changed else "❌ DGD NOT updating"
        print(f"  DIAGNOSTIC: Layer {i} W_norm: {init_n:.4f} → {final_n:.4f} {status}")
