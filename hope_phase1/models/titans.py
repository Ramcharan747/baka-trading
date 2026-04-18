"""
Self-Referential Titans — the core sequence processor in HOPE.

Implements Equations 86-93 from the Nested Learning paper.

Key insight: EVERY component of Titans is an associative memory that
generates its OWN training targets by passing v through itself.
This is what makes it "self-referential" (Schmidhuber 1993).

Six memory modules, all updated via DGD:
  M_k:      key projection      (adaptive, updated in-context)
  M_v:      value projection     (adaptive, updated in-context)
  M_η:      learning rate        (adaptive, predicts per-token η)
  M_α:      weight decay         (adaptive, predicts per-token α)
  M_mem:    main memory          (fast-weight matrix, updated via DGD)
  W_q:      query projection     (STATIC — paper ablation shows minimal impact)

DGD update (Eq 93 with L2 regression loss):
  error = M_□·k_t - v̂_□,t
  M_□,t = M_□,t-1 · (α_t·I - η_t·k_t·k_t^T) - η_t · error · k_t^T

Self-referential value generation (Eq 87):
  v̂_□,t = M_□,t-1(v_t)

Each memory generates its own target by passing v_t through itself.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .memory import MemoryModule


class SelfReferentialTitans(nn.Module):
    """
    Implements the full self-referential Titans block (Eqs 86-93).

    Architecture:
      Input x_t → adaptive projections via M_k, M_v, M_η, M_α
                → static query via W_q
                → retrieve from M_mem via q_t
                → generate self-referential targets v̂_□ = M_□(v_t)
                → DGD update of all memory matrices (Eq 93)
                → gated output

    The memory matrices are NOT nn.Parameters — they live in the state dict,
    updated by DGD inside torch.no_grad(), DETACHED from outer autograd.
    The MLP networks (M_k_net etc.) ARE nn.Parameters, trained by AdamW.

    This separation is critical:
      - Outer loop (AdamW): learns the MLP projections
      - Inner loop (DGD): learns the fast-weight matrices
    """

    def __init__(self, d_model: int, inner_lr: float = 0.01,
                 inner_decay: float = 0.99, grad_clip: float = 1.0):
        super().__init__()
        self.d_model = d_model
        self.inner_lr_init = inner_lr
        self.inner_decay_init = inner_decay
        self.grad_clip = grad_clip

        # Static query projection (paper shows this barely matters — Table 6)
        self.W_q = nn.Linear(d_model, d_model, bias=False)

        # Adaptive memory modules (Eq 89: M(x) = x + W1 σ(W2 x))
        # These are the MLP networks whose WEIGHTS are trained by AdamW.
        # Their OUTPUTS (k_t, v_t, η_t, α_t) are used in the inner loop.
        self.M_k_net = MemoryModule(d_model)          # key projection
        self.M_v_net = MemoryModule(d_model)          # value projection
        self.M_eta_net = MemoryModule(d_model, out_dim=1)  # scalar LR per token
        self.M_alpha_net = MemoryModule(d_model, out_dim=1)  # scalar decay per token

        # W_base: the meta-learned initial state of M_mem.
        # This IS an nn.Parameter — it's optimized by AdamW.
        # At the start of each sequence, M_mem is initialized to W_base.
        # During the sequence, DGD updates M_mem away from W_base.
        self.W_base = nn.Parameter(torch.eye(d_model) * 0.1)

        # Output gate and projection
        self.out_gate = nn.Sequential(nn.Linear(d_model, d_model), nn.SiLU())
        self.out_proj = nn.Linear(d_model, d_model)

    def init_state(self, batch_size: int, device: torch.device) -> dict:
        """
        Initialize all fast-weight memory matrices.

        M_k, M_v, M_mem all start from W_base (meta-learned).
        These are cloned per batch element because DGD operates independently.
        """
        return {
            "M_k": [self.W_base.data.clone() for _ in range(batch_size)],
            "M_v": [self.W_base.data.clone() for _ in range(batch_size)],
            "M_mem": [self.W_base.data.clone() for _ in range(batch_size)],
            "step": 0,
        }

    def forward_chunk(self, x_chunk: torch.Tensor, state: dict) -> tuple:
        """
        Process one chunk of timesteps.

        Args:
            x_chunk: [batch, chunk_size, d_model]
            state:   dict with per-batch memory matrices (detached)

        Returns:
            (output, new_state)
            output: [batch, chunk_size, d_model]
        """
        B, C, D = x_chunk.shape
        outputs = []

        for t in range(C):
            x_t = x_chunk[:, t, :]  # [B, D]

            # ─── Step 1: Adaptive projections (Eq 86) ───
            # These MLP networks ARE in the autograd graph.
            # Their weights are trained by AdamW.
            k_t = self.M_k_net(x_t)       # [B, D]
            v_t = self.M_v_net(x_t)       # [B, D]
            eta_raw = self.M_eta_net(x_t)  # [B, 1]
            alpha_raw = self.M_alpha_net(x_t)  # [B, 1]

            # Scale η and α to valid ranges
            eta_t = self.inner_lr_init * torch.sigmoid(eta_raw)   # [B, 1], positive
            alpha_t = torch.sigmoid(alpha_raw)                     # [B, 1], in (0,1)

            # Static query projection (only q is non-adaptive)
            q_t = self.W_q(x_t)  # [B, D]

            # L2 normalize keys and queries (paper Section 8.3)
            k_t = F.normalize(k_t, dim=-1)
            q_t = F.normalize(q_t, dim=-1)

            # ─── Step 2: Retrieve from main memory (Eq 94) ───
            # M_mem is the fast-weight matrix. Retrieval = matrix-vector multiply.
            o_t = (state["M_mem"][0] @ q_t[0]).unsqueeze(0)  # [1, D]

            # ─── Step 3: Self-referential value generation (Eq 87/95) ───
            # v̂_□,t = M_□,t-1(v_t) for □ ∈ {k, v, mem}
            # Each memory generates its OWN training target.
            v_hat_k = self.M_k_net(v_t)    # [1, D]
            v_hat_v = self.M_v_net(v_t)    # [1, D]
            v_hat_mem = (state["M_mem"][0] @ v_t[0]).unsqueeze(0)  # [1, D]

            # ─── Step 4: DGD update (Eq 93) — DETACHED from outer autograd ───
            # M_□,t = M_□,t-1 · (α_t·I - η_t·k_t·k_t^T) - η_t · (M_□·k - v̂_□) · k^T
            #
            # This update is INSIDE torch.no_grad():
            #   - The fast-weight matrices are NOT nn.Parameters
            #   - They don't participate in the outer AdamW gradient
            #   - They are the "inner loop" of the nested learning system
            with torch.no_grad():
                eta_b = eta_t[0].item()
                alpha_b = alpha_t[0].item()
                k_b = k_t[0].detach()       # [D]
                v_hat_k_b = v_hat_k[0].detach()
                v_hat_v_b = v_hat_v[0].detach()
                v_hat_mem_b = v_hat_mem[0].detach()

                # Precompute k·k^T (rank-1 outer product) [D, D]
                kk_t = torch.outer(k_b, k_b)

                # DGD decay matrix: (α·I - η·k·k^T)
                decay = alpha_b * torch.eye(D, device=x_t.device) - eta_b * kk_t

                # ── Update M_k ──
                error_k = state["M_k"][0] @ k_b - v_hat_k_b
                if error_k.norm() > self.grad_clip:
                    error_k = error_k * (self.grad_clip / error_k.norm())
                state["M_k"][0] = state["M_k"][0] @ decay - eta_b * torch.outer(error_k, k_b)

                # ── Update M_v ──
                error_v = state["M_v"][0] @ k_b - v_hat_v_b
                if error_v.norm() > self.grad_clip:
                    error_v = error_v * (self.grad_clip / error_v.norm())
                state["M_v"][0] = state["M_v"][0] @ decay - eta_b * torch.outer(error_v, k_b)

                # ── Update M_mem ──
                error_mem = state["M_mem"][0] @ k_b - v_hat_mem_b
                if error_mem.norm() > self.grad_clip:
                    error_mem = error_mem * (self.grad_clip / error_mem.norm())
                state["M_mem"][0] = state["M_mem"][0] @ decay - eta_b * torch.outer(error_mem, k_b)

            # ─── Step 5: Gated output ───
            combined = o_t + v_t          # o_t has grad through q_t; v_t has grad through M_v_net
            gate = self.out_gate(x_t)
            out_t = self.out_proj(combined * gate)
            outputs.append(out_t)

        output = torch.stack(outputs, dim=1)  # [B, C, D]
        state["step"] += C
        return output, state
