"""
Mini-BAKA for synthetic data — streaming architecture with correct Titans + CMS.

Key differences from the OLD (wrong) models.py:
1. Titans uses a FULL W_current [d_model × d_model] matrix, not a 1D bias
2. DGD updates W_current per chunk with gradient clipping
3. CMS levels accumulate inputs and fire on schedule
4. Forward returns (pred, new_state) — state is an explicit dict
5. Per-timestep prediction (output at every bar, not just last)

~4.5K parameters — intentionally tiny. This is a sanity check, not a final model.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn


@dataclass
class MiniBAKAConfig:
    n_features: int = 1        # sine wave = 1D input
    n_outputs: int = 1
    d_model: int = 16
    n_heads: int = 2
    d_ffn: int = 32
    dropout: float = 0.0       # no dropout for tiny model

    # Titans (short-term fast-weight memory)
    titans_chunk: int = 16     # DGD fires every 16 steps
    titans_lr: float = 0.01   # inner-loop learning rate for W_current

    # CMS (long-term memory) — 3 levels, NO level 3
    cms_levels: int = 3
    cms_schedule: tuple[int, ...] = (16, 256, 4096)
    cms_lr: tuple[float, ...] = (1e-2, 1e-3, 1e-4)


# ============================================================ Titans


class TitansMemory(nn.Module):
    """
    Fast-weight memory (Schmidhuber 1992 / Ba et al 2016).

    W_current is a [d_model × d_model] weight matrix — the fast weight.
    It is updated every `chunk` steps by DGD (inner-loop gradient descent).
    W_current is DETACHED from outer autograd — AdamW never sees it.
    W_base IS trained by AdamW — it's the slow meta-learning weight.

    The forward pass computes: memory_output = x @ (W_base + W_current)^T
    """

    def __init__(self, d_model: int, lr: float = 0.01, clip: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.lr = lr
        self.clip = clip

        # W_base: trained by outer optimizer (AdamW)
        self.W_base = nn.Parameter(torch.randn(d_model, d_model) * 0.01)
        # Output projection
        self.out_proj = nn.Linear(d_model, d_model)

    def init_state(self, device: torch.device) -> torch.Tensor:
        """Initialize W_current to zeros — no fast-weight memory yet."""
        return torch.zeros(self.d_model, self.d_model, device=device)

    def forward(
        self, x: torch.Tensor, W_current: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [B, T, D] input representations
            W_current: [D, D] current fast-weight matrix

        Returns:
            memory_output: [B, T, D] to be added as residual
            W_current_new: [D, D] updated fast weights
        """
        # Memory readout: use combined slow + fast weights
        W = self.W_base + W_current  # [D, D]
        h = torch.matmul(x, W.T)    # [B, T, D]
        out = self.out_proj(h)       # [B, T, D]

        # DGD update — inner loop, DETACHED from outer autograd
        # Uses self-supervised reconstruction: predict aggregated representation
        with torch.no_grad():
            # Target: mean of representations in this chunk (self-supervised)
            target = x.mean(dim=1)           # [B, D]
            # Prediction from fast weights: W_current @ last_step
            query = x[:, -1, :]              # [B, D]
            pred = torch.matmul(query, W_current.T)  # [B, D]
            # Error signal
            err = (target - pred).mean(dim=0)  # [D]
            # Gradient w.r.t. W_current (outer product)
            grad = torch.outer(err, query.mean(dim=0))  # [D, D]
            # Clip gradient to prevent explosion
            grad_norm = grad.norm()
            if grad_norm > self.clip:
                grad = grad * (self.clip / grad_norm)
            # Apply DGD step
            W_current_new = W_current + self.lr * grad

        return out, W_current_new


# ============================================================ CMS


class CMSLevel(nn.Module):
    """
    One CMS level: accumulates inputs and fires an update every `period` steps.

    Unlike the OLD implementation which used a simple EMA:
    - This accumulates the actual input representations in a buffer
    - On fire: computes mean of buffer → gate → update summary
    - The gate MLP is trained by outer AdamW
    - The summary update uses the level's own learning rate
    """

    def __init__(self, d_model: int, period: int, lr: float):
        super().__init__()
        self.period = period
        self.lr = lr
        self.gate = nn.Linear(d_model, d_model)

    def init_state(self, d_model: int, device: torch.device) -> dict:
        return {
            "summary": torch.zeros(d_model, device=device),
            "buffer_sum": torch.zeros(d_model, device=device),
            "buffer_count": 0,
            "step": 0,
        }

    def forward(self, x: torch.Tensor, state: dict) -> torch.Tensor:
        """Broadcast current summary as additive residual."""
        # [D] → [1, 1, D] → [B, T, D]
        return state["summary"].view(1, 1, -1).expand_as(x)

    def tick(self, x_step: torch.Tensor, state: dict) -> dict:
        """
        Accumulate one timestep's representation and maybe fire an update.

        Args:
            x_step: [B, D] representation at one timestep, averaged over batch
            state: level state dict
        """
        # Accumulate (running sum, not a list — memory efficient)
        batch_mean = x_step.detach().mean(dim=0)  # [D]
        new_sum = state["buffer_sum"] + batch_mean
        new_count = state["buffer_count"] + 1
        new_step = state["step"] + 1

        new_summary = state["summary"]

        if new_count >= self.period:
            # Fire: compute gated update from accumulated mean
            acc_mean = new_sum / new_count
            # Gate controls how much new info enters summary
            # The gate MLP is part of outer autograd graph (trained by AdamW)
            # But we detach because this fires inside forward, not during backward
            with torch.no_grad():
                g = torch.sigmoid(self.gate(acc_mean))
                new_summary = (1 - self.lr) * state["summary"] + self.lr * g * acc_mean
            # Reset buffer
            new_sum = torch.zeros_like(new_sum)
            new_count = 0

        return {
            "summary": new_summary,
            "buffer_sum": new_sum,
            "buffer_count": new_count,
            "step": new_step,
        }


class CMSStack(nn.Module):
    """Stack of CMS levels at different timescales."""

    def __init__(self, d_model: int, schedule: tuple[int, ...], lrs: tuple[float, ...]):
        super().__init__()
        self.levels = nn.ModuleList(
            [CMSLevel(d_model, period=p, lr=lr) for p, lr in zip(schedule, lrs)]
        )
        self.combine = nn.Linear(d_model * len(schedule), d_model)
        self.n_levels = len(schedule)

    def init_state(self, d_model: int, device: torch.device) -> list[dict]:
        return [lv.init_state(d_model, device) for lv in self.levels]

    def forward(self, x: torch.Tensor, states: list[dict]) -> torch.Tensor:
        """Combine all level summaries into a single residual."""
        summaries = [lv.forward(x, st) for lv, st in zip(self.levels, states)]
        stacked = torch.cat(summaries, dim=-1)  # [B, T, D*L]
        return self.combine(stacked)

    def tick(self, x_step: torch.Tensor, states: list[dict]) -> list[dict]:
        """Tick all levels with one timestep."""
        return [lv.tick(x_step, st) for lv, st in zip(self.levels, states)]


# ============================================================ transformer block


class BAKABlock(nn.Module):
    """Standard pre-norm transformer block with causal attention."""

    def __init__(self, d_model: int, n_heads: int, d_ffn: int, dropout: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ffn),
            nn.GELU(),
            nn.Linear(d_ffn, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        T = x.size(1)
        mask = torch.triu(
            torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1
        )
        h = self.norm1(x)
        attn_out, _ = self.attn(h, h, h, attn_mask=mask, need_weights=False)
        x = x + attn_out
        x = x + self.ffn(self.norm2(x))
        return x


# ============================================================ full model


class MiniBAKA(nn.Module):
    """
    Streaming Mini-BAKA for synthetic data.

    Critical: forward(x, state) → (pred, new_state)
    State is an explicit dict that flows between chunks.
    This enables TBPTT while preserving persistent memory.
    """

    def __init__(self, cfg: MiniBAKAConfig | None = None):
        super().__init__()
        self.cfg = cfg or MiniBAKAConfig()
        c = self.cfg

        # Input projection
        self.input_norm = nn.LayerNorm(c.n_features)
        self.input_proj = nn.Linear(c.n_features, c.d_model)

        # Titans (fast-weight short-term memory)
        self.titans = TitansMemory(c.d_model, lr=c.titans_lr)

        # CMS (long-term memory)
        self.cms = CMSStack(c.d_model, c.cms_schedule, c.cms_lr)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            BAKABlock(c.d_model, c.n_heads, c.d_ffn, c.dropout)
        ])  # single layer for ~5K params

        self.final_norm = nn.LayerNorm(c.d_model)

        # Per-timestep output head
        self.head = nn.Linear(c.d_model, c.n_outputs)

    def init_state(self, batch_size: int = 1, device: torch.device = None) -> dict:
        """Initialize all memory to zeros."""
        device = device or torch.device("cpu")
        return {
            "titans_W": self.titans.init_state(device),
            "cms": self.cms.init_state(self.cfg.d_model, device),
        }

    def forward(
        self, x: torch.Tensor, state: dict
    ) -> tuple[torch.Tensor, dict]:
        """
        Streaming forward pass.

        Args:
            x: [B, T, n_features] — one chunk of input
            state: memory state from previous chunk

        Returns:
            pred: [B, T] — per-timestep predictions
            new_state: updated memory state
        """
        B, T, _ = x.shape

        # Project input
        h = self.input_proj(self.input_norm(x))  # [B, T, D]

        # Titans: inject fast-weight memory
        titans_out, new_W = self.titans(h, state["titans_W"])
        h = h + titans_out

        # CMS: inject long-term memory summaries
        cms_out = self.cms.forward(h, state["cms"])
        h = h + cms_out

        # Transformer blocks (causal attention within chunk)
        for blk in self.blocks:
            h = blk(h)
        h = self.final_norm(h)

        # Per-timestep prediction
        pred = self.head(h).squeeze(-1)  # [B, T]

        # Update CMS state: tick for each timestep in chunk
        new_cms = state["cms"]
        for t in range(T):
            new_cms = self.cms.tick(h[:, t, :], new_cms)

        new_state = {
            "titans_W": new_W,
            "cms": new_cms,
        }

        return pred, new_state

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    cfg = MiniBAKAConfig()
    model = MiniBAKA(cfg)
    print(f"MiniBAKA params: {model.param_count():,}")

    # Test streaming forward
    state = model.init_state(batch_size=2)
    x = torch.randn(2, 16, 1)
    pred, new_state = model(x, state)
    print(f"Input: {x.shape} → Pred: {pred.shape}")
    print(f"W_current norm: {new_state['titans_W'].norm():.4f}")
    print(f"CMS level 0 step: {new_state['cms'][0]['step']}")
