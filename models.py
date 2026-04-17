"""
Mini-BAKA (finance variant) and LSTM baseline.

Mini-BAKA keeps the architectural *spirit* of the full BAKA language model —
short-term Titans memory + multi-level CMS — but at a size appropriate for
financial data:

    - d_model = 32, 2 layers, 4 heads  -> ~30K params at the default size
    - Titans W_current: small MLP updated by inner-loop DGD every `titans_chunk`
    - CMS with 4 levels at update intervals [16, 256, 4096, 65536]

The goal at Stage 1 is to *exercise* the full pipeline end-to-end with a
model small enough to train on a Colab T4. The actual ablation comparison
vs. LSTM happens at Stage 2.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================ config


@dataclass
class BAKAFinanceConfig:
    n_features: int = 11
    n_outputs: int = 1           # 1 = predict forward return; 3 = {long, short, conf}
    d_model: int = 32
    n_layers: int = 2
    n_heads: int = 4
    d_ffn: int = 128
    dropout: float = 0.1
    # Titans (short-term memory)
    d_memory: int = 16
    titans_chunk: int = 16
    titans_lr: float = 1e-2
    # CMS (long-term memory)
    cms_levels: int = 4
    cms_schedule: tuple[int, ...] = (16, 256, 4096, 65536)
    cms_lr: tuple[float, ...] = (1e-3, 1e-4, 1e-5, 1e-6)

    def __post_init__(self):
        assert self.d_model % self.n_heads == 0
        assert len(self.cms_schedule) == self.cms_levels
        assert len(self.cms_lr) == self.cms_levels


# ============================================================ input / output


class FinanceInputLayer(nn.Module):
    """Replaces token embedding. Projects financial features into d_model."""

    def __init__(self, n_features: int, d_model: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(n_features, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # [B, T, F] -> [B, T, D]
        return self.proj(x)


class FinanceOutputHead(nn.Module):
    """Replaces LM head. Returns a signal per sequence (last-step readout)."""

    def __init__(self, d_model: int, n_outputs: int, dropout: float = 0.1):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, n_outputs),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # [B, T, D] -> [B, n_out]
        return self.head(x[:, -1, :])


# ============================================================ Titans


class TitansMemory(nn.Module):
    """
    Tiny short-term memory: a 2-layer MLP whose weights are updated by an
    inner-loop deep gradient descent (DGD) step every `chunk` tokens.

    Interface:
        y = memory(x)                           # normal forward
        loss.backward() over chunk tokens       # outer loop (via optimizer)
        memory.dgd_update(x_chunk, target)      # inner loop (fast weights)
    """

    def __init__(self, d_model: int, d_memory: int, lr: float = 1e-2):
        super().__init__()
        self.d_model = d_model
        self.d_memory = d_memory
        self.lr = lr
        # Slow (trained) weights
        self.in_proj = nn.Linear(d_model, d_memory)
        self.out_proj = nn.Linear(d_memory, d_model)
        # Fast weights: a residual tensor added at read-time, reset per session.
        self.register_buffer(
            "fast_bias", torch.zeros(d_model), persistent=False
        )

    def reset(self) -> None:
        self.fast_bias.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = torch.tanh(self.in_proj(x))
        y = self.out_proj(h) + self.fast_bias
        return y

    @torch.no_grad()
    def dgd_update(self, x_chunk: torch.Tensor, target: torch.Tensor) -> None:
        """
        One gradient step of the fast_bias toward minimizing the chunk-local
        reconstruction error. Acts as a persistent "surprise" accumulator.
        """
        pred = self.forward(x_chunk.mean(dim=1, keepdim=False))  # [B, D]
        err = (target - pred).mean(dim=0)                        # [D]
        self.fast_bias += self.lr * err


# ============================================================ CMS


class CMSLevel(nn.Module):
    """
    One CMS level = running summary updated every `period` steps.
    We use an EMA-style accumulator with its own learnable mixing gate.
    """

    def __init__(self, d_model: int, period: int, lr: float):
        super().__init__()
        self.period = period
        self.lr = lr
        self.gate = nn.Linear(d_model, d_model)
        self.register_buffer(
            "summary", torch.zeros(d_model), persistent=False
        )
        self.register_buffer(
            "step", torch.zeros((), dtype=torch.long), persistent=False
        )

    def reset(self) -> None:
        self.summary.zero_()
        self.step.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Broadcast the current summary back into the sequence."""
        B, T, D = x.shape
        return self.summary.view(1, 1, D).expand(B, T, D)

    @torch.no_grad()
    def maybe_update(self, x_step: torch.Tensor) -> None:
        """x_step: [B, D] — one timestep aggregated over the batch."""
        self.step += 1
        if int(self.step.item()) % self.period == 0:
            # Sigmoid-gated EMA into the summary.
            batch_mean = x_step.mean(dim=0)
            gate = torch.sigmoid(self.gate(batch_mean))
            self.summary.mul_(1 - self.lr).add_(self.lr * gate * batch_mean)


class CMSStack(nn.Module):
    def __init__(self, d_model: int, schedule: tuple[int, ...], lrs: tuple[float, ...]):
        super().__init__()
        self.levels = nn.ModuleList(
            [CMSLevel(d_model, period=p, lr=lr) for p, lr in zip(schedule, lrs)]
        )
        # Combine all levels back into a single d_model residual.
        self.combine = nn.Linear(d_model * len(schedule), d_model)

    def reset(self) -> None:
        for lv in self.levels:
            lv.reset()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        summaries = [lv(x) for lv in self.levels]
        stacked = torch.cat(summaries, dim=-1)  # [B, T, D*L]
        return self.combine(stacked)

    @torch.no_grad()
    def tick(self, x_step: torch.Tensor) -> None:
        for lv in self.levels:
            lv.maybe_update(x_step)


# ============================================================ transformer block


class BAKABlock(nn.Module):
    def __init__(self, cfg: BAKAFinanceConfig):
        super().__init__()
        self.norm1 = nn.LayerNorm(cfg.d_model)
        self.attn = nn.MultiheadAttention(
            cfg.d_model, cfg.n_heads, dropout=cfg.dropout, batch_first=True
        )
        self.norm2 = nn.LayerNorm(cfg.d_model)
        self.ffn = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_ffn),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.d_ffn, cfg.d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Causal mask: prevent any step from attending to the future.
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


class MiniBAKAFinance(nn.Module):
    def __init__(self, cfg: BAKAFinanceConfig | None = None):
        super().__init__()
        self.cfg = cfg or BAKAFinanceConfig()
        c = self.cfg

        self.input = FinanceInputLayer(c.n_features, c.d_model)
        self.titans = TitansMemory(c.d_model, c.d_memory, lr=c.titans_lr)
        self.cms = CMSStack(c.d_model, c.cms_schedule, c.cms_lr)
        self.blocks = nn.ModuleList([BAKABlock(c) for _ in range(c.n_layers)])
        self.final_norm = nn.LayerNorm(c.d_model)
        self.head = FinanceOutputHead(c.d_model, c.n_outputs, c.dropout)

    # --- memory control --------------------------------------------------

    def reset_memory(self) -> None:
        """Call at the start of each trading session."""
        self.titans.reset()
        self.cms.reset()

    # --- forward ---------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, n_features]
        h = self.input(x)

        # Inject memory summaries as additive residuals.
        h = h + self.titans(h)
        h = h + self.cms(h)

        for blk in self.blocks:
            h = blk(h)
        h = self.final_norm(h)
        out = self.head(h)  # [B, n_outputs]

        # Inner-loop updates after the forward pass (train + eval both).
        with torch.no_grad():
            # Use the chunk-averaged representation as a self-supervised target.
            chunk = h[:, -self.cfg.titans_chunk :, :]
            self.titans.dgd_update(chunk, chunk.mean(dim=1))
            self.cms.tick(h[:, -1, :])

        return out

    # --- ablation knobs --------------------------------------------------

    def ablate_cms_level(self, level_idx: int) -> None:
        """Zero out one CMS level's contribution (for ablations)."""
        with torch.no_grad():
            self.cms.levels[level_idx].summary.zero_()
            # Freeze it so tick() can't revive it mid-run.
            for p in self.cms.levels[level_idx].parameters():
                p.requires_grad_(False)

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ============================================================ LSTM baseline


class LSTMBaseline(nn.Module):
    def __init__(
        self,
        n_features: int,
        hidden_dim: int = 64,
        n_layers: int = 2,
        n_outputs: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )
        self.head = nn.Linear(hidden_dim, n_outputs)
        self._state: tuple[torch.Tensor, torch.Tensor] | None = None

    def reset_memory(self) -> None:
        self._state = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Detach the persistent state to prevent backprop through history.
        state = self._state
        if state is not None:
            state = (state[0].detach(), state[1].detach())
            # Only reuse if batch size matches
            if state[0].size(1) != x.size(0):
                state = None
        out, new_state = self.lstm(x, state)
        self._state = new_state
        return self.head(out[:, -1, :])

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ============================================================ smoke test


if __name__ == "__main__":
    cfg = BAKAFinanceConfig(n_features=11, n_outputs=1)
    baka = MiniBAKAFinance(cfg)
    lstm = LSTMBaseline(n_features=11, hidden_dim=64, n_outputs=1)

    print(f"Mini-BAKA  params: {baka.param_count():,}")
    print(f"LSTM       params: {lstm.param_count():,}")

    x = torch.randn(4, 32, 11)  # [batch, seq, features]
    y_baka = baka(x)
    y_lstm = lstm(x)
    print(f"Mini-BAKA out: {y_baka.shape}")
    print(f"LSTM      out: {y_lstm.shape}")
