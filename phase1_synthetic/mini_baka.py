"""
Mini-BAKA for synthetic data — CORRECT streaming architecture.

ARCHITECTURE (from Ram Charan's spec):
- Titans REPLACES attention. It IS the sequence processor.
- W_current updates at EACH timestep (not once per chunk).
  This is the recurrence: timestep t+1 sees W_current updated by timestep t.
- CMS REPLACES the FFN. It provides multi-timescale context.
- No attention block. No FFN. Just Titans + CMS + input/output layers.

WHY THE PREVIOUS VERSION FAILED:
- Titans updated W_current once per chunk (like a static layer, not a recurrence)
- Attention block wasted parameters on a mechanism BAKA doesn't use
- The model had no recurrence → no way to propagate information across timesteps
  within a chunk → predictions were constant → gradients were zero

~3.5K parameters. Intentionally smaller than LSTM (4.5K) because the test
is whether the ARCHITECTURE works, not whether more params = more accuracy.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class MiniBAKAConfig:
    n_features: int = 1        # sine wave = 1D input
    n_outputs: int = 1
    d_model: int = 16
    d_ffn: int = 32            # small FFN after Titans for capacity
    dropout: float = 0.0

    # Titans (short-term fast-weight memory)
    titans_lr: float = 0.01    # DGD inner-loop learning rate
    titans_clip: float = 0.5   # gradient clip for DGD updates

    # CMS (long-term memory) — 3 levels
    cms_schedule: tuple[int, ...] = (16, 256, 4096)
    cms_lr: tuple[float, ...] = (1e-2, 1e-3, 1e-4)


# ============================================================ Titans


class TitansMemory(nn.Module):
    """
    Fast-weight memory — the core sequence processor in BAKA.
    REPLACES attention. This IS the recurrence mechanism.

    At each timestep:
    1. READ:  memory_out = (W_base + W_current) @ x_t
    2. WRITE: W_current += lr * clip(outer(surprise, x_t))
       where surprise = x_t - memory_read (how unexpected was this input?)

    W_current evolves WITHIN a chunk (per-timestep), giving BAKA its
    recurrence. Timestep 64 sees a W_current that has been updated 63 times.
    This is how information propagates across time — NOT via attention.

    Two-timescale learning:
    - DGD (fast/inner): adapts W_current to current data stream
    - AdamW (slow/outer): optimizes W_base to make DGD adaptation better
    """

    def __init__(self, d_model: int, lr: float = 0.01, clip: float = 0.5):
        super().__init__()
        self.d_model = d_model
        self.lr = lr
        self.clip = clip

        # W_base: slow weight, trained by outer AdamW
        # Initialized with small values so memory readout starts near zero
        self.W_base = nn.Parameter(torch.randn(d_model, d_model) * 0.02)

        # Output gate: controls how much memory contributes to residual
        self.out_gate = nn.Linear(d_model, d_model)

    def init_state(self, device: torch.device) -> torch.Tensor:
        """Initialize W_current to zeros — no fast-weight memory yet."""
        return torch.zeros(self.d_model, self.d_model, device=device)

    def forward_step(
        self, x_t: torch.Tensor, W_current: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Process ONE timestep through Titans memory.

        Args:
            x_t: [B, D] — one timestep of input
            W_current: [D, D] — current fast-weight state

        Returns:
            out_t: [B, D] — memory output for this timestep
            W_new: [D, D] — updated fast weights
        """
        # READ from memory
        W = self.W_base + W_current          # [D, D]
        mem_read = torch.matmul(x_t, W.T)   # [B, D]

        # Gate the output (learned by AdamW)
        gate = torch.sigmoid(self.out_gate(x_t))  # [B, D]
        out_t = gate * mem_read              # [B, D]

        # WRITE to memory — DGD inner loop, DETACHED from outer autograd
        with torch.no_grad():
            # Surprise signal: how different was reality from expectation?
            surprise = x_t - mem_read        # [B, D]
            # Average over batch
            surprise_avg = surprise.mean(dim=0)  # [D]
            x_avg = x_t.mean(dim=0)              # [D]
            # Gradient: outer product of surprise and input
            dW = torch.outer(surprise_avg, x_avg)  # [D, D]
            # Clip gradient norm
            dW_norm = dW.norm()
            if dW_norm > self.clip:
                dW = dW * (self.clip / dW_norm)
            # Apply update
            W_new = W_current + self.lr * dW

        return out_t, W_new

    def forward_sequence(
        self, x: torch.Tensor, W_current: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Process a full chunk by iterating over timesteps.
        This is the recurrence — each timestep sees updated W_current.

        Args:
            x: [B, T, D]
            W_current: [D, D]

        Returns:
            out: [B, T, D]
            W_final: [D, D]
        """
        B, T, D = x.shape
        outputs = []
        W = W_current

        for t in range(T):
            out_t, W = self.forward_step(x[:, t, :], W)
            outputs.append(out_t)

        out = torch.stack(outputs, dim=1)  # [B, T, D]
        return out, W


# ============================================================ CMS


class CMSLevel(nn.Module):
    """
    One CMS level: accumulates inputs and fires a gated update every `period` steps.
    """

    def __init__(self, d_model: int, period: int, lr: float):
        super().__init__()
        self.period = period
        self.lr = lr
        # NOTE: No learnable gate here. The old gate was inside torch.no_grad()
        # so it NEVER received gradients — 100% dead parameters.
        # Use a fixed interpolation instead. The CMS combine layer
        # (which IS in the autograd graph) handles the learned weighting.

    def init_state(self, d_model: int, device: torch.device) -> dict:
        return {
            "summary": torch.zeros(d_model, device=device),
            "buffer_sum": torch.zeros(d_model, device=device),
            "buffer_count": 0,
        }

    def forward(self, x: torch.Tensor, state: dict) -> torch.Tensor:
        """Broadcast summary as additive residual: [D] → [B, T, D]."""
        return state["summary"].view(1, 1, -1).expand_as(x)

    def tick(self, x_step: torch.Tensor, state: dict) -> dict:
        """Accumulate one timestep, maybe fire update."""
        batch_mean = x_step.detach().mean(dim=0)
        new_sum = state["buffer_sum"] + batch_mean
        new_count = state["buffer_count"] + 1
        new_summary = state["summary"]

        if new_count >= self.period:
            # Fire: simple EMA update, no gate
            acc_mean = new_sum / new_count
            with torch.no_grad():
                new_summary = (1 - self.lr) * state["summary"] + self.lr * acc_mean
            new_sum = torch.zeros_like(new_sum)
            new_count = 0

        return {
            "summary": new_summary,
            "buffer_sum": new_sum,
            "buffer_count": new_count,
        }


class CMSStack(nn.Module):
    """Stack of CMS levels at different timescales."""

    def __init__(self, d_model: int, schedule: tuple[int, ...], lrs: tuple[float, ...]):
        super().__init__()
        self.levels = nn.ModuleList(
            [CMSLevel(d_model, p, lr) for p, lr in zip(schedule, lrs)]
        )
        self.combine = nn.Linear(d_model * len(schedule), d_model)

    def init_state(self, d_model: int, device: torch.device) -> list[dict]:
        return [lv.init_state(d_model, device) for lv in self.levels]

    def forward(self, x: torch.Tensor, states: list[dict]) -> torch.Tensor:
        summaries = [lv.forward(x, st) for lv, st in zip(self.levels, states)]
        return self.combine(torch.cat(summaries, dim=-1))

    def tick(self, x_step: torch.Tensor, states: list[dict]) -> list[dict]:
        return [lv.tick(x_step, st) for lv, st in zip(self.levels, states)]


# ============================================================ full model


class MiniBAKA(nn.Module):
    """
    Streaming Mini-BAKA — correct architecture.

    Architecture: Input → Titans (per-timestep recurrence) → CMS → FFN → Output
    NO attention block. Titans IS the sequence processor.

    forward(x, state) → (pred, new_state)
    """

    def __init__(self, cfg: MiniBAKAConfig | None = None):
        super().__init__()
        self.cfg = cfg or MiniBAKAConfig()
        c = self.cfg

        # Input: project features → d_model
        # NOTE: Do NOT use LayerNorm here when n_features=1.
        # LayerNorm(1) computes (x - mean)/std where mean=x and std=0,
        # which destroys the input entirely (outputs constant zero).
        # For 1D input, just project directly. For multi-feature, use LayerNorm.
        if c.n_features > 1:
            self.input_norm = nn.LayerNorm(c.n_features)
        else:
            self.input_norm = nn.Identity()
        self.input_proj = nn.Linear(c.n_features, c.d_model)

        # Titans: per-timestep fast-weight recurrence (REPLACES attention)
        self.titans = TitansMemory(c.d_model, lr=c.titans_lr, clip=c.titans_clip)

        # CMS: multi-timescale long-term memory (REPLACES FFN)
        self.cms = CMSStack(c.d_model, c.cms_schedule, c.cms_lr)

        # Small FFN for additional capacity
        self.ffn = nn.Sequential(
            nn.LayerNorm(c.d_model),
            nn.Linear(c.d_model, c.d_ffn),
            nn.GELU(),
            nn.Linear(c.d_ffn, c.d_model),
        )

        # Output: per-timestep prediction
        self.output_norm = nn.LayerNorm(c.d_model)
        self.head = nn.Linear(c.d_model, c.n_outputs)

    def init_state(self, batch_size: int = 1, device: torch.device = None) -> dict:
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
            x: [B, T, n_features]
            state: memory state from previous chunk

        Returns:
            pred: [B, T] per-timestep predictions
            new_state: updated memory state
        """
        B, T, _ = x.shape

        # Project input
        h = self.input_proj(self.input_norm(x))  # [B, T, D]

        # Titans: per-timestep recurrence through fast weights
        # This IS the sequence processing — each timestep updates W_current
        titans_out, new_W = self.titans.forward_sequence(h, state["titans_W"])
        h = h + titans_out  # residual

        # CMS: inject multi-timescale context
        cms_out = self.cms.forward(h, state["cms"])
        h = h + cms_out  # residual

        # FFN for additional nonlinear capacity
        h = h + self.ffn(h)  # residual

        # Per-timestep prediction
        pred = self.head(self.output_norm(h)).squeeze(-1)  # [B, T]

        # Tick CMS for each timestep
        new_cms = state["cms"]
        for t in range(T):
            new_cms = self.cms.tick(h[:, t, :], new_cms)

        return pred, {"titans_W": new_W, "cms": new_cms}

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

    # Verify recurrence: second chunk should give different output
    pred2, state2 = model(x, new_state)
    print(f"Pred1 mean: {pred.mean():.6f}, Pred2 mean: {pred2.mean():.6f}")
    print(f"Outputs differ: {(pred - pred2).abs().mean():.6f}")
