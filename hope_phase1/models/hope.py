"""
HOPE: Hierarchical Online Persistent Encoding.

Full model combining Self-Referential Titans + Continuum Memory System.

Architecture (Figure 5 from paper):
  Input x_t
    → Input projection (Linear, NO LayerNorm for 1D input)
    → n_layers × HOPEBlock:
        → Self-Referential Titans (high-frequency, adapts every chunk)
        → CMS Level 0 (medium frequency, e.g. every 16 steps)
        → CMS Level 1 (lower frequency, e.g. every 64 steps)
        → CMS Level 2 (lowest frequency, e.g. every 256 steps)
    → Output head (LayerNorm → Linear)
    → prediction

Eq 97: y_t = MLP^(f_K)( MLP^(f_{K-1})( ... MLP^(f_1)(o_t) ... ))
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn

from .titans import SelfReferentialTitans
from .cms import CMSBlock


@dataclass
class HopeConfig:
    """Configuration for MiniHOPE model."""
    # Model dimensions
    n_features: int = 1         # input features (1 for sine, 10 for finance)
    d_model: int = 24           # must match LSTM for fair comparison (~5K params)
    n_layers: int = 1           # number of HOPE blocks (keep 1 for Phase 1)
    n_outputs: int = 1          # prediction target dimension

    # Titans (inner memory, fast update)
    inner_lr: float = 0.01      # η_t initialization (overridden adaptively per token)
    inner_decay: float = 0.99   # α_t initialization (overridden adaptively per token)
    grad_clip: float = 1.0      # gradient clipping inside DGD

    # CMS (outer memory, slow update)
    cms_levels: int = 3
    cms_schedule: list = field(default_factory=lambda: [16, 64, 256])
    cms_lr: list = field(default_factory=lambda: [1e-3, 1e-4, 1e-5])

    # Training
    chunk_size: int = 16        # timesteps per DGD update
    dropout: float = 0.0        # keep 0 for small models


class HOPEBlock(nn.Module):
    """
    One HOPE block = Self-Referential Titans + sequential CMS.

    Architecture (Figure 5):
      x → Titans → LayerNorm → CMS_0 → CMS_1 → CMS_2 → y
    """

    def __init__(self, config: HopeConfig):
        super().__init__()
        d = config.d_model

        self.titans = SelfReferentialTitans(
            d_model=d,
            inner_lr=config.inner_lr,
            inner_decay=config.inner_decay,
            grad_clip=config.grad_clip,
        )

        self.norm = nn.LayerNorm(d)

        self.cms_levels = nn.ModuleList([
            CMSBlock(d, config.cms_schedule[i], config.cms_lr[i])
            for i in range(config.cms_levels)
        ])

    def forward(self, x: torch.Tensor, state: dict,
                step: int) -> tuple[torch.Tensor, dict]:
        """
        x:     [batch, seq_len, d_model]
        state: dict with Titans memory state
        step:  global step counter
        """
        # Titans: self-referential fast update
        o, new_state = self.titans.forward_chunk(x, state)

        # CMS: multi-timescale slow update (sequential, Eq 97)
        h = self.norm(o)
        for cms in self.cms_levels:
            h = cms(h)

        return h, new_state

    def post_backward(self, global_step: int):
        """
        Called after loss.backward() — handles CMS gradient accumulation.

        This is the CMS update mechanism:
        1. Capture CMS gradients from the autograd graph
        2. Accumulate them in the CMS buffer
        3. Apply when the schedule fires
        """
        for cms in self.cms_levels:
            cms.accumulate_gradients()
            cms.maybe_update(global_step)

    def reset_cms_buffers(self):
        """Reset CMS gradient buffers (call at start of each epoch)."""
        for cms in self.cms_levels:
            cms.reset_buffer()


class MiniHOPE(nn.Module):
    """
    Complete Mini-HOPE for streaming sequential prediction.

    n_features → input_proj → n_layers × HOPEBlock → output head

    For Phase 1 (synthetic sine): n_features=1, n_outputs=1
    For Phase 2 (financial):      n_features=10, n_outputs=1
    """

    def __init__(self, config: HopeConfig):
        super().__init__()
        self.config = config
        d = config.d_model

        # Input projection (replaces token embedding)
        # CRITICAL: NO LayerNorm for n_features=1
        # LayerNorm(1) computes (x - mean)/std where mean=x, std=0 → kills input
        if config.n_features == 1:
            self.input_proj = nn.Linear(config.n_features, d)
        else:
            self.input_proj = nn.Sequential(
                nn.LayerNorm(config.n_features),
                nn.Linear(config.n_features, d),
                nn.SiLU(),
            )

        # HOPE layers
        self.layers = nn.ModuleList([
            HOPEBlock(config) for _ in range(config.n_layers)
        ])

        # Output head
        self.output_norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, config.n_outputs)

        # Initialize head to near-zero (prevent early overconfident predictions)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def init_state(self, batch_size: int, device: torch.device) -> list:
        """Initialize Titans state for all layers."""
        return [
            layer.titans.init_state(batch_size, device)
            for layer in self.layers
        ]

    def forward(self, x: torch.Tensor, states: list,
                step: int = 0) -> tuple[torch.Tensor, list]:
        """
        Streaming forward pass.

        Args:
            x:      [batch, seq_len, n_features]
            states: list of per-layer Titans states
            step:   global step counter (for CMS scheduling)

        Returns:
            (predictions, new_states)
            predictions: [batch, seq_len, n_outputs]
        """
        # Input projection
        h = self.input_proj(x)  # [batch, seq_len, d_model]

        # HOPE layers (Titans + CMS)
        new_states = []
        for i, layer in enumerate(self.layers):
            h, new_state = layer(h, states[i], step)
            new_states.append(new_state)

        # Output head
        h = self.output_norm(h)
        pred = self.head(h)  # [batch, seq_len, n_outputs]

        return pred, new_states

    def post_backward(self, global_step: int):
        """Called after loss.backward() — handles CMS gradient accumulation."""
        for layer in self.layers:
            layer.post_backward(global_step)

    def reset_cms_buffers(self):
        """Reset CMS buffers at start of epoch."""
        for layer in self.layers:
            layer.reset_cms_buffers()
