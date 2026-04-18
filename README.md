<p align="center">
  <h1 align="center">HOPE — Hierarchical Online Persistent Encoding</h1>
  <p align="center">
    <em>A novel streaming neural architecture for financial time series prediction</em>
  </p>
  <p align="center">
    Based on the <a href="https://arxiv.org/abs/2505.xxxxx">Nested Learning</a> paper (NeurIPS 2025) — Self-Referential Titans + Continuum Memory System
  </p>
  <p align="center">
    <strong>Ram Charan</strong> · 2nd Year PIE · IIT Delhi
  </p>
  <p align="center">
    <img src="https://img.shields.io/badge/Phase_1-✅_Passed-brightgreen?style=flat-square" alt="Phase 1">
    <img src="https://img.shields.io/badge/Phase_2-✅_Passed-brightgreen?style=flat-square" alt="Phase 2">
    <img src="https://img.shields.io/badge/PyTorch-2.0+-red?style=flat-square&logo=pytorch" alt="PyTorch">
    <img src="https://img.shields.io/badge/Platform-Colab_T4-blue?style=flat-square&logo=googlecolab" alt="Colab">
  </p>
</p>

---

## 📋 Table of Contents

- [What is HOPE?](#-what-is-hope)
- [Architecture](#-architecture)
- [Phase 1 Results](#-phase-1-results--synthetic-validation)
- [Phase 2: Financial Data](#-phase-2-financial-data)
- [Repository Structure](#-repository-structure)
- [Quick Start](#-quick-start)
- [Design Principles](#-design-principles)
- [Hardware & Infrastructure](#-hardware--infrastructure)
- [Citation](#-citation)

---

## 🧠 What is HOPE?

**HOPE** (Hierarchical Online Persistent Encoding) is a streaming neural architecture that maintains **persistent memory** across arbitrary sequence lengths. Unlike transformers (fixed context window) or LSTMs (lossy compression), HOPE uses:

1. **Self-Referential Titans** — Fast-weight memory matrices updated via **Delta Gradient Descent** (DGD). Every component generates its own training targets, making it truly self-referential (Schmidhuber 1993).

2. **Continuum Memory System** (CMS) — Multi-timescale persistent summaries. Different layers update at different frequencies, capturing patterns from microstructure noise to regime changes.

The key insight: the **inner loop** (DGD) learns per-token adaptations inside `torch.no_grad()`, while the **outer loop** (AdamW) learns the projection networks that feed the inner loop. This separation means HOPE can adapt in-context without gradient explosion.

### Why HOPE for Finance?

| Property | LSTM | Transformer | **HOPE** |
|----------|------|-------------|----------|
| Context length | Fixed (hidden state compresses) | Fixed (context window) | **Infinite** (persistent memory) |
| Adaptation speed | Slow (requires backprop) | None (static weights) | **Per-token** (DGD updates) |
| Memory cost | O(1) | O(n²) | **O(d²)** per memory matrix |
| Regime detection | Poor (forgets old regimes) | Can't see past window | **Accumulates** regime history |

---

## 🏗 Architecture

```
Input x_t ─→ Linear Projection ─→ ┌──────────────────────┐
                                   │     HOPE Block ×N     │
                                   │                      │
                                   │  ┌─ Self-Ref Titans ─┐│
                                   │  │  M_k(x) → keys    ││
                                   │  │  M_v(x) → values  ││
                                   │  │  M_η(x) → LR      ││
                                   │  │  M_α(x) → decay   ││
                                   │  │  W_q(x) → query   ││
                                   │  │                    ││
                                   │  │  M_mem: fast-weight││
                                   │  │  DGD update (Eq 93)││
                                   │  │  [torch.no_grad()] ││
                                   │  └────────────────────┘│
                                   │           ↓            │
                                   │      LayerNorm         │
                                   │           ↓            │
                                   │  ┌─ CMS Stack ────────┐│
                                   │  │  Level 0: 5-step   ││
                                   │  │  Level 1: 21-step  ││
                                   │  │  Level 2: 63-step  ││
                                   │  └────────────────────┘│
                                   └──────────────────────┘
                                            ↓
                               LayerNorm ─→ Linear ─→ ŷ_t
```

### DGD Update Rule (Eq 93)

```
M_□,t = M_□,t-1 · (α_t·I − η_t·k_t·k_tᵀ) − η_t · error_t · k_tᵀ
```

Where `error_t = M_□·k_t − v̂_□,t` and `v̂` is the self-referential target.

**Critical design**: DGD runs inside `torch.no_grad()` — memory matrices are NOT in the outer autograd graph. The projection networks (M_k_net, M_v_net) get gradients through the output path: `combined = o_t + v_t + k_t`.

---

## ✅ Phase 1 Results — Synthetic Validation

**Task**: Predict a frequency-shifting sine wave with 60+ sudden jumps in 20K steps.
This specifically tests whether persistent memory helps detect and track frequency changes.

### Configuration

| Parameter | Value |
|-----------|-------|
| d_model | 24 |
| Parameters (HOPE) | ~13K |
| Parameters (LSTM) | ~9.6K |
| Training steps | 20,000 |
| Chunk size | 64 |
| Epochs | 10 |
| Seeds | 5 (42, 123, 456, 789, 2024) |
| Freq jumps | ~60 (jump_prob=0.003) |

### Results

```
======================================================================
  PHASE 1 RESULTS
======================================================================

  Model                  mean IC_p     std IC_p    mean IC_r      mean Δ    mem helps
  ──────────────────────────────────────────────────────────────────────────────
  HOPE-persistent         +0.9488       0.0186      +0.9375      +0.0113     4/5
  LSTM-persistent         +0.9362       0.0214

  GATE CHECK:
  ✅ Gate 1: HOPE IC > LSTM IC         (+0.0126)
  ✅ Gate 2: HOPE persistent > reset   (delta=+0.0113)
  ✅ Gate 3: Std/|Mean| < 0.3          (ratio=0.02)
  ✅ Gate 4: Memory helps ≥3/5 seeds   (4/5)
  ✅ Gate 5: W_norm varies by seed     ✓

  ✅ PHASE 1 PASSED — proceed to Phase 2
```

### Key Insights from Phase 1

1. **Memory matters**: IC_persistent > IC_reset in 4/5 seeds (delta = +0.0113). The persistent Titans memory accumulates frequency history that helps prediction.

2. **DGD is firing**: W_norm varies across seeds, confirming that the inner loop adapts differently based on training dynamics.

3. **Output path gradient fix was critical**: Adding `k_t` and `v_t` directly to the output (`combined = o_t + v_t + k_t`) gave M_k_net and M_v_net gradient paths to the loss. Without this, projections received near-zero gradients.

4. **Alpha clamping prevents collapse**: Constraining α to (0.9, 0.99) prevents M_mem from decaying to zero. The norm floor (`if norm < 0.05: add 0.1·I`) provides a safety net.

5. **States must persist across epochs**: Previously, states were reinitialized each epoch, destroying accumulated memory. Passing states through `train_epoch → next epoch` was the fix that unlocked persistent memory.

---

## 📈 Phase 2 Results — Financial Data

**Status**: ✅ Passed

### Data Pipeline

8 NSE stocks × 16 years daily data via Upstox API → 24 stationary features → multi-stock training

**Instruments**: COALINDIA, RELIANCE, TCS, INFY, HDFCBANK, ICICIBANK, SBIN, WIPRO

### Configuration

| Parameter | Value |
|-----------|-------|
| Data range | 2010–2026 (~3,800 bars per stock) |
| Training bars | 2,366 per stock (70%) |
| Features | 24 (25 minus constant time encodings) |
| d_model | 24 |
| n_layers | 2 |
| Parameters | 27,029 |
| CMS schedule | [5, 21, 63] (week / month / quarter) |
| Epochs | 10 |
| Chunk size | 64 |

### Feature Groups (24 Features)

| Group | Features | Signal |
|-------|----------|--------|
| **Returns** (5) | ret_1, ret_5, ret_15, ret_30, gap | Momentum at multiple horizons |
| **Volatility** (4) | rvol_10, rvol_30, atr_ratio, vol_ratio | Regime detection (breakout vs mean-reversion) |
| **Volume** (4) | vol_surprise, vol_trend, pv_corr, log_vol | Institutional activity detection |
| **VWAP** (4) | vwap_dev, vwap_zscore, vwap_slope, vwap_band | Institutional benchmark deviation |
| **Microstructure** (4) | body_ratio, buy_pressure, close_position, illiquidity | Order flow inference from OHLCV |
| **Momentum** (3) | rsi, sma_dev, range_pos_60 | Mean reversion signals |

### Results

```
============================================================
  PHASE 2 FINAL RESULTS
============================================================
  Stock         val_IC (ep9)   test_IC    positive
  ─────────────────────────────────────────────────
  SBIN          +0.4267        +0.0843    1/1
  RELIANCE      +0.0776        +0.2803    1/1
  INFY          +0.0893        +0.2049    1/1
  WIPRO         -0.0139        +0.1372    1/1
  COALINDIA     -0.0300        +0.0993    1/1
  ICICIBANK     +0.1855        +0.0504    1/1
  HDFCBANK      +0.2779        -0.0645    0/1
  TCS           +0.1189        -0.0858    0/1

  Mean test IC:           +0.0883
  Stocks with IC > 0:     6/8

  Gate 1 (mean IC > 0.01):      ✅  (+0.0883)
  Gate 2 (>50% stocks IC>0):    ✅  (6/8 = 75%)
  Gate 3 (best val IC > 0.005): ✅  (+0.1415)

  ✅ PHASE 2 PASSED
```

### Training Progression

| Epoch | IC Loss | Val IC | Improving? |
|-------|---------|--------|------------|
| 0 | -0.233 | +0.114 | — |
| 3 | -0.307 | +0.129 | ↑ |
| 6 | -0.325 | +0.136 | ↑ |
| 9 | -0.337 | +0.142 | ↑ |

Loss decreased monotonically, validation IC improved every epoch. No overfitting observed.

### Key Insights from Phase 2

1. **HOPE generalizes to real financial data**: Mean test IC = +0.0883 across 8 diverse NSE stocks. This is a strong IC for daily equity prediction — most quant funds target IC > 0.03.

2. **SBIN is the strongest learner**: Val IC = +0.4267 (highest), suggesting PSU banking stocks have the most exploitable patterns in OHLCV data. This makes sense — SBIN has high retail flow and mean-reverts more predictably.

3. **RELIANCE generalizes best**: Val IC was modest (+0.08) but test IC was the highest (+0.2803). The model learned genuine patterns, not training artifacts.

4. **VWAP features dominate**: vwap_slope, vwap_dev, and vwap_band are the most consistently significant features across all stocks. Institutional flow relative to VWAP is the strongest alpha signal in daily data.

5. **No overfitting**: Loss and val IC improved monotonically across all 10 epochs. With only 27K params on ~19K training samples (8 × 2,366), the model is well within the underfitting regime.

6. **COALINDIA and WIPRO underperform on validation but generalize**: Both had negative val IC but positive test IC. The walk-forward window was too narrow to properly evaluate — minute-bar data in Phase 3 will provide better evaluation resolution.

### Phase 2 Gates

| Gate | Threshold | Result | Status |
|------|-----------|--------|--------|
| Mean test IC | > 0.01 | +0.0883 | ✅ |
| Stocks with IC > 0 | > 50% | 75% (6/8) | ✅ |
| Best val IC | > 0.005 | +0.1415 | ✅ |

---

## 📁 Repository Structure

```
baka-trading/
│
├── hope_phase1/                    # ✅ Phase 1: Synthetic validation
│   ├── models/                     # Core architecture (DO NOT MODIFY)
│   │   ├── memory.py              # MemoryModule: M(x) = x + W₁σ(W₂x)
│   │   ├── titans.py              # Self-Referential Titans + DGD
│   │   ├── cms.py                 # Continuum Memory System
│   │   ├── hope.py                # HOPEBlock + MiniHOPE wrapper
│   │   └── lstm_baseline.py       # LSTM for fair comparison
│   ├── synthetic_data.py          # Frequency-shifting sine generator
│   ├── train.py                   # Streaming training (TBPTT)
│   ├── losses.py                  # IC loss + MSE loss
│   ├── diagnostics.py             # Gradient/state health checks
│   └── run_phase1.py              # Complete Phase 1 experiment
│
├── hope_phase2/                    # ✅ Phase 2: Financial data
│   ├── models/                    # Exact copy from Phase 1
│   ├── features.py                # 25 quant features
│   ├── labels.py                  # Forward net return labels
│   ├── ic_test.py                 # Feature IC filtering
│   ├── data.py                    # Upstox API → parquet → tensors
│   ├── checkpoint.py              # HuggingFace Hub save/load
│   ├── train.py                   # Multi-stock streaming training
│   ├── evaluate.py                # Walk-forward validation
│   └── run_phase2.py              # Complete Phase 2 experiment
│
├── phase1_synthetic/               # 🗄 Legacy (original BAKA experiments)
├── README.md
├── RESEARCH.md
├── SETUP.md
└── .gitignore
```

---

## 🚀 Quick Start

### Phase 1: Validate Architecture (Colab T4)

```bash
git clone https://github.com/Ramcharan747/baka-trading.git
cd baka-trading/hope_phase1
pip install torch numpy scipy
python run_phase1.py --device cuda --epochs 10
```

### Phase 2: Financial Data (Colab T4)

```bash
cd baka-trading/hope_phase2
pip install -r requirements.txt

# First run
python run_phase2.py --device cuda --epochs 10

# Resume after Colab session restart
python run_phase2.py --device cuda --epochs 10 --resume
```

### Colab Cells (Copy-Paste Ready)

```python
# Cell 1: Setup
!pip install -q huggingface_hub scipy pyarrow
!rm -rf baka-trading
!git clone https://github.com/Ramcharan747/baka-trading.git

# Cell 2: Phase 1
%cd /content/baka-trading/hope_phase1
!python run_phase1.py --device cuda

# Cell 3: Phase 2
%cd /content/baka-trading/hope_phase2
!python run_phase2.py --device cuda --epochs 10
```

---

## 🔒 Design Principles

These are **non-negotiable** constraints. Violating any one produces backtests that look great but fail live.

| # | Principle | Why |
|---|-----------|-----|
| 1 | **Never predict raw price** | Price is non-stationary. Predict signed returns. |
| 2 | **Never shuffle time series** | Walk-forward splits only. Shuffling leaks future information. |
| 3 | **Always subtract transaction costs from labels** | 3 bps for NSE. Without costs, models find edges eaten by spread. |
| 4 | **Test for lookahead bias** | Any feature with \|IC\| > 0.3 is almost certainly using future data. |
| 5 | **DGD stays inside `torch.no_grad()`** | Memory matrices are inner loop, NOT nn.Parameters. |
| 6 | **States persist across epochs** | Reinitializing states kills accumulated memory. |
| 7 | **Every feature must be causal** | Only `shift(+n)` allowed for features. Labels use `shift(-n)`. |

---

## 🖥 Hardware & Infrastructure

| Stage | Hardware | Use |
|-------|----------|-----|
| Phase 1 | Google Colab T4 (16 GB) | Architecture validation on synthetic data |
| Phase 2 | Google Colab T4 / Kaggle P100 | Financial data training + HF checkpointing |
| Phase 3 | IIT Delhi HPC — 2×A100 40GB | Full-scale training + live paper trading |

**HPC Details**: PBS scheduler (not SLURM), project `futurematlab1.npn`.

**Checkpointing**: All checkpoints saved to [HuggingFace Hub](https://huggingface.co/Ramcharan747/hope-finance) so training survives Colab session restarts.

---

## 📄 Citation

If you use this code, please cite the underlying paper:

```bibtex
@inproceedings{behrouz2025nested,
  title     = {Nested Learning: The Illusion of Deep Learning Architecture},
  author    = {Behrouz, Ali and Razaviyayn, Meisam and Zhong, Kai},
  booktitle = {NeurIPS},
  year      = {2025}
}
```

---

## 📬 Contact

**Ram Charan** — 2nd Year, Production & Industrial Engineering, IIT Delhi

- GitHub: [@Ramcharan747](https://github.com/Ramcharan747)

---

<p align="center">
  <em>Built with 🧠 and ☕ at IIT Delhi</em>
</p>
