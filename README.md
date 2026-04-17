# BAKA Trading — Multi-Timescale Neural Network for NSE

Research codebase for testing whether a novel multi-timescale memory
architecture (BAKA — Titans short-term memory + 4-level CMS long-term
memory) outperforms LSTM and Mamba-style baselines on Indian equity
intraday data.

> **Status:** Stage 1 (Colab-scale proof of concept) — pipeline
> end-to-end, awaiting minute-bar data for the real comparison.

## Quick links

- **Research gap & baselines:** [RESEARCH.md](./RESEARCH.md)
- **Colab / Kaggle / IITD HPC setup:** [SETUP.md](./SETUP.md)

## What this repo contains

| File | What it does |
|------|-------------|
| `data_download.py` | Pulls NSE data via yfinance / nsepy / Zerodha Kite, caches to parquet |
| `kaggle_loader.py` | Loads Kaggle-hosted NSE datasets (per-symbol CSVs or combined files) |
| `features.py` | 11-13 **stationary** features (returns, vol, momentum, range, time-of-day) |
| `ic_test.py` | Information Coefficient test, lookahead-bias guard, regime-conditioned IC |
| `models.py` | Mini-BAKA (Titans + 4-level CMS + causal attention, ~36K params) + LSTM baseline |
| `train.py` | Sequential walk-forward training — **no shuffling**, IC/Sharpe loss |
| `paper_trading.py` | `PaperTradingSimulator` with costs, slippage, stops, position caps |
| `run_experiment.py` | Master orchestrator — download → features → IC → train → paper trade |

## The rules this codebase is built around

These come from hard-won experience in financial ML. Violate any of them
and your backtest Sharpe will look amazing while live performance is zero.

1. **Never predict raw price.** Always predict returns. Price is
   non-stationary; a model predicting price learns "yesterday's price" and
   achieves tiny MSE while being useless.
2. **Never shuffle financial data.** Walk-forward splits only. Shuffling
   destroys the temporal continuity that BAKA's memory depends on.
3. **Always include transaction costs in labels.** 3 bps round-trip for
   NSE. Without costs, models find tiny edges eaten entirely by spread.
4. **Always test for lookahead bias.** Any feature with |IC| > 0.3 is
   almost certainly using future data — the pipeline raises on this.
5. **Walk-forward validation always.** Train on months 1–6, test on 7.
   Then train on 1–7, test on 8. Never test on data the model saw.

## Quickstart — run on NIFTY daily (3 years)

```bash
pip install -r requirements.txt
python run_experiment.py \
    --symbol NIFTY --start 2022-01-01 --end 2024-12-31 \
    --interval 1d --model lstm \
    --window 20 --batch-size 32 --epochs 3
```

## Quickstart — run on the Kaggle Nifty 100 dataset

```bash
pip install -q kaggle
mkdir -p ~/.kaggle && mv kaggle.json ~/.kaggle/  # your API key
chmod 600 ~/.kaggle/kaggle.json
kaggle datasets download -d debashis74017/algo-trading-data-nifty-100-data-with-indicators -p data/kaggle --unzip

python run_experiment.py \
    --source kaggle --kaggle-path data/kaggle \
    --symbol RELIANCE --start 2020-01-01 --end 2024-12-31 \
    --model baka --use-kaggle-indicators \
    --window 60 --batch-size 128 --epochs 5
```

The Kaggle dataset has indicators (RSI, MACD, etc.) pre-computed. Passing
`--use-kaggle-indicators` merges them into the feature set after the
standard stationary features are computed.

## CMS ablation — the paper's central experiment

```bash
for level in -1 0 1 2 3; do
    python run_experiment.py \
        --symbol NIFTY --start 2020-01-01 --end 2024-12-31 \
        --model baka --cms-ablate $level \
        --window 32 --batch-size 64 --epochs 10
done
```

`--cms-ablate -1` = no ablation (full BAKA). `--cms-ablate 3` zeros the
slowest level (65,536-step update). If BAKA's long-memory story holds,
removing level 3 should degrade IC at regime transitions.

## Model sizing guidance

Financial data has much lower information density than language. A 300M
param model on 2 years of minute bars will memorize training data
completely.

- **Stage 1 (Colab T4):** 10K–50K params → Mini-BAKA default (~36K)
- **Stage 2 (Kaggle P100):** 50K–200K params → `d_model=64, n_layers=3`
- **Stage 3 (IITD HPC A100):** 200K–1M params → `d_model=128, n_layers=4`

## Success gates at each stage

### Stage 1 — before moving to Stage 2
- [ ] 3+ features with |IC| > 0.02 and p < 0.05 on NSE data
- [ ] Zero lookahead bias confirmed
- [ ] LSTM baseline trains without error; OOS IC > 0.01

### Stage 2 — before moving to Stage 3
- [ ] Mini-BAKA IC significantly higher than LSTM (t-test p < 0.05)
- [ ] CMS ablation: removing level-3 hurts IC
- [ ] IC positive in 2+ of 3 regimes (bull / bear / high-vol)
- [ ] Walk-forward IC positive in > 60% of periods

### Stage 3 — before any live trading
- [ ] Paper trading Sharpe > 1.0 over 1 month on unseen data
- [ ] Max drawdown < 15%
- [ ] Win rate > 45% (with avg win > avg loss)
- [ ] IC stable week-over-week in paper trading

## Citation-worthy architecture detail

Mini-BAKA adds two residual summary streams on top of a standard
causal-attention transformer:

- **Titans fast weights** — a small MLP whose bias vector is updated
  by one DGD gradient step every `titans_chunk` tokens. Captures
  intraday surprise.
- **CMS stack** — 4 EMA-style summaries running at periods
  `[16, 256, 4096, 65536]` with learning rates
  `[1e-3, 1e-4, 1e-5, 1e-6]`. Level 0 sees microstructure; level 3
  persists across days.

Both structures are cleared by `model.reset_memory()` at the start of
every training epoch and every paper-trading session.

## License / contact

Research code — no license yet. Contact: Ram Charan (2nd year PIE, IIT Delhi).
