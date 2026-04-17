# Research Summary — Baselines & Prior Art

## 200-word synthesis

**DeepLOB (Zhang, Zohren, Roberts 2019, IEEE TSP 67(11))** is the strongest
tick-level baseline: CNN + Inception module over the limit order book "image"
(10 levels × bid/ask × price/size), followed by a 64-unit LSTM predicting
mid-price movement labels (up/stay/down). It learned universal microstructure
features that transferred across LSE instruments. The relevant takeaway for
BAKA: LOB structure is spatial *and* temporal, and the LSTM component captures
only ~100 ticks of memory — a clear ceiling that BAKA's CMS is designed to
break.

**Temporal Fusion Transformer (Lim et al. 2019, arXiv:1912.09363)** is the
canonical deep-learning forecasting baseline: gated variable selection +
LSTM encoder + interpretable multi-head attention, with quantile loss.
On noisy financial data it beats DeepAR and MQ-RNN by ~7–9% on quantile loss.
TFT is interpretable but expensive — hundreds of thousands of params for
modest horizons.

**Mamba / S4 on finance (CMDMamba 2025, MambaTS, CryptoMamba)** show selective
state-space models match or beat Transformers at linear cost on financial
sequences. But they still have a *single* recurrent timescale — the core
weakness BAKA attacks with its 4-level CMS running at [16, 256, 4096, 65536]
step schedules.

**NSE-specific literature** is limited to day-wise LSTM/CNN studies with
weak walk-forward validation; a minute-bar multi-timescale study on NSE
would be publishable.

## The publishable research gap

| Dimension        | DeepLOB | TFT  | Mamba/SSM | **BAKA (ours)**                     |
|------------------|:-------:|:----:|:---------:|:-----------------------------------:|
| Memory timescales| 1 (LSTM)| 1    | 1         | **4 (CMS) + 1 fast (Titans)**       |
| State persists across sessions | no | no | no | **yes (CMS level-3 = 65536 steps)** |
| Online adaptation during trading | no | no | no | **yes (Titans inner loop)**         |
| Attention cost   | O(n²)   | O(n²)| O(n)      | O(n) + local attention              |

**Target claim:** On NSE minute-bar data, BAKA's multi-timescale persistent
memory yields higher mean walk-forward IC and Sharpe than DeepLOB, TFT, and
a matched-param Mamba baseline, with the advantage concentrated in regime
transitions (where short-term memory fails and long-term memory pays off).

## Key citations

- [DeepLOB (arXiv:1808.03668)](https://arxiv.org/abs/1808.03668)
- [Temporal Fusion Transformer (arXiv:1912.09363)](https://arxiv.org/abs/1912.09363)
- [CMDMamba for financial time series (PMC12303894, 2025)](https://pmc.ncbi.nlm.nih.gov/articles/PMC12303894/)
- [MambaTS (ICLR 2025)](https://openreview.net/forum?id=vEtDApqkNR)
- [State-Space Models for Market Microstructure (Kinlay 2026)](https://jonathankinlay.com/2026/03/state-space-models-for-market-microstructure-can-mamba-replace-transformers-in-high-frequency-finance/)
- [NSE Stock Market Prediction Using Deep-Learning Models](https://www.sciencedirect.com/science/article/pii/S1877050918307828)

## Implications for this pipeline

1. Must include **DeepLOB-lite** as a third baseline (not just LSTM) once we
   secure LOB data via Kite.
2. Evaluation must emphasize **regime-conditioned IC** — that is where BAKA's
   multi-timescale story pays off; flat averaged IC will wash it out.
3. CMS ablation (drop each of the 4 levels one at a time) is the single
   most important experiment for the paper. Bake it into `run_experiment.py`.
