# BAKA Trading — Setup Instructions

This repo runs end-to-end on three environments. Pick one.

---

## 1. Google Colab (Stage 1 — proof of concept, T4 free tier)

### Cell 1 — clone / upload
```python
# If you've pushed this to a private GitHub repo:
!git clone https://github.com/<your-user>/baka-trading.git
%cd baka-trading

# Or upload the files manually via the left sidebar, then:
# %cd /content/baka-trading
```

### Cell 2 — install (Colab already has torch / numpy / pandas / scipy)
```python
!pip install -q yfinance pyarrow
# Skip nsepy — it hits NSE from Colab's US IPs and fails intermittently.
# yfinance + ^NSEI is fine for daily NIFTY 2015-present.
```

### Cell 3 — smoke test (run this FIRST to verify the stack is healthy)
```python
!python features.py
!python ic_test.py
!python models.py
```
Expected: each prints a small shape/param summary with no exceptions.

### Cell 4 — full experiment (LSTM baseline, ~2 min)
```python
!python run_experiment.py \
    --symbol NIFTY --start 2022-01-01 --end 2024-12-31 \
    --interval 1d --model lstm \
    --window 20 --batch-size 32 --epochs 3 \
    --skip-paper-trading
```

### Cell 5 — BAKA with CMS ablation sweep (~10 min)
```python
for level in [-1, 0, 1, 2, 3]:
    !python run_experiment.py \
        --symbol NIFTY --start 2022-01-01 --end 2024-12-31 \
        --interval 1d --model baka --cms-ablate {level} \
        --window 20 --batch-size 32 --epochs 3 \
        --skip-paper-trading
```

### Cell 6 — collect results
```python
import json, glob, pandas as pd
rows = []
for p in glob.glob("artifacts/wf_*.json"):
    data = json.load(open(p))
    ics = [r["ic"] for r in data]
    rows.append({"file": p, "n_periods": len(ics),
                 "mean_ic": sum(ics)/len(ics) if ics else float("nan")})
pd.DataFrame(rows).sort_values("mean_ic", ascending=False)
```

### Colab gotchas
- T4 is fine for Mini-BAKA (~36K params). Don't scale past d_model=64.
- yfinance 1-min bars are only available for the last ~7 days — use 1d.
- `!pip install nsepy` sometimes breaks due to BeautifulSoup version pins.
  Stick with yfinance on Colab.

---

## 2. Kaggle (Stage 2 — full comparison, P100 30h/week free)

### Notebook setup
1. Create a new notebook, attach **GPU P100** accelerator.
2. Upload the repo as a Kaggle Dataset (`baka-trading-code`) or clone from GitHub.
3. Add `"Internet: on"` in the notebook settings (needed for yfinance).

### Cell 1 — setup
```python
!cp -r /kaggle/input/baka-trading-code/* /kaggle/working/
%cd /kaggle/working
!pip install -q yfinance pyarrow
```

### Cell 2 — longer experiment, daily NIFTY + BANKNIFTY
```python
for sym in ["NIFTY", "BANKNIFTY"]:
    for model in ["lstm", "baka"]:
        !python run_experiment.py \
            --symbol {sym} --start 2015-01-01 --end 2024-12-31 \
            --interval 1d --model {model} \
            --window 32 --batch-size 64 --epochs 10
```

### Cell 3 — CMS ablation (the paper's central experiment)
```python
# Zero out each CMS level one at a time — if level-3 (65536) matters,
# removing it should degrade IC in regime transitions.
for level in [-1, 0, 1, 2, 3]:
    !python run_experiment.py \
        --symbol NIFTY --start 2015-01-01 --end 2024-12-31 \
        --interval 1d --model baka --cms-ablate {level} \
        --window 32 --batch-size 64 --epochs 10
```

### Cell 4 — persist artifacts
```python
# Kaggle saves /kaggle/working to the notebook output — artifacts/ is kept.
!ls -la artifacts
```

### Kaggle gotchas
- P100 is ~2× faster than T4. You can push `d_model` up to 128 and `n_layers`
  to 4 without running out of VRAM.
- Watch the 30h/week GPU quota — kill long-running experiments if the IC
  isn't moving after 5 minutes.

---

## 3. IITD HPC (Stage 3 — full experiment, 2× A100, PBS scheduler)

### One-time setup
```bash
ssh <netid>@hpc.iitd.ac.in
cd $SCRATCH
git clone <your-repo> baka-trading
cd baka-trading

module load compiler/python/3.10 compiler/cuda/12.1
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### PBS job script — `run.pbs`
```bash
#!/bin/bash
#PBS -N baka-trading
#PBS -P futurematlab1.npn
#PBS -q standard
#PBS -l select=1:ncpus=8:ngpus=1:centos=icelake
#PBS -l walltime=04:00:00
#PBS -o logs/job.out
#PBS -e logs/job.err

cd $PBS_O_WORKDIR
module load compiler/python/3.10 compiler/cuda/12.1
source .venv/bin/activate

mkdir -p logs artifacts

# Full experiment — 10 years of daily data, LSTM + BAKA + CMS ablations.
for sym in NIFTY BANKNIFTY; do
    for model in lstm baka; do
        python run_experiment.py \
            --symbol $sym --start 2015-01-01 --end 2024-12-31 \
            --interval 1d --model $model \
            --window 32 --batch-size 128 --epochs 20 \
            --device cuda
    done
done

# CMS ablation on NIFTY
for level in -1 0 1 2 3; do
    python run_experiment.py \
        --symbol NIFTY --start 2015-01-01 --end 2024-12-31 \
        --interval 1d --model baka --cms-ablate $level \
        --window 32 --batch-size 128 --epochs 20 \
        --device cuda
done
```

### Submit / monitor
```bash
# Submit
qsub run.pbs

# Check status
qstat -u $USER

# Tail logs live
tail -f logs/job.out

# Kill if needed
qdel <job-id>
```

### IITD gotchas
- **Use `qsub`, not `sbatch`** — IITD uses PBS, not SLURM.
- The `icelake` constraint matches A100 nodes; drop it if your project has
  fallback V100 allocation.
- `$SCRATCH` is purged periodically — copy final artifacts to `$HOME` after
  the run completes.
- Internet from compute nodes may be blocked — pre-download data on the
  login node into `data/*.parquet`, then submit the job. The pipeline
  reads the cache automatically.

---

## Reproducing the Stage 1 smoke-test results

On my MacBook (CPU only), with the default args above on NIFTY daily
2022-01-01 to 2024-12-31 (673 aligned samples, 11 features):

| Model      | Mean walk-forward IC | Periods IC > 0 |
|------------|----------------------|----------------|
| LSTM       | +0.55                | 4/4            |
| Mini-BAKA  | +0.52                | 4/4            |

**Important caveat:** with only ~32 validation samples per walk-forward
period, IC has high variance — these numbers are about "the plumbing
works", not "we beat the market". Stage 2 and 3 runs on 10 years of data
will give IC estimates you can trust.

## Next steps after Stage 1 passes

1. Get Zerodha Kite API credentials → 1-minute bars → 375× more samples per day.
2. Add LOB features (`order_imbalance`, `spread_bps`) — these are the single
   strongest short-horizon predictors.
3. Swap in a DeepLOB-lite baseline (CNN + LSTM over the order book) for a
   fairer comparison than plain LSTM.
4. Run the CMS ablation at Stage 2 scale — that's the paper's main result.
