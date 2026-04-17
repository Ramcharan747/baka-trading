"""
Ready-to-use Colab cells for baka-trading.

Copy each CELL block into a separate Colab cell, in order.
These use kagglehub for dataset download (no kaggle CLI needed).

The Kaggle Nifty 500 dataset:
    - 500+ symbols, 1-minute bars from 2015 to 2026
    - Files named {SYMBOL}_minute.csv
    - Columns: date, open, high, low, close, volume (NO indicators)
    - Indicators are computed in-pipeline by kaggle_loader.compute_indicators()
"""

# ============================================================
# CELL 1 — Setup: clone repo + install deps + download dataset
# ============================================================
CELL_1 = '''
# === CELL 1: clone repo, install deps, download Kaggle Nifty 500 dataset ===
import os

# 1) Clone the repo
!git clone https://github.com/Ramcharan747/baka-trading.git
%cd /content/baka-trading

# 2) Install deps (Colab already has torch/numpy/pandas/scipy)
!pip install -q yfinance pyarrow kagglehub

# 3) Set your Kaggle API key and download the dataset
#    Get your key from: https://www.kaggle.com/settings -> Account -> API -> "Create New Token"
os.environ["KAGGLE_KEY"] = "KGAT_41981dd2f0533efa5c4a5a8f1de2434c"

import kagglehub
dataset_path = kagglehub.dataset_download(
    "debashis74017/algo-trading-data-nifty-100-data-with-indicators"
)
print(f"Dataset downloaded to: {dataset_path}")

# 4) Inspect what we got
import subprocess
result = subprocess.run(["ls", dataset_path], capture_output=True, text=True)
files = result.stdout.strip().split("\\n")
print(f"\\nTotal files: {len(files)}")
print("First 20 files:")
for f in files[:20]:
    print(f"  {f}")

# 5) List available symbols
!python kaggle_loader.py "{dataset_path}" --list | head -30
'''

# ============================================================
# CELL 2 — Quick sanity check on one symbol
# ============================================================
CELL_2 = '''
# === CELL 2: sanity check — load one symbol and peek at the data ===
import kaggle_loader as kl
import pandas as pd

# Use the dataset_path from Cell 1
# SYMBOL choices: COALINDIA, BEL, ACC, ADANIPORTS, INFY, TCS, HDFCBANK, etc.
SYMBOL = "COALINDIA"

df = kl.load_kaggle_dataset(dataset_path, SYMBOL)
print(f"Rows: {len(df):,}")
print(f"Date range: {df.index.min()} -> {df.index.max()}")
print(f"Columns ({len(df.columns)}): {list(df.columns)}")
print(df.head())

# Compute indicators (the dataset has NO pre-computed indicators)
indicators = kl.compute_indicators(df)
print(f"\\nComputed indicator columns ({len(indicators.columns)}):")
print(list(indicators.columns))
print(indicators.dropna().head())

ohlcv, _ = kl.split_ohlcv_and_indicators(df)
print(f"\\nOHLCV cols : {list(ohlcv.columns)}")
'''

# ============================================================
# CELL 3 — Train LSTM baseline on Kaggle data
# ============================================================
CELL_3 = '''
# === CELL 3: train LSTM baseline on Kaggle data ===
SYMBOL = "COALINDIA"
START  = "2024-01-01"
END    = "2024-06-30"

!python run_experiment.py \\
    --source kaggle --kaggle-path "{dataset_path}" \\
    --symbol {SYMBOL} --start {START} --end {END} \\
    --model lstm --use-kaggle-indicators \\
    --window 60 --batch-size 128 --epochs 5 \\
    --skip-paper-trading \\
    --device cuda
'''

# ============================================================
# CELL 4 — Train Mini-BAKA (and run the CMS ablation sweep)
# ============================================================
CELL_4 = '''
# === CELL 4: train Mini-BAKA and ablate each CMS level ===
SYMBOL = "COALINDIA"
START  = "2024-01-01"
END    = "2024-06-30"

# -1 = full BAKA, 0-3 = zero out that CMS level
for level in [-1, 0, 1, 2, 3]:
    print(f"\\n{'='*60}\\n  BAKA with --cms-ablate {level}\\n{'='*60}")
    !python run_experiment.py \\
        --source kaggle --kaggle-path "{dataset_path}" \\
        --symbol {SYMBOL} --start {START} --end {END} \\
        --model baka --cms-ablate {level} --use-kaggle-indicators \\
        --window 60 --batch-size 128 --epochs 5 \\
        --skip-paper-trading \\
        --device cuda
'''

# ============================================================
# CELL 5 — Paper-trade on held-out tail + summarize all results
# ============================================================
CELL_5 = '''
# === CELL 5: paper trade on held-out tail + collect all walk-forward results ===
import json, glob
import pandas as pd

SYMBOL = "COALINDIA"
START  = "2024-01-01"
END    = "2024-06-30"

# Run paper trading on unseen tail for both models
for model in ["lstm", "baka"]:
    print(f"\\n{'='*60}\\n  Paper trading: {model}\\n{'='*60}")
    !python run_experiment.py \\
        --source kaggle --kaggle-path "{dataset_path}" \\
        --symbol {SYMBOL} --start {START} --end {END} \\
        --model {model} --use-kaggle-indicators \\
        --window 60 --batch-size 128 --epochs 5 \\
        --device cuda

# Collect walk-forward IC and paper-trading metrics from artifacts/
print("\\n\\n" + "="*60)
print(f"  Summary — {SYMBOL}")
print("="*60)

rows = []
for p in sorted(glob.glob("artifacts/wf_*.json")):
    data = json.load(open(p))
    if not data:
        continue
    ics = [r["ic"] for r in data]
    rows.append({
        "file": p.split("/")[-1],
        "n_periods": len(ics),
        "mean_IC": sum(ics) / len(ics),
        "positive_periods": sum(1 for i in ics if i > 0),
    })
print("\\nWalk-forward IC:")
print(pd.DataFrame(rows).to_string(index=False))

print("\\nPaper trading metrics:")
for p in sorted(glob.glob("artifacts/paper_*.json")):
    m = json.load(open(p))
    name = p.split("/")[-1].replace("paper_", "").replace(".json", "")
    print(f"  {name:30s}  return={m['total_return']*100:+6.2f}%  "
          f"sharpe={m['sharpe']:+6.2f}  maxDD={m['max_dd']*100:+6.2f}%  "
          f"trades={m['trades']:4d}  winrate={m['win_rate']*100:4.1f}%")
'''


if __name__ == "__main__":
    print("=" * 60)
    print("  Colab cells for baka-trading")
    print("=" * 60)
    for i, (name, cell) in enumerate([
        ("Setup", CELL_1),
        ("Sanity Check", CELL_2),
        ("LSTM Baseline", CELL_3),
        ("BAKA Ablation", CELL_4),
        ("Paper Trading", CELL_5),
    ], 1):
        print(f"\n{'='*60}")
        print(f"  CELL {i}: {name}")
        print(f"{'='*60}")
        print(cell)
