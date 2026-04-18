"""
Synthetic data generators for HOPE Phase 1 testing.

The frequency-shifting sine wave specifically tests:
  1. Short-term memory: detect sudden frequency jumps quickly
  2. Long-term memory: track slow drift across hundreds of steps

Why this task favors persistent memory over LSTM:
  - LSTM's fixed lookback cannot track slow drift as well
  - LSTM state reset kills the learned frequency estimate
  - HOPE's persistent fast-weight matrix should accumulate frequency knowledge
"""
from __future__ import annotations

import numpy as np


def generate_frequency_shifting_sine(
    n_steps: int = 100000,
    seed: int = 42,
    noise_std: float = 0.05,
    drift_rate: float = 0.0001,
    jump_prob: float = 0.003,
    jump_std: float = 0.05,
    freq_min: float = 0.01,
    freq_max: float = 0.5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate a sine wave with slowly drifting frequency + sudden jumps.

    Args:
        n_steps:    Total number of timesteps
        seed:       Random seed for reproducibility
        noise_std:  Observation noise standard deviation
        drift_rate: Standard deviation of per-step frequency drift
        jump_prob:  Probability of a sudden frequency jump per step
        jump_std:   Standard deviation of jump magnitude
        freq_min:   Minimum allowed frequency
        freq_max:   Maximum allowed frequency

    Returns:
        x:    [n_steps-1] input signal (sin values + noise)
        y:    [n_steps-1] prediction target (next-step value)
        freq: [n_steps-1] true frequency at each timestep
    """
    rng = np.random.RandomState(seed)
    freq = np.zeros(n_steps)
    freq[0] = 0.1
    phase = 0.0
    x = np.zeros(n_steps)

    for t in range(1, n_steps):
        # Slow drift
        freq[t] = freq[t - 1] + drift_rate * rng.randn()
        freq[t] = np.clip(freq[t], freq_min, freq_max)

        # Sudden jump (rare but important for testing)
        if rng.rand() < jump_prob:
            freq[t] += rng.randn() * jump_std
            freq[t] = np.clip(freq[t], freq_min, freq_max)

        phase += freq[t]
        x[t] = np.sin(phase) + rng.randn() * noise_std

    # Target: predict next value
    y = np.roll(x, -1)
    y[-1] = 0

    return (
        x[:-1].astype(np.float32),
        y[:-1].astype(np.float32),
        freq[:-1],
    )
