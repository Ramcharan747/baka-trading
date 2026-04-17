"""
Frequency-shifting sine wave — synthetic test for persistent memory.

A model with persistent memory should track the slowly drifting frequency
and predict the next value well. A model without persistent memory (or with
frequent resets) will always be slightly behind after frequency jumps.

This is the litmus test: if BAKA's Titans/CMS memory doesn't help HERE,
it won't help on financial data either.
"""
from __future__ import annotations

import numpy as np


def generate_frequency_shifting_sine(
    n_steps: int = 100_000,
    base_freq: float = 0.1,
    freq_drift: float = 0.0001,
    freq_jump_prob: float = 0.001,
    freq_jump_scale: float = 0.05,
    noise_std: float = 0.05,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generates a sine wave with slowly drifting frequency + occasional jumps.

    Why this tests persistent memory:
    - Slow drift: memory across 100s of steps tracks the frequency trend
    - Sudden jumps: short-term memory (Titans) should detect shifts quickly
    - Long-term drift: CMS should track the multi-week regime change

    Returns:
        x: [n_steps-1] observed noisy values (model input)
        y: [n_steps-1] true next value (prediction target)
        freq: [n_steps-1] true frequency at each step (for analysis)
    """
    rng = np.random.RandomState(seed)

    freq = np.zeros(n_steps)
    freq[0] = base_freq
    phase = 0.0
    x = np.zeros(n_steps)

    for t in range(1, n_steps):
        # Slow drift
        freq[t] = freq[t - 1] + freq_drift * rng.randn()
        freq[t] = np.clip(freq[t], 0.01, 0.5)

        # Occasional sudden jump
        if rng.rand() < freq_jump_prob:
            freq[t] += rng.randn() * freq_jump_scale
            freq[t] = np.clip(freq[t], 0.01, 0.5)

        phase += freq[t]
        x[t] = np.sin(phase) + rng.randn() * noise_std

    # Labels: next-step value (1-step ahead prediction)
    y = np.roll(x, -1)
    y[-1] = 0.0  # undefined last step

    return x[:-1].astype(np.float32), y[:-1].astype(np.float32), freq[:-1].astype(np.float32)


if __name__ == "__main__":
    x, y, freq = generate_frequency_shifting_sine()
    print(f"Generated {len(x):,} steps")
    print(f"Frequency range: {freq.min():.4f} - {freq.max():.4f}")
    print(f"Number of freq jumps (|df| > 0.01): "
          f"{(np.abs(np.diff(freq)) > 0.01).sum()}")
    print(f"Signal std: {x.std():.4f}, noise std: 0.05")
