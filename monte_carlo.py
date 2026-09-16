"""Monte Carlo stress tests for the order of realized trades."""

from __future__ import annotations

import numpy as np


def run_monte_carlo(
    pnls,
    initial_equity: float,
    simulations: int = 10_000,
    seed: int = 7,
) -> dict:
    values = np.asarray(list(pnls), dtype=float)
    if values.size == 0:
        return {
            "simulations": 0,
            "median_final_equity": initial_equity,
            "worst_final_equity": initial_equity,
            "median_max_drawdown": 0.0,
            "worst_max_drawdown": 0.0,
            "probability_of_loss": 0.0,
        }
    if simulations < 1:
        raise ValueError("simulations must be positive")

    rng = np.random.default_rng(seed)
    final_equities = np.empty(simulations)
    drawdowns = np.empty(simulations)
    for index in range(simulations):
        shuffled = rng.permutation(values)
        curve = initial_equity + np.cumsum(shuffled)
        peaks = np.maximum.accumulate(np.r_[initial_equity, curve])
        drawdowns[index] = np.max(peaks[1:] - curve)
        final_equities[index] = curve[-1]

    return {
        "simulations": simulations,
        "median_final_equity": float(np.median(final_equities)),
        "worst_final_equity": float(np.min(final_equities)),
        "median_max_drawdown": float(np.median(drawdowns)),
        "worst_max_drawdown": float(np.max(drawdowns)),
        "probability_of_loss": float(np.mean(final_equities < initial_equity)),
    }