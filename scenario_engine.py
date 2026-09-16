"""Deterministic execution and data-feed stress scenarios."""

from __future__ import annotations

import pandas as pd


def filter_scenario(df, scenario_type):
    if scenario_type == "trend":
        return df[df["Regime"].isin([1, -1])]
    if scenario_type == "range":
        return df[df["Regime"] == 0]
    if scenario_type == "chaos":
        return df[df["Regime"] == 2]
    if scenario_type in {None, "all"}:
        return df
    raise ValueError("scenario_type must be trend, range, chaos, or all")


def simulate_market_outage(prices: pd.DataFrame, outage_bars: int = 10):
    """Remove a contiguous feed window to test stale-data handling."""

    if outage_bars < 1:
        raise ValueError("outage_bars must be positive")
    if prices.empty:
        return prices.copy()
    start = max((len(prices) - outage_bars) // 2, 0)
    return prices.drop(prices.index[start : start + outage_bars])


def stress_spread(trades: pd.DataFrame, multiplier: float = 5.0) -> pd.DataFrame:
    """Attach a scenario spread without mutating the source trade frame."""

    if multiplier < 1:
        raise ValueError("spread multiplier must be at least 1")
    stressed = trades.copy()
    stressed["ScenarioSpreadMultiplier"] = multiplier
    return stressed