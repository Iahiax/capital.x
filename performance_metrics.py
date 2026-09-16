"""Stability and component metrics for research reports."""

from __future__ import annotations

import pandas as pd


def stability_metrics(trades_df: pd.DataFrame) -> dict:
    executed = trades_df[trades_df["Status"] == "EXECUTED"].copy()
    if executed.empty:
        return {
            "trades": 0,
            "weekly_pnl_std": 0.0,
            "max_consecutive_losses": 0,
            "longest_no_trade_gap_days": 0.0,
        }
    pnls = executed["PnL"].to_numpy(dtype=float)
    loss_runs = []
    current = 0
    for pnl in pnls:
        current = current + 1 if pnl < 0 else 0
        loss_runs.append(current)
    times = pd.to_datetime(executed["Time"], utc=True).sort_values()
    gap_days = times.diff().dt.total_seconds().div(86400).max()
    week = pd.to_datetime(executed["Time"], utc=True).dt.tz_localize(None).dt.to_period("W")
    weekly = executed.assign(Week=week).groupby("Week")["PnL"].sum()
    return {
        "trades": len(executed),
        "weekly_pnl_std": float(weekly.std(ddof=0) if len(weekly) else 0.0),
        "max_consecutive_losses": max(loss_runs, default=0),
        "longest_no_trade_gap_days": float(gap_days if pd.notna(gap_days) else 0.0),
    }