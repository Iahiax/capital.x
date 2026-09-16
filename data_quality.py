"""Market-candle validation and conservative outlier repair."""

from __future__ import annotations

import numpy as np
import pandas as pd


def sanitize_candles(df: pd.DataFrame) -> pd.DataFrame:
    required = ["Open", "High", "Low", "Close", "Volume"]
    missing = set(required).difference(df.columns)
    if missing:
        raise ValueError(f"Missing candle columns: {sorted(missing)}")

    clean = df.copy()
    clean[required] = clean[required].apply(pd.to_numeric, errors="coerce")
    clean = clean.dropna(subset=required)
    valid_ohlc = (
        clean["High"].ge(clean[["Open", "Close", "Low"]].max(axis=1))
        & clean["Low"].le(clean[["Open", "Close", "High"]].min(axis=1))
        & clean["Volume"].ge(0)
    )
    clean = clean.loc[valid_ohlc].copy()
    returns = clean["Close"].pct_change()
    median = returns.rolling(101, center=True, min_periods=20).median()
    mad = (returns - median).abs().rolling(101, center=True, min_periods=20).median()
    suspicious_return = (returns - median).abs().gt(8 * (mad + 1e-8))
    replacement = clean["Close"].rolling(5, center=True, min_periods=1).median()
    for column in ["Open", "High", "Low", "Close"]:
        clean.loc[suspicious_return, column] = replacement.loc[suspicious_return]
    volume_median = clean["Volume"].rolling(101, center=True, min_periods=20).median()
    suspicious_volume = clean["Volume"].gt(20 * volume_median)
    clean.loc[suspicious_volume, "Volume"] = volume_median.loc[suspicious_volume]
    return clean.replace([np.inf, -np.inf], np.nan).dropna(subset=required)