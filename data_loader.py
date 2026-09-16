"""Historical and deterministic sample market data loaders."""

from __future__ import annotations

import logging
import time

import numpy as np
import pandas as pd
import requests

import config
from session_manager import create_session

logger = logging.getLogger(__name__)


def _empty_candles() -> pd.DataFrame:
    return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])

def normalize_candles(candles: pd.DataFrame | None) -> pd.DataFrame:
    """Return candles in the indexed OHLCV shape used by the feature pipeline."""

    if candles is None or candles.empty:
        return _empty_candles().rename_axis("Time")

    normalized = candles.copy()
    if "Time" in normalized.columns:
        normalized["Time"] = pd.to_datetime(normalized["Time"], utc=True)
        normalized = normalized.set_index("Time")
    elif not isinstance(normalized.index, pd.DatetimeIndex):
        raise TypeError("Candle data must have a DatetimeIndex or Time column")
    else:
        normalized.index = pd.to_datetime(normalized.index, utc=True)

    required = ["Open", "High", "Low", "Close", "Volume"]
    missing = [column for column in required if column not in normalized.columns]
    if missing:
        raise ValueError(f"Missing candle columns: {missing}")

    normalized = normalized[required].copy()
    normalized = normalized[~normalized.index.duplicated(keep="last")]
    return normalized.sort_index()
def fetch_batch(CST: str, XST: str, start: str, end: str) -> pd.DataFrame:
    url = f"{config.get_base_url()}/prices/{config.EPIC}/{config.RESOLUTION}"
    headers = {
        "X-CAP-API-KEY": config.API_KEY,
        "CST": CST,
        "X-SECURITY-TOKEN": XST,
    }
    params = {"from": start, "to": end, "pageSize": 1000}

    try:
        response = requests.get(
            url, headers=headers, params=params, timeout=config.REQUEST_TIMEOUT
        )
        response.raise_for_status()
        data = response.json()
    except (requests.RequestException, ValueError) as exc:
        logger.error("Capital.com data request failed: %s", exc)
        return _empty_candles()

    if "prices" not in data:
        logger.error("Capital.com response did not contain prices")
        return _empty_candles()

    rows = []
    for item in data["prices"]:
        try:
            if "snapshotTime" not in item:
                continue
            rows.append(
                {
                    "Time": pd.to_datetime(item["snapshotTime"], utc=True),
                    "Open": float(item["openPrice"]["bid"]),
                    "High": float(item["highPrice"]["bid"]),
                    "Low": float(item["lowPrice"]["bid"]),
                    "Close": float(item["closePrice"]["bid"]),
                    "Volume": float(item.get("lastTradedVolume", 0) or 0),
                }
            )
        except (KeyError, TypeError, ValueError):
            continue

    if not rows:
        return _empty_candles()
    return pd.DataFrame(rows)

def fetch_recent_candles(
    CST: str,
    XST: str,
    lookback_minutes: int | None = None,
    end: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Fetch the latest minute candles for one live-trading poll."""

    minutes = (
        lookback_minutes
        if lookback_minutes is not None
        else config.LIVE_CANDLE_LOOKBACK_MINUTES
    )
    if minutes <= 0:
        raise ValueError("lookback_minutes must be positive")

    end_time = pd.Timestamp.now(tz="UTC") if end is None else pd.Timestamp(end)
    end_time = (
        end_time.tz_localize("UTC")
        if end_time.tzinfo is None
        else end_time.tz_convert("UTC")
    )
    start_time = end_time - pd.Timedelta(minutes=minutes)
    candles = fetch_batch(
        CST,
        XST,
        start_time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        end_time.strftime("%Y-%m-%dT%H:%M:%SZ"),
    )
    return normalize_candles(candles)
def load_full_year_data(history_days: int | None = None) -> pd.DataFrame:
    """Load historical candles in API-sized weekly batches."""

    config.validate_live_config()
    days = history_days if history_days is not None else config.HISTORY_DAYS
    if days <= 0:
        raise ValueError("history_days must be positive")

    CST, XST = create_session()
    end = pd.Timestamp.now(tz="UTC")
    start = end - pd.Timedelta(days=days)

    all_data = []
    current = start

    while current < end:
        batch_end = min(current + pd.Timedelta(days=7), end)
        df_batch = fetch_batch(
            CST, XST,
            current.strftime("%Y-%m-%dT%H:%M:%SZ"),
            batch_end.strftime("%Y-%m-%dT%H:%M:%SZ")
        )

        if not df_batch.empty:
            all_data.append(df_batch)

        current = batch_end
        time.sleep(0.5)

    if not all_data:
        raise RuntimeError("Capital.com returned no candles")

    return normalize_candles(pd.concat(all_data, ignore_index=True))


def load_recent_data(minutes: int = 240) -> pd.DataFrame:
    """Fetch a rolling window used to refresh the live feature context."""

    if minutes < 30:
        raise ValueError("minutes must be at least 30")
    config.validate_live_config()
    cst, xst = create_session()
    end = pd.Timestamp.now(tz="UTC")
    start = end - pd.Timedelta(minutes=minutes)
    candles = fetch_batch(
        cst,
        xst,
        start.strftime("%Y-%m-%dT%H:%M:%SZ"),
        end.strftime("%Y-%m-%dT%H:%M:%SZ"),
    )
    if candles.empty:
        raise RuntimeError("Capital.com returned no recent candles")
    return candles.drop_duplicates(subset=["Time"]).set_index("Time").sort_index()


def generate_sample_data(periods: int = 12_000, seed: int = 7) -> pd.DataFrame:
    """Generate deterministic OHLCV candles for offline tests and demos."""

    if periods < 400:
        raise ValueError("periods must be at least 400")

    rng = np.random.default_rng(seed)
    index = pd.date_range(
        "2024-01-01 00:00:00", periods=periods, freq="min", tz="UTC"
    )
    segment = max(periods // 4, 1)
    drifts = np.repeat([0.00003, -0.000025, 0.0, 0.00002], segment)
    drifts = np.resize(drifts, periods)
    returns = drifts + rng.normal(0, 0.00012, periods)
    close = 1.10 * np.exp(np.cumsum(returns))
    open_ = np.r_[close[0], close[:-1]]
    spread = np.abs(rng.normal(0.00012, 0.00004, periods)) + 0.00002
    high = np.maximum(open_, close) + spread
    low = np.minimum(open_, close) - spread
    volume = rng.lognormal(mean=5.0, sigma=0.35, size=periods)

    return pd.DataFrame(
        {
            "Open": open_,
            "High": high,
            "Low": low,
            "Close": close,
            "Volume": volume,
        },
        index=index,
    )

def merge_candles(
    historical: pd.DataFrame, latest: pd.DataFrame
) -> pd.DataFrame:
    """Merge a fresh API response into history, preferring the fresh candle."""

    historical = normalize_candles(historical)
    latest = normalize_candles(latest)
    if historical.empty:
        return latest
    if latest.empty:
        return historical

    merged = pd.concat([historical, latest])
    return merged[~merged.index.duplicated(keep="last")].sort_index()
