"""Capital.com session handling and historical price loading."""

import time

import numpy as np
import pandas as pd
import requests

from config import (
    API_KEY,
    BATCH_SIZE,
    EMAIL,
    EPIC,
    HISTORY_DAYS,
    PASSWORD,
    RESOLUTION,
    USE_DEMO,
    validate_capital_credentials,
)

BASE_URL = (
    "https://demo-api-capital.backend-capital.com/api/v1"
    if USE_DEMO else
    "https://api-capital.backend-capital.com/api/v1"
)

def create_session():
    validate_capital_credentials()
    url = f"{BASE_URL}/session"

    headers = {
        "X-CAP-API-KEY": API_KEY,
        "Content-Type": "application/json"
    }

    data = {
        "identifier": EMAIL,
        "password": PASSWORD,
        "encryptedPassword": False
    }

    try:
        r = requests.post(url, headers=headers, json=data, timeout=30)
    except requests.RequestException as exc:
        raise RuntimeError(f"Capital.com session request failed: {exc}") from exc

    if r.status_code != 200:
        try:
            error_code = r.json().get("errorCode", "unknown")
        except ValueError:
            error_code = "unknown"
        raise RuntimeError(
            f"Capital.com session rejected the credentials (HTTP {r.status_code}, "
            f"{error_code}). Check the API password and demo/live endpoint."
        )

    cst = r.headers.get("CST")
    security_token = r.headers.get("X-SECURITY-TOKEN")

    if not cst or not security_token:
        raise RuntimeError(
            "Capital.com did not return both CST and X-SECURITY-TOKEN headers."
        )

    print("Capital.com session created successfully.")
    return cst, security_token


def fetch_batch(CST, XST, start, end):
    url = f"{BASE_URL}/prices/{EPIC}/{RESOLUTION}"

    headers = {
        "X-CAP-API-KEY": API_KEY,
        "CST": CST,
        "X-SECURITY-TOKEN": XST,
    }

    params = {
        "from": start,
        "to": end,
        "pageSize": BATCH_SIZE,
    }

    try:
        r = requests.get(url, headers=headers, params=params, timeout=30)
    except requests.RequestException as exc:
        raise RuntimeError(f"Capital.com price request failed: {exc}") from exc

    if r.status_code != 200:
        raise RuntimeError(
            f"Capital.com price request failed with HTTP {r.status_code}."
        )

    try:
        data = r.json()
    except ValueError as exc:
        raise RuntimeError("Capital.com returned invalid JSON for price data.") from exc

    if "prices" not in data:
        raise RuntimeError("Capital.com response did not contain a prices field.")

    rows = []
    for item in data["prices"]:
        if "snapshotTime" not in item:
            continue

        rows.append({
            "Time": pd.to_datetime(item["snapshotTime"], utc=True),
            "Open": item["openPrice"]["bid"],
            "High": item["highPrice"]["bid"],
            "Low": item["lowPrice"]["bid"],
            "Close": item["closePrice"]["bid"],
            "Volume": item.get("lastTradedVolume", 0)
        })

    return pd.DataFrame(rows).drop_duplicates("Time")


def load_full_year_data():
    print(f"Loading {HISTORY_DAYS} days of Capital.com price data.")

    CST, XST = create_session()

    end = pd.Timestamp.now(tz="UTC")
    start = end - pd.Timedelta(days=HISTORY_DAYS)

    all_data = []
    current = start

    while current < end:
        # The API page size is limited; half-day windows keep minute data
        # within one response and avoid silently truncating each batch.
        batch_end = min(current + pd.Timedelta(hours=12), end)

        print(f"Loading prices from {current} to {batch_end}")

        df_batch = fetch_batch(
            CST, XST,
            current.strftime("%Y-%m-%dT%H:%M:%SZ"),
            batch_end.strftime("%Y-%m-%dT%H:%M:%SZ")
        )

        if not df_batch.empty:
            all_data.append(df_batch)

        current = batch_end
        time.sleep(1.05)

    if not all_data:
        raise RuntimeError("Capital.com returned no historical price data.")

    df = pd.concat(all_data, ignore_index=True)

    if "Time" not in df.columns:
        raise RuntimeError("Historical price data is missing the Time column.")

    df = df.drop_duplicates().set_index("Time").sort_index()

    print("Loaded candles:", len(df))
    return df


def generate_sample_data(rows: int = 3000, seed: int = 7) -> pd.DataFrame:
    """Create deterministic minute candles for safe local smoke tests."""

    if rows < 400:
        raise ValueError("Sample data requires at least 400 rows.")

    rng = np.random.default_rng(seed)
    times = pd.date_range(
        end=pd.Timestamp.now(tz="UTC").floor("min"),
        periods=rows,
        freq="min",
    )
    returns = rng.normal(0, 0.00025, rows)
    close = 1.08 + np.cumsum(returns)
    open_price = np.concatenate(([close[0]], close[:-1]))
    spread = rng.uniform(0.00005, 0.00035, rows)
    high = np.maximum(open_price, close) + spread
    low = np.minimum(open_price, close) - spread
    volume = rng.integers(80, 220, rows)

    return pd.DataFrame(
        {
            "Open": open_price,
            "High": high,
            "Low": low,
            "Close": close,
            "Volume": volume,
        },
        index=times,
    )
