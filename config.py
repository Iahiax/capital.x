"""Configuration loaded from environment variables.

The original repository contained account credentials in source control. The
application now has safe empty defaults so research/sample mode works without
credentials, while live mode fails with a useful message when configuration is
missing.
"""

from __future__ import annotations

import os
from pathlib import Path
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except ValueError:
        return default


EMAIL = os.getenv("CAPITAL_IDENTIFIER", "")
PASSWORD = os.getenv("CAPITAL_API_PASSWORD", "")
API_KEY = os.getenv("CAPITAL_API_KEY", "")
FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY", "")

USE_DEMO = _env_bool("CAPITAL_USE_DEMO", True)
EPIC = os.getenv("CAPITAL_EPIC", "CS.D.EURUSD.MINI.IP")
RESOLUTION = os.getenv("CAPITAL_RESOLUTION", "MINUTE")
HISTORY_DAYS = _env_int("CAPITAL_HISTORY_DAYS", 365)
REQUEST_TIMEOUT = _env_float("CAPITAL_REQUEST_TIMEOUT", 30.0)
LIVE_CANDLE_LOOKBACK_MINUTES = _env_int("LIVE_CANDLE_LOOKBACK_MINUTES", 15)

TIMEZONE = os.getenv("MARKET_TIMEZONE", "Asia/Riyadh")
try:
    MARKET_TZ = ZoneInfo(TIMEZONE)
except ZoneInfoNotFoundError:
    TIMEZONE = "UTC"
    MARKET_TZ = ZoneInfo("UTC")

LEVERAGE = _env_float("CAPITAL_LEVERAGE", 100.0)
INITIAL_EQUITY = _env_float("INITIAL_EQUITY", 10_000.0)
RISK_PER_TRADE = _env_float("RISK_PER_TRADE", 0.005)
SPREAD = _env_float("SPREAD", 0.0001)
COMMISSION_PER_TRADE = _env_float("COMMISSION_PER_TRADE", 0.0)
MAX_LOOKAHEAD_MINUTES = _env_int("MAX_LOOKAHEAD_MINUTES", 60)
SLIPPAGE_ATR_MULTIPLIER = _env_float("SLIPPAGE_ATR_MULTIPLIER", 0.05)
SWAP_PER_DAY = _env_float("SWAP_PER_DAY", 0.0)
MAX_OPEN_POSITIONS = _env_int("MAX_OPEN_POSITIONS", 1)
MAX_DAILY_DD = _env_float("MAX_DAILY_DD", 0.03)
MAX_TOTAL_DD = _env_float("MAX_TOTAL_DD", 0.20)

MARKET_CLOSE_WEEKDAY = 4
MARKET_CLOSE_HOUR = 23
MARKET_OPEN_WEEKDAY = 0
MARKET_OPEN_HOUR = 0

NEWS_LOOKBACK_MINUTES = _env_int("NEWS_LOOKBACK_MINUTES", 60)
NEWS_COOLDOWN_MINUTES = _env_int("NEWS_COOLDOWN_MINUTES", 30)
META_OOF_SPLITS = _env_int("META_OOF_SPLITS", 3)
WALK_FORWARD_TRAIN_DAYS = _env_int("WALK_FORWARD_TRAIN_DAYS", 90)
WALK_FORWARD_TEST_DAYS = _env_int("WALK_FORWARD_TEST_DAYS", 30)
MODEL_DIR = Path(
    os.getenv("MODEL_DIR", str(Path(__file__).resolve().parent / "models"))
)


def get_base_url() -> str:
    """Return the current Capital.com API base URL."""

    if USE_DEMO:
        return "https://demo-api-capital.backend-capital.com/api/v1"
    return "https://api-capital.backend-capital.com/api/v1"


def validate_live_config() -> None:
    """Raise a clear error before attempting an authenticated API request."""

    missing = [
        name
        for name, value in {
            "CAPITAL_API_KEY": API_KEY,
            "CAPITAL_IDENTIFIER": EMAIL,
            "CAPITAL_API_PASSWORD": PASSWORD,
        }.items()
        if not value
    ]
    if missing:
        raise RuntimeError(
            "Live/demo Capital.com access requires: " + ", ".join(missing)
        )