"""Runtime configuration.

Credentials are deliberately read from environment variables. The old project
contained live credentials in source control, so never put account values back
in this file or in a committed ``.env`` file.
"""

import os
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent
MODEL_DIR = PROJECT_DIR / "models"

API_KEY = os.getenv("CAPITAL_API_KEY", "")
EMAIL = os.getenv("CAPITAL_IDENTIFIER", "")
PASSWORD = os.getenv("CAPITAL_API_PASSWORD", "")
USE_DEMO = os.getenv("CAPITAL_USE_DEMO", "true").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
EPIC = os.getenv("CAPITAL_EPIC", "CS.D.EURUSD.MINI.IP")
RESOLUTION = os.getenv("CAPITAL_RESOLUTION", "MINUTE")

FULL_YEAR_MINUTES = 525600
BATCH_SIZE = 1000
HISTORY_DAYS = int(os.getenv("CAPITAL_HISTORY_DAYS", "365"))

INITIAL_EQUITY = float(os.getenv("INITIAL_EQUITY", "10000"))
RISK_PER_TRADE = float(os.getenv("RISK_PER_TRADE", "0.005"))
MAX_LOOKAHEAD_MINUTES = int(os.getenv("MAX_LOOKAHEAD_MINUTES", "50"))

SPREAD = float(os.getenv("SPREAD", "0.0001"))
COMMISSION_PER_TRADE = float(os.getenv("COMMISSION_PER_TRADE", "0.0"))

FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY", "")
NEWS_LOOKBACK_MINUTES = int(os.getenv("NEWS_LOOKBACK_MINUTES", "60"))
NEWS_COOLDOWN_MINUTES = int(os.getenv("NEWS_COOLDOWN_MINUTES", "30"))


def validate_capital_credentials() -> None:
    """Fail with an actionable message instead of sending an empty login."""

    missing = [
        name
        for name, value in (
            ("CAPITAL_API_KEY", API_KEY),
            ("CAPITAL_IDENTIFIER", EMAIL),
            ("CAPITAL_API_PASSWORD", PASSWORD),
        )
        if not value
    ]
    if missing:
        raise RuntimeError(
            "Missing Capital.com credentials: "
            + ", ".join(missing)
            + ". Add them as environment secrets before live data mode."
        )