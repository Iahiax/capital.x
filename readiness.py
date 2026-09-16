"""Explicit readiness gates for continuous and live execution."""

from __future__ import annotations

import os


def validate_live_readiness(validation: dict) -> None:
    """Raise unless chronological validation and explicit live approval passed."""

    if not validation.get("approved", False):
        raise RuntimeError(
            "LIVE is blocked: walk-forward validation did not approve the strategy."
        )
    if os.getenv("CAPITAL_USE_DEMO", "true").strip().lower() != "false":
        raise RuntimeError(
            "LIVE is blocked: set CAPITAL_USE_DEMO=false only after reviewing the gate."
        )
    if os.getenv("LIVE_TRADING_APPROVED", "").strip().upper() != "YES":
        raise RuntimeError(
            "LIVE is blocked: set LIVE_TRADING_APPROVED=YES after manual review."
        )