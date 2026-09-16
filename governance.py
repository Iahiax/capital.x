"""Explicit three-part trade approval gate."""

from __future__ import annotations


def approve_trade(
    *,
    signal_present: bool,
    risk_allowed: bool,
    environment_safe: bool,
    model_disagreement: float = 0.0,
    max_model_disagreement: float = 0.35,
) -> tuple[bool, str]:
    if not signal_present:
        return False, "no_signal"
    if not risk_allowed:
        return False, "risk_guard"
    if not environment_safe:
        return False, "environment_guard"
    if model_disagreement > max_model_disagreement:
        return False, "model_disagreement"
    return True, "approved"