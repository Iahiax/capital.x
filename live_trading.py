"""Reusable live-order adapter.

This module is deliberately not called by the research pipeline. It provides
one explicit broker boundary for integrations that already have a BrokerClient.
"""

from __future__ import annotations

from broker_client import BrokerClient


def execute_trade(broker: BrokerClient, signal: dict) -> dict | None:
    """Submit one validated signal with its protective levels."""

    direction = signal.get("direction")
    size = float(signal.get("size", 0))
    if direction not in {"BUY", "SELL"}:
        raise ValueError("signal.direction must be BUY or SELL")
    if size <= 0:
        raise ValueError("signal.size must be positive")

    return broker.open_market_order(
        direction=direction,
        size=size,
        stop_loss=signal.get("stop_loss"),
        take_profit=signal.get("take_profit"),
    )
