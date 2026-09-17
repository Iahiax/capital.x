"""Small Capital.com trading client."""

from __future__ import annotations

import logging

import requests

import config
from session_manager import create_session

logger = logging.getLogger(__name__)


class BrokerClient:
    def __init__(self):
        self.cst, self.xst = create_session()

    def _headers(self) -> dict[str, str]:
        return {
            "X-CAP-API-KEY": config.API_KEY,
            "CST": self.cst,
            "X-SECURITY-TOKEN": self.xst,
            "Content-Type": "application/json",
        }

    def get_open_positions(self) -> list[dict] | None:
        url = f"{config.get_base_url()}/positions"
        try:
            response = requests.get(
                url, headers=self._headers(), timeout=config.REQUEST_TIMEOUT
            )
            response.raise_for_status()
            return response.json().get("positions", [])
        except (requests.RequestException, ValueError) as exc:
            logger.error("Capital.com positions request failed: %s", exc)
            return None

    def get_recent_candles(
        self,
        lookback_minutes: int | None = None,
        end=None,
    ):
        """Fetch fresh market candles using this client's authenticated session."""

        from data_loader import fetch_recent_candles

        return fetch_recent_candles(
            self.cst,
            self.xst,
            lookback_minutes=lookback_minutes,
            end=end,
        )

    def open_market_order(
        self,
        direction: str,
        size: float,
        stop_loss: float | None = None,
        take_profit: float | None = None,
    ) -> dict | None:
        if direction not in {"BUY", "SELL"}:
            raise ValueError("direction must be BUY or SELL")
        if size <= 0:
            raise ValueError("size must be positive")

        url = f"{config.get_base_url()}/positions"
        body = {
            "epic": config.EPIC,
            "direction": direction,
            "size": size,
            "orderType": "MARKET",
        }
        if stop_loss is not None:
            body["stopLevel"] = stop_loss
        if take_profit is not None:
            body["limitLevel"] = take_profit

        try:
            response = requests.post(
                url,
                headers=self._headers(),
                json=body,
                timeout=config.REQUEST_TIMEOUT,
            )
            response.raise_for_status()
            return response.json()
        except (requests.RequestException, ValueError) as exc:
            logger.error("Capital.com order request failed: %s", exc)
            return None

    def close_position(self, deal_id: str) -> dict | None:
        url = f"{config.get_base_url()}/positions/{deal_id}"
        try:
            response = requests.delete(
                url,
                headers=self._headers(),
                timeout=config.REQUEST_TIMEOUT,
            )
            response.raise_for_status()
            return response.json()
        except (requests.RequestException, ValueError) as exc:
            logger.error("Capital.com close request failed: %s", exc)
            return None