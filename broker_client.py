"""Small Capital.com trading client."""

from __future__ import annotations

import datetime
import logging

import numpy as np
import pandas as pd
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


class SimulatedLiveBroker:
    """High-fidelity simulated broker for continuous autonomous paper-trading."""

    def __init__(self, initial_equity: float = config.INITIAL_EQUITY):
        from data_loader import generate_sample_data

        self.equity = float(initial_equity)
        self.balance = float(initial_equity)
        self.positions: list[dict] = []
        self._deal_counter = 0
        self._candles = generate_sample_data(periods=1_200)
        self._current_price = float(self._candles["Close"].iloc[-1])

    def get_open_positions(self) -> list[dict]:
        return list(self.positions)

    def get_recent_candles(self, lookback_minutes: int | None = 15):
        last_time = self._candles.index[-1]
        next_time = last_time + pd.Timedelta(minutes=1)
        rng = np.random.default_rng()
        change = float(rng.normal(0, 0.00012))
        new_close = max(round(float(self._candles["Close"].iloc[-1]) + change, 5), 0.5)
        new_open = float(self._candles["Close"].iloc[-1])
        spread = 0.0001
        new_high = max(new_open, new_close) + spread
        new_low = min(new_open, new_close) - spread
        new_vol = float(rng.lognormal(mean=5.0, sigma=0.3))

        new_candle = pd.DataFrame(
            {
                "Open": [new_open],
                "High": [new_high],
                "Low": [new_low],
                "Close": [new_close],
                "Volume": [new_vol],
            },
            index=pd.DatetimeIndex([next_time]),
        )
        self._candles = pd.concat([self._candles, new_candle]).iloc[-2000:]
        self._current_price = new_close

        self._check_positions(new_high, new_low)
        return self._candles.tail(lookback_minutes or 15)

    def _check_positions(self, high: float, low: float):
        remaining = []
        for pos in self.positions:
            deal_id = pos["dealId"]
            direction = pos["direction"]
            entry = pos["openPrice"]
            sl = pos.get("stopLoss")
            tp = pos.get("takeProfit")
            size = pos["size"]

            closed = False
            pnl = 0.0
            if direction == "BUY":
                if sl is not None and low <= sl:
                    pnl = (sl - entry) * size * 100000.0
                    closed = True
                    logger.info("🛑 [PAPER TRADE] Deal #%s hit SL @ %.5f | PnL: $%.2f", deal_id, sl, pnl)
                elif tp is not None and high >= tp:
                    pnl = (tp - entry) * size * 100000.0
                    closed = True
                    logger.info("🎯 [PAPER TRADE] Deal #%s hit TP @ %.5f | PnL: $%.2f", deal_id, tp, pnl)
            else:
                if sl is not None and high >= sl:
                    pnl = (entry - sl) * size * 100000.0
                    closed = True
                    logger.info("🛑 [PAPER TRADE] Deal #%s hit SL @ %.5f | PnL: $%.2f", deal_id, sl, pnl)
                elif tp is not None and low <= tp:
                    pnl = (entry - tp) * size * 100000.0
                    closed = True
                    logger.info("🎯 [PAPER TRADE] Deal #%s hit TP @ %.5f | PnL: $%.2f", deal_id, tp, pnl)

            if closed:
                self.balance += pnl
                self.equity = self.balance
            else:
                remaining.append(pos)
        self.positions = remaining

    def open_market_order(
        self,
        direction: str,
        size: float,
        stop_loss: float | None = None,
        take_profit: float | None = None,
    ) -> dict:
        self._deal_counter += 1
        deal_id = f"SIM_{self._deal_counter}"
        entry = self._current_price
        pos = {
            "dealId": deal_id,
            "direction": direction,
            "size": size,
            "openPrice": entry,
            "stopLoss": stop_loss,
            "takeProfit": take_profit,
            "time": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        }
        self.positions.append(pos)
        logger.info(
            "⚡ [PAPER ORDER] %s Deal #%s: Size=%.2f @ %.5f | SL=%.5f | TP=%.5f",
            direction, deal_id, size, entry, stop_loss or 0.0, take_profit or 0.0,
        )
        return {"dealId": deal_id, "status": "OPEN"}

    def close_position(self, deal_id: str) -> dict:
        self.positions = [p for p in self.positions if p.get("dealId") != deal_id]
        return {"dealId": deal_id, "status": "CLOSED"}