"""Conservative live trading loop with refresh, persistence, and stop guards."""

from __future__ import annotations

import datetime
import logging
import threading

from broker_client import BrokerClient
from config import (
    INITIAL_EQUITY,
    LIVE_CANDLE_LOOKBACK_MINUTES,
    MARKET_TZ,
    MAX_DAILY_DD,
    MAX_OPEN_POSITIONS,
    MAX_TOTAL_DD,
)
from data_loader import merge_candles, normalize_candles
from governance import approve_trade
from risk_engine import RiskEngine
from risk_manager import calculate_position_size
from scheduler import should_stop_new_trades
from strategy import (
    AI_STATE,
    generate_signal,
    get_drift_status,
    init_ai_model,
    refresh_ai_features,
    retrain_ai_model,
)
from trade_store import TradeStore

logger = logging.getLogger(__name__)
STOP_EVENT = threading.Event()
LIVE_STATUS = {"equity": INITIAL_EQUITY, "positions": [], "last_data_at": None}


def request_stop() -> None:
    STOP_EVENT.set()


def _close_all_positions(broker: BrokerClient, positions: list[dict]) -> None:
    for position in positions:
        deal_id = position.get("position", {}).get("dealId") or position.get(
            "dealId"
        )
        if deal_id:
            broker.close_position(str(deal_id))


def initialize_ai_strategy(history_df):
    """Compatibility wrapper for callers that initialize the live strategy."""

    init_ai_model(history_df)


def run_trading_bot(
    history_df=None,
    refresh_fn=None,
    poll_seconds: int = 60,
    stale_after_seconds: int = 90,
    store_path: str = "trading_state.sqlite3",
    equity_fn=None,
    *,
    broker=None,
    poll_interval: float | None = None,
    candle_lookback_minutes: int = LIVE_CANDLE_LOOKBACK_MINUTES,
    max_iterations: int | None = None,
    auto_retrain: bool = True,
    retrain_interval_candles: int = 120,
):
    """Run live continuous trading with autonomous self-retraining and risk governance.

    ``refresh_fn`` is retained for callers that provide their own market-data
    loader. When omitted, the authenticated broker session fetches recent
    candles directly before each decision.
    """

    STOP_EVENT.clear()
    broker = broker or BrokerClient()
    if equity_fn is None:
        if hasattr(broker, "equity"):
            equity_fn = lambda: float(getattr(broker, "equity"))
        else:
            equity_fn = lambda: INITIAL_EQUITY

    if hasattr(broker, "_current_price"):
        # Simulated broker does not experience network stale timeouts
        stale_after_seconds = max(stale_after_seconds, 86_400)

    store = TradeStore(store_path)
    wait_seconds = poll_seconds if poll_interval is None else poll_interval
    risk_engine = RiskEngine(MAX_DAILY_DD, MAX_TOTAL_DD)
    processed_signals: set[str] = set()
    iterations = 0
    candles_since_retrain = 0

    if history_df is not None:
        initialize_ai_strategy(history_df)
        candle_history = normalize_candles(history_df)
    elif AI_STATE.get("candles_df") is not None:
        candle_history = normalize_candles(AI_STATE["candles_df"])
    else:
        raise ValueError("history_df is required before starting live trading")

    if candle_history.empty:
        raise ValueError("history_df must contain at least one candle")

    last_candle_time = candle_history.index.max()
    last_data_at = datetime.datetime.now(datetime.UTC)

    while not STOP_EVENT.is_set():
        if max_iterations is not None and iterations >= max_iterations:
            break
        iterations += 1

        now = datetime.datetime.now(datetime.UTC)
        equity = float(equity_fn())
        risk_engine.update(equity)
        trading_date = now.astimezone(MARKET_TZ).date().isoformat()
        equity_today_start = store.get_or_create_day_start(trading_date, equity)
        has_new_candle = False

        try:
            latest_candles = (
                broker.get_recent_candles(lookback_minutes=candle_lookback_minutes)
                if refresh_fn is None
                else refresh_fn()
            )
            if latest_candles is not None and not latest_candles.empty:
                merged_candles = merge_candles(candle_history, latest_candles)
                newest_candle_time = merged_candles.index.max()
                if newest_candle_time > last_candle_time:
                    candle_history = merged_candles
                    refresh_ai_features(candle_history)
                    last_candle_time = newest_candle_time
                    last_data_at = now
                    has_new_candle = True
                    candles_since_retrain += 1
        except (RuntimeError, TypeError, ValueError, OSError) as exc:
            logger.error("Live data refresh failed: %s", exc)

        positions = broker.get_open_positions()
        if positions is None:
            logger.critical("Open positions are unknown; refusing to place orders.")
            STOP_EVENT.wait(wait_seconds)
            continue

        LIVE_STATUS.update(
            equity=equity,
            positions=positions,
            last_data_at=last_data_at.isoformat(),
        )

        if (now - last_data_at).total_seconds() > stale_after_seconds:
            logger.critical("Market data is stale; closing positions and stopping.")
            _close_all_positions(broker, positions)
            break

        if not has_new_candle:
            logger.warning("No newer market candle; skipping this decision cycle.")
            STOP_EVENT.wait(wait_seconds)
            continue

        drift = get_drift_status()
        severe_drift = {name: value for name, value in drift.items() if value > 0.40}
        if severe_drift and auto_retrain:
            logger.warning(
                "⚠️ [MARKET DRIFT DETECTED] Drift in %s. Launching autonomous self-retraining on rolling window...",
                list(severe_drift.keys())[:3],
            )
            try:
                retrain_ai_model(candle_history)
                candles_since_retrain = 0
            except Exception as exc:
                logger.error("Autonomous retraining failed on drift: %s", exc)
        elif severe_drift:
            logger.critical("Severe feature drift detected: %s", severe_drift)
            STOP_EVENT.wait(wait_seconds)
            continue

        if auto_retrain and candles_since_retrain >= retrain_interval_candles:
            logger.info(
                "⏰ [SCHEDULED RETRAINING] Reached %d new candles. Autonomously updating AI models...",
                candles_since_retrain,
            )
            try:
                retrain_ai_model(candle_history)
                candles_since_retrain = 0
            except Exception as exc:
                logger.error("Periodic autonomous retraining failed: %s", exc)

        if should_stop_new_trades(now):
            logger.info("Market schedule blocks new trades.")
            STOP_EVENT.wait(wait_seconds)
            continue
        if not risk_engine.can_trade(equity_today_start, equity):
            logger.critical("Risk limits reached; no new trades will be opened.")
            STOP_EVENT.wait(wait_seconds)
            continue

        signal, meta = generate_signal()
        store.record_decision(signal, meta)
        curr_price = float(candle_history["Close"].iloc[-1])
        logger.info(
            "📊 [CYCLE #%d] Price: %.5f | Equity: $%.2f | Positions: %d | Decision: %s",
            iterations,
            curr_price,
            equity,
            len(positions),
            "Watching" if not signal else f"⚡ {signal} SIGNAL",
        )
        risk_allowed = risk_engine.can_trade(equity_today_start, equity)
        environment_safe = (
            has_new_candle
            and (now - last_data_at).total_seconds() <= stale_after_seconds
        )
        approved, approval_reason = approve_trade(
            signal_present=bool(signal),
            risk_allowed=risk_allowed,
            environment_safe=environment_safe,
        )
        if not approved:
            logger.info("Trade decision rejected: %s", approval_reason)
            STOP_EVENT.wait(wait_seconds)
            continue

        signal_key = str(meta.get("signal_time"))
        if signal_key in processed_signals:
            logger.warning("Duplicate signal ignored: %s", signal_key)
            STOP_EVENT.wait(wait_seconds)
            continue
        processed_signals.add(signal_key)

        if len(positions) >= MAX_OPEN_POSITIONS:
            logger.warning("Open-position limit reached; signal ignored.")
        else:
            stop_pips = meta.get("stop_pips", 20)
            pip_value = meta.get("pip_value", 0.0001)
            size = calculate_position_size(equity, stop_pips, pip_value)
            if size > 0:
                entry = meta["entry_price"]
                stop_distance = meta.get("stop_distance", stop_pips * pip_value)
                take_profit_distance = meta.get(
                    "take_profit_distance", stop_distance * 1.5
                )
                if signal == "BUY":
                    stop_loss = entry - stop_distance
                    take_profit = entry + take_profit_distance
                else:
                    stop_loss = entry + stop_distance
                    take_profit = entry - take_profit_distance
                response = broker.open_market_order(
                    signal,
                    size,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                )
                store.record_order(signal, size, response)

        STOP_EVENT.wait(wait_seconds)

    logger.info("Trading loop stopped.")


def get_live_status() -> dict:
    return dict(LIVE_STATUS)