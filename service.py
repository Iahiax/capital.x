"""Long-running trading service entrypoint with safe defaults."""

from __future__ import annotations

import logging
import signal
import threading

import config
from config import WALK_FORWARD_TEST_DAYS, WALK_FORWARD_TRAIN_DAYS
from data_loader import generate_sample_data, load_full_year_data
from readiness import validate_live_readiness
from trade_engine import request_stop, run_trading_bot
from walk_forward import validate_strategy_walk_forward

logger = logging.getLogger(__name__)
SERVICE_STOP = threading.Event()


def _install_signal_handlers() -> None:
    def stop_handler(signum, _frame):
        logger.info("Received signal %s; stopping trading service.", signum)
        SERVICE_STOP.set()
        request_stop()

    signal.signal(signal.SIGTERM, stop_handler)
    signal.signal(signal.SIGINT, stop_handler)


def run_sample_service(poll_seconds: int = 30) -> None:
    """Keep a local, network-free process alive for workflow verification."""

    _install_signal_handlers()
    sample = generate_sample_data(1_000)
    logger.info("Sample service started with %d local candles.", len(sample))
    while not SERVICE_STOP.is_set():
        logger.info("Sample service heartbeat; no broker orders are possible.")
        SERVICE_STOP.wait(poll_seconds)
    logger.info("Sample service stopped.")


def run_continuous_service(mode: str = "DEMO", sample: bool = False) -> None:
    """Run the broker-backed loop continuously, or a safe local service."""

    if sample:
        run_sample_service()
        return

    _install_signal_handlers()
    mode = mode.upper()
    if mode not in {"DEMO", "LIVE"}:
        raise ValueError("service mode must be DEMO or LIVE")
    config.USE_DEMO = mode == "DEMO"

    history = load_full_year_data()
    validation = validate_strategy_walk_forward(
        history,
        train_window=f"{WALK_FORWARD_TRAIN_DAYS}D",
        test_window=f"{WALK_FORWARD_TEST_DAYS}D",
    )
    logger.info("Continuous-service walk-forward result: %s", validation)
    if mode == "LIVE":
        validate_live_readiness(validation)

    try:
        run_trading_bot(history_df=history)
    finally:
        request_stop()