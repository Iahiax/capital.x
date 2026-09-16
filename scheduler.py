# scheduler.py

import datetime
from config import MARKET_CLOSE_WEEKDAY, MARKET_CLOSE_HOUR

def time_to_market_close(now):
    close_time = now.replace(
        hour=MARKET_CLOSE_HOUR,
        minute=0,
        second=0,
        microsecond=0
    )
    delta = close_time - now
    return delta.total_seconds() / 3600

def should_stop_new_trades(now):
    if now.weekday() == MARKET_CLOSE_WEEKDAY:
        hours_left = time_to_market_close(now)
        return hours_left <= 1.0
    return False
