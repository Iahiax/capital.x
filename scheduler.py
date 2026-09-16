"""Trading schedule guards for the weekly forex market."""

import datetime

from config import (
    MARKET_CLOSE_HOUR,
    MARKET_CLOSE_WEEKDAY,
    MARKET_OPEN_HOUR,
    MARKET_OPEN_WEEKDAY,
    MARKET_TZ,
)


def _market_time(now: datetime.datetime) -> datetime.datetime:
    if now.tzinfo is None:
        now = now.replace(tzinfo=datetime.UTC)
    return now.astimezone(MARKET_TZ)


def time_to_market_close(now: datetime.datetime) -> float:
    """
    حساب عدد الساعات المتبقية حتى إغلاق السوق في اليوم الحالي.
    """
    now = _market_time(now)
    close_time = now.replace(
        hour=MARKET_CLOSE_HOUR,
        minute=0,
        second=0,
        microsecond=0,
    )
    delta = close_time - now
    return delta.total_seconds() / 3600.0


def should_stop_new_trades(now: datetime.datetime) -> bool:
    """
    هل يجب إيقاف فتح صفقات جديدة؟
    - إذا كان اليوم هو يوم الإغلاق (مثلاً الجمعة)
    - وإذا بقي أقل من ساعة على الإغلاق
    """
    now = _market_time(now)
    if is_market_closed(now):
        return True
    if now.weekday() == MARKET_CLOSE_WEEKDAY:
        hours_left = time_to_market_close(now)
        return hours_left <= 1.0
    return False


def is_market_closed(now: datetime.datetime) -> bool:
    now = _market_time(now)
    weekday = now.weekday()
    if weekday in {5, 6}:
        return True
    return (
        weekday == MARKET_CLOSE_WEEKDAY and now.hour >= MARKET_CLOSE_HOUR
    ) or (weekday == MARKET_OPEN_WEEKDAY and now.hour < MARKET_OPEN_HOUR)
