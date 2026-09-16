# scheduler.py
"""
وحدة جدولة السوق:
- إيقاف فتح صفقات جديدة قبل إغلاق السوق بساعة
- السماح بمراقبة الصفقات المفتوحة فقط
- إمكانية توسيعها لاحقًا لمنطق فتح السوق
"""

import datetime
from config import MARKET_CLOSE_WEEKDAY, MARKET_CLOSE_HOUR


def time_to_market_close(now: datetime.datetime) -> float:
    """
    حساب عدد الساعات المتبقية حتى إغلاق السوق في اليوم الحالي.
    """
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
    if now.weekday() == MARKET_CLOSE_WEEKDAY:
        hours_left = time_to_market_close(now)
        return hours_left <= 1.0
    return False


def is_market_closed(now: datetime.datetime) -> bool:
    """
    يمكنك لاحقًا توسيع هذه الدالة لتحديد:
    - هل السوق مغلق الآن؟
    - مثلاً من بعد الإغلاق حتى فتح السوق يوم الاثنين.
    الآن نتركها بسيطة.
    """
    return False
