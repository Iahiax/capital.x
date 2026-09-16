# risk_manager.py
"""
وحدة إدارة المخاطر:
- حساب حجم الصفقة بناءً على:
    - رأس المال الحالي
    - نسبة المخاطرة لكل صفقة
    - عدد نقاط وقف الخسارة
    - قيمة النقطة
- احترام الرافعة المالية 100:1
"""

from config import INITIAL_EQUITY, RISK_PER_TRADE, LEVERAGE, SPREAD, COMMISSION_PER_TRADE


def calculate_position_size(
    equity: float,
    stop_pips: float,
    pip_value: float = 0.0001,
) -> float:
    """
    حساب حجم الصفقة (وحدات) بحيث:
    - إذا ضرب الستوب نخسر فقط نسبة RISK_PER_TRADE من رأس المال
    - لا نتجاوز القيمة الاسمية القصوى حسب الرافعة

    equity: رأس المال الحالي
    stop_pips: عدد نقاط وقف الخسارة
    pip_value: قيمة النقطة الواحدة
    """

    if stop_pips <= 0:
        return 0.0

    # المبلغ المعرض للخطر في هذه الصفقة
    base_risk_amount = equity * RISK_PER_TRADE

    # تعديل بسيط لأخذ السبريد والعمولة في الاعتبار
    effective_risk_amount = base_risk_amount - COMMISSION_PER_TRADE
    if effective_risk_amount <= 0:
        effective_risk_amount = base_risk_amount

    # عدد الوحدات بحيث إذا ضرب الستوب نخسر effective_risk_amount
    # تقريبًا: الخسارة ≈ stop_pips * pip_value * units
    units = effective_risk_amount / (stop_pips * pip_value)

    # القيمة الاسمية التقريبية
    notional = units * pip_value * 100000  # تقريب تقريبي

    # الحد الأقصى للقيمة الاسمية حسب الرافعة
    max_notional = equity * LEVERAGE

    if notional > max_notional:
        scale = max_notional / notional
        units *= scale

    return max(units, 0.0)


def update_equity_after_trade(
    equity: float,
    profit_loss: float,
) -> float:
    """
    تحديث رأس المال بعد صفقة:
    profit_loss: الربح أو الخسارة (بالدولار)
    """
    new_equity = equity + profit_loss
    return max(new_equity, 0.0)
