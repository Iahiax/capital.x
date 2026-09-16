# risk_manager.py

from config import INITIAL_EQUITY, RISK_PER_TRADE, LEVERAGE

def calculate_position_size(equity, stop_pips, pip_value):
    risk_amount = equity * RISK_PER_TRADE
    units = risk_amount / (stop_pips * pip_value)

    notional = units * pip_value * 100000  # مثال
    max_notional = equity * LEVERAGE

    if notional > max_notional:
        scale = max_notional / notional
        units *= scale

    return units
