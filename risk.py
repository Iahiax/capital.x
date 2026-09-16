# risk.py

import pandas as pd
from config import INITIAL_EQUITY, RISK_PER_TRADE

def apply_risk_management(signals_df):
    trades = []

    for _, s in signals_df.iterrows():
        atr = s['ATR']
        sl = atr * 0.7
        tp = atr * 1.5

        base_risk = INITIAL_EQUITY * RISK_PER_TRADE
        quality_factor = min(max(s['Score'] / 100.0, 0.5), 1.5)
        risk_amount = base_risk * quality_factor

        position_size = risk_amount / sl if sl > 0 else 0

        trades.append({
            'Time': s['Time'],
            'Type': s['Type'],
            'Entry': s['Price'],
            'SL': sl,
            'TP': tp,
            'Size': position_size,
            'Score': s['Score']
        })

    return pd.DataFrame(trades)
