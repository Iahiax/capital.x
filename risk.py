# risk.py

import pandas as pd
from config import INITIAL_EQUITY, RISK_PER_TRADE

def apply_risk_management(signals_df):
    trades = []

    for _, s in signals_df.iterrows():
        atr = s['ATR']
        sl = atr * 0.7
        tp = atr * 1.5

        risk_amount = INITIAL_EQUITY * RISK_PER_TRADE
        position_size = risk_amount / sl if sl > 0 else 0

        trades.append({
            'Time': s['Time'],
            'Type': s['Type'],
            'Entry': s['Price'],
            'SL': sl,
            'TP': tp,
            'Size': position_size
        })

    return pd.DataFrame(trades)
