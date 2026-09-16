# risk.py

import pandas as pd
from config import INITIAL_EQUITY, RISK_PER_TRADE

def apply_risk_management(signals_df):
    trades = []

    for _, s in signals_df.iterrows():
        atr = s['ATR']
        score = s['Score']

        sl = atr * (1.0 - min(score / 200.0, 0.5))      # كلما زاد Score قل SL نسبياً
        tp = atr * (1.0 + min(score / 150.0, 1.0))      # كلما زاد Score زاد TP نسبياً

        base_risk = INITIAL_EQUITY * RISK_PER_TRADE
        quality_factor = min(max(score / 100.0, 0.5), 1.5)
        risk_amount = base_risk * quality_factor

        position_size = risk_amount / sl if sl > 0 else 0

        trades.append({
            'Time': s['Time'],
            'Type': s['Type'],
            'Entry': s['Price'],
            'SL': sl,
            'TP': tp,
            'Size': position_size,
            'Score': score
        })

    return pd.DataFrame(trades)
