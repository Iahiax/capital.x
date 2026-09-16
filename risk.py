# risk.py

def apply_risk_management(signals):
    trades = []

    for _, s in signals.iterrows():
        sl = s['ATR'] * 0.7
        tp = s['ATR'] * 1.5

        trades.append({
            'Time': s['Time'],
            'Type': s['Type'],
            'Entry': s['Price'],
            'SL': sl,
            'TP': tp
        })

    return trades
