# signals.py

import pandas as pd

def generate_signals(df):
    signals = []

    for idx, row in df.iterrows():
        hour = row['Hour']

        if not (8 <= hour <= 11 or 14 <= hour <= 17):
            continue

        if (
            row['AI_Prob'] > 0.80 and
            row['TrendStrength'] > 0 and
            row['Kalman_Fast_Slope'] > 0 and
            row['BuyPressure'] > 0.6 and
            row['RVOL'] > 1.2 and
            row['ShockIndex'] < 1.5 and
            row['SmartDiv'] > 0
        ):
            signals.append({
                'Time': idx,
                'Type': 'LONG',
                'Price': row['Close'],
                'ATR': row['ATR']
            })

        if (
            row['AI_Prob'] < 0.20 and
            row['TrendStrength'] < 0 and
            row['Kalman_Fast_Slope'] < 0 and
            row['SellPressure'] > 0.6 and
            row['RVOL'] > 1.2 and
            row['ShockIndex'] < 1.5 and
            row['SmartDiv'] < 0
        ):
            signals.append({
                'Time': idx,
                'Type': 'SHORT',
                'Price': row['Close'],
                'ATR': row['ATR']
            })

    return pd.DataFrame(signals)
