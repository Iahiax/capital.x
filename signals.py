# signals.py

import pandas as pd

def generate_signals(df):
    signals = []

    for idx, row in df.iterrows():
        hour = idx.hour
        if not (8 <= hour <= 11 or 14 <= hour <= 17):
            continue

        score = (
            row['AI_Prob'] * 50 +
            row['TrendStrength'] * 10 +
            row['BuyPressure'] * 10 +
            row['RVOL'] * 10 -
            row['NoiseIndex'] * 10 -
            row['ShockIndex'] * 10
        )

        if score > 60:
            if row['AI_Prob'] > 0.80 and row['TrendStrength'] > 0:
                signals.append({'Time': idx, 'Type': 'LONG', 'Price': row['Close'], 'ATR': row['ATR']})

            if row['AI_Prob'] < 0.20 and row['TrendStrength'] < 0:
                signals.append({'Time': idx, 'Type': 'SHORT', 'Price': row['Close'], 'ATR': row['ATR']})

    return pd.DataFrame(signals)
