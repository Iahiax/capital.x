# regime.py

import pandas as pd
import numpy as np

def compute_regime(df):
    df = df.copy()

    # ترند بسيط مبني على EMA_20 و EMA_100
    df['TrendStrength'] = df['EMA_20'] - df['EMA_100']

    # قياس تذبذب السعر
    df['Range'] = df['High'] - df['Low']
    df['Range_Mean'] = df['Range'].rolling(50).mean()
    df['NoiseIndex'] = df['Range'] / (df['Range_Mean'] + 1e-6)

    regimes = []
    for i, row in df.iterrows():
        ts = row['TrendStrength']
        noise = row['NoiseIndex']

        if abs(ts) > row['ATR'] * 0.5 and noise < 1.2:
            # ترند واضح
            regimes.append(1 if ts > 0 else -1)
        elif noise < 1.5:
            # تذبذب خفيف
            regimes.append(0)
        else:
            # فوضى / ضوضاء عالية
            regimes.append(2)  # 2 = تجنب التداول
    df['Regime'] = regimes

    return df
