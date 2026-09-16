# signals.py

import pandas as pd

def generate_signals(df):
    signals = []

    for idx, row in df.iterrows():
        hour = row['Hour']
        regime = row['Regime']

        # فلترة وقتية
        if not (8 <= hour <= 11 or 14 <= hour <= 17):
            continue

        # تجنب الفوضى
        if regime == 2:
            continue

        # LONG فقط في ترند صاعد
        if regime == 1:
            if (
                row['AI_Prob'] > 0.80 and
                row['TrendStrength'] > 0 and
                row['Kalman_Fast_Slope'] > 0 and
                row['BuyPressure'] > 0.6 and
                row['RVOL'] > 1.2 and
                row['ShockIndex'] < 1.5 and
                row['SmartDiv'] > 0 and
                row['NoiseIndex'] < 1.5
            ):
                signals.append({
                    'Time': idx,
                    'Type': 'LONG',
                    'Price': row['Close'],
                    'ATR': row['ATR']
                })

        # SHORT فقط في ترند هابط
        if regime == -1:
            if (
                row['AI_Prob'] < 0.20 and
                row['TrendStrength'] < 0 and
                row['Kalman_Fast_Slope'] < 0 and
                row['SellPressure'] > 0.6 and
                row['RVOL'] > 1.2 and
                row['ShockIndex'] < 1.5 and
                row['SmartDiv'] < 0 and
                row['NoiseIndex'] < 1.5
            ):
                signals.append({
                    'Time': idx,
                    'Type': 'SHORT',
                    'Price': row['Close'],
                    'ATR': row['ATR']
                })

        # في التذبذب (Regime = 0) يمكن تقليل الإشارات أو تجاهلها
        # هنا مثلاً نتجاهلها تماماً:
        # if regime == 0: continue

    return pd.DataFrame(signals)
