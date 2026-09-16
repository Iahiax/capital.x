# signals.py

import pandas as pd

def compute_signal_score(row):
    score = 0
    score += row['AI_Prob'] * 50
    score += (row['TrendStrength'] / (row['ATR'] + 1e-6)) * 10
    score += row['BuyPressure'] * 10
    score += row['RVOL'] * 10
    score -= row['ShockIndex'] * 10
    score -= row['NoiseIndex'] * 10
    return score

def generate_signals(df, news_blackout=None):
    signals = []

    for idx, row in df.iterrows():
        hour = row['Hour']
        regime = row['Regime']
        mq = row['MarketQuality']

        if not (8 <= hour <= 11 or 14 <= hour <= 17):
            continue

        if news_blackout is not None and news_blackout.loc[idx]:
            continue

        if mq < 0.0:
            continue

        score = compute_signal_score(row)
        if score < 60:
            continue

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
                    'ATR': row['ATR'],
                    'Score': score
                })

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
                    'ATR': row['ATR'],
                    'Score': score
                })

    return pd.DataFrame(signals)
