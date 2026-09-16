# signals.py

import pandas as pd
from meta_model import MetaDecisionModel


def compute_signal_score(row):
    score = 0
    score += row['AI_Prob'] * 50
    score += (row['TrendStrength'] / (row['ATR'] + 1e-6)) * 10
    score += row['BuyPressure'] * 10
    score += row['RVOL'] * 10
    score -= row['ShockIndex'] * 10
    score -= row['NoiseIndex'] * 10
    score += row['AggressiveBuy'] * 5
    score -= row['AggressiveSell'] * 5
    return score


def dynamic_thresholds(df):
    vol = df['ATR'].rolling(200).mean().iloc[-1]
    base_ai_long = 0.8
    base_ai_short = 0.2
    avg_vol = df['ATR'].mean()

    if vol > avg_vol:
        base_ai_long += 0.05
        base_ai_short -= 0.05
    else:
        base_ai_long -= 0.05
        base_ai_short += 0.05

    return base_ai_long, base_ai_short


def generate_signals(df_feat: pd.DataFrame, news_blackout=None, meta_model: MetaDecisionModel = None):
    signals = []

    ai_long_thr, ai_short_thr = dynamic_thresholds(df_feat)

    for idx, row in df_feat.iterrows():
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

        meta_prob = 1.0
        if meta_model is not None:
            meta_prob = meta_model.predict_prob(
                row['AI_Prob'],
                score,
                regime,
                mq
            )
            if meta_prob < 0.65:
                continue

        if regime == 1:
            if (
                row['AI_Prob'] > ai_long_thr and
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
                    'Score': score,
                    'MetaProb': meta_prob
                })

        if regime == -1:
            if (
                row['AI_Prob'] < ai_short_thr and
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
                    'Score': score,
                    'MetaProb': meta_prob
                })

    return pd.DataFrame(signals)
