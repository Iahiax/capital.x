"""Vectorized market-regime classification."""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_regime(df: pd.DataFrame) -> pd.DataFrame:
    """
    يحسب وضع السوق (Regime) لكل شمعة:
    1  = ترند صاعد قوي
    -1 = ترند هابط قوي
    0  = تذبذب (Range)
    2  = فوضى / ضوضاء عالية (تجنب التداول)

    يعتمد على:
    - TrendStrength = EMA_20 - EMA_100
    - ATR
    - NoiseIndex (مبني على Range)
    """

    df = df.copy()
    if "TrendStrength" not in df.columns:
        df["TrendStrength"] = df["EMA_20"] - df["EMA_100"]

    df["Range"] = df["High"] - df["Low"]
    df["Range_Mean"] = df["Range"].rolling(50).mean()
    df["NoiseIndex"] = df["Range"] / (df["Range_Mean"] + 1e-6)
    trend_signal = df["TrendStrength"] / (df["ATR"] + 1e-6)
    trend_up = 1.0 / (1.0 + np.exp(-trend_signal))
    trend_down = 1.0 - trend_up
    range_probability = np.exp(-np.abs(trend_signal)) * np.clip(
        2.0 - df["NoiseIndex"], 0.0, 1.0
    )
    probability_total = trend_up + trend_down + range_probability
    df["RegimeProbUp"] = trend_up / probability_total
    df["RegimeProbDown"] = trend_down / probability_total
    df["RegimeProbRange"] = range_probability / probability_total

    atr_rank = df["ATR"].rolling(200, min_periods=20).rank(pct=True)
    df["VolatilityRegime"] = np.select(
        [
            atr_rank.le(0.33),
            atr_rank.le(0.66),
            atr_rank.le(0.90),
        ],
        [0, 1, 2],
        default=3,
    ).astype(int)

    trend = df["TrendStrength"].to_numpy()
    noise = df["NoiseIndex"].to_numpy()
    atr = df["ATR"].to_numpy()
    clear_trend = (np.abs(trend) > atr * 0.5) & (noise < 1.2)
    range_market = ~clear_trend & (noise < 1.5)
    df["Regime"] = np.select(
        [clear_trend & (trend > 0), clear_trend & (trend < 0), range_market],
        [1, -1, 0],
        default=2,
    ).astype(int)
    return df


def split_by_regime(df: pd.DataFrame) -> dict:
    """
    يقسم البيانات إلى أربعة أجزاء حسب الـ Regime:
    - uptrend  : ترند صاعد
    - downtrend: ترند هابط
    - range    : تذبذب
    - chaos    : فوضى (غالباً لا نتداول فيها)

    يستخدم لاحقاً في:
    - تدريب نماذج خاصة لكل وضع سوق (Regime‑Specific Models)
    """

    if 'Regime' not in df.columns:
        df = compute_regime(df)

    return {
        'uptrend': df[df['Regime'] == 1],
        'downtrend': df[df['Regime'] == -1],
        'range': df[df['Regime'] == 0],
        'chaos': df[df['Regime'] == 2],
    }


def add_regime_to_features(df_features: pd.DataFrame) -> pd.DataFrame:
    """
    دالة مساعدة:
    تُستخدم داخل features.py بعد بناء الميزات،
    لإضافة عمود Regime إلى DataFrame الميزات.
    """

    return compute_regime(df_features)
