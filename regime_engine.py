# regime_engine.py
# محرك "وضع السوق" + تقسيم البيانات حسب الـ Regime
# هذا الملف يعتمد على وجود الأعمدة:
# EMA_20, EMA_100, ATR, High, Low, Close, Volume (من features.py)

import pandas as pd
import numpy as np


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

    # تأكيد وجود TrendStrength و ATR و Range و NoiseIndex
    if 'TrendStrength' not in df.columns:
        df['TrendStrength'] = df['EMA_20'] - df['EMA_100']

    df['Range'] = df['High'] - df['Low']
    df['Range_Mean'] = df['Range'].rolling(50).mean()
    df['NoiseIndex'] = df['Range'] / (df['Range_Mean'] + 1e-6)

    regimes = []
    for _, row in df.iterrows():
        ts = row['TrendStrength']
        noise = row['NoiseIndex']
        atr = row['ATR']

        # ترند واضح مع ضوضاء منخفضة
        if abs(ts) > atr * 0.5 and noise < 1.2:
            regimes.append(1 if ts > 0 else -1)

        # تذبذب خفيف
        elif noise < 1.5:
            regimes.append(0)

        # فوضى / ضوضاء عالية
        else:
            regimes.append(2)

    df['Regime'] = regimes
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
