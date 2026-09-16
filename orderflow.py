# orderflow.py
# ميزات تقريبية لتدفق الأوامر (Order Flow) من شكل الشموع فقط

import pandas as pd
import numpy as np


def add_orderflow_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    يضيف ميزات تقريبية لتدفق الأوامر:
    - Body        : حجم جسم الشمعة
    - UpperWick   : الذيل العلوي
    - LowerWick   : الذيل السفلي
    - WickRatio   : نسبة الذيل إلى الجسم
    - AggressiveBuy  : شموع تدل على ضغط شراء
    - AggressiveSell : شموع تدل على ضغط بيع
    """

    df = df.copy()

    # جسم الشمعة
    df['Body'] = (df['Close'] - df['Open']).abs()

    # الأذيال
    df['UpperWick'] = df['High'] - df[['Close', 'Open']].max(axis=1)
    df['LowerWick'] = df[['Close', 'Open']].min(axis=1) - df['Low']

    # نسبة الذيل إلى الجسم
    df['WickRatio'] = (df['UpperWick'] + df['LowerWick']) / (df['Body'] + 1e-6)

    # ضغط شراء/بيع تقريبي
    df['AggressiveBuy'] = (
        (df['Body'] > 0) &
        (df['UpperWick'] < df['LowerWick'])
    ).astype(int)

    df['AggressiveSell'] = (
        (df['Body'] < 0) &
        (df['UpperWick'] > df['LowerWick'])
    ).astype(int)

    return df
