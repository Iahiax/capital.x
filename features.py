# features.py

import pandas as pd
import numpy as np
from orderflow import add_orderflow_features
from regime_engine import compute_regime


def kalman_filter(series, process_variance=1e-5, measurement_variance=1e-2):
    n = len(series)
    xhat = np.zeros(n)
    P = np.zeros(n)
    xhat[0] = series.iloc[0]
    P[0] = 1.0
    Q = process_variance
    R = measurement_variance
    for k in range(1, n):
        xhat_minus = xhat[k - 1]
        P_minus = P[k - 1] + Q
        K = P_minus / (P_minus + R)
        xhat[k] = xhat_minus + K * (series.iloc[k] - xhat_minus)
        P[k] = (1 - K) * P_minus
    return pd.Series(xhat, index=series.index)


def add_market_quality(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['TrendStrength'] = df['EMA_20'] - df['EMA_100']
    df['Range'] = df['High'] - df['Low']
    df['Range_Mean'] = df['Range'].rolling(50).mean()
    df['NoiseIndex'] = df['Range'] / (df['Range_Mean'] + 1e-6)

    df['MarketQuality'] = \
        (df['TrendStrength'] / (df['ATR'] + 1e-6)) * 0.4 + \
        df['RVOL'] * 0.3 - \
        df['NoiseIndex'] * 0.3

    return df


def add_multi_timeframe_features(df: pd.DataFrame) -> pd.DataFrame:
    df_5m = df['Close'].resample('5T').ohlc()
    df_15m = df['Close'].resample('15T').ohlc()

    df_5m['EMA_5m_20'] = df_5m['close'].ewm(span=20).mean()
    df_15m['EMA_15m_20'] = df_15m['close'].ewm(span=20).mean()

    df_5m = df_5m[['EMA_5m_20']]
    df_15m = df_15m[['EMA_15m_20']]

    df = pd.merge_asof(
        df.sort_index(), df_5m.sort_index(),
        left_index=True, right_index=True, direction='backward'
    )
    df = pd.merge_asof(
        df.sort_index(), df_15m.sort_index(),
        left_index=True, right_index=True, direction='backward'
    )

    return df


def create_pro_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df['EMA_20'] = df['Close'].ewm(span=20).mean()
    df['EMA_100'] = df['Close'].ewm(span=100).mean()
    df['EMA_300'] = df['Close'].ewm(span=300).mean()

    exp1 = df['Close'].ewm(span=12).mean()
    exp2 = df['Close'].ewm(span=26).mean()
    df['MACD'] = exp1 - exp2
    df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()

    df['ATR'] = (df['High'] - df['Low']).ewm(span=14).mean()

    df['Kalman_Fast'] = kalman_filter(df['Close'], 1e-4, 1e-2)
    df['Kalman_Fast_Slope'] = df['Kalman_Fast'].diff()

    df['BuyPressure'] = (df['Close'] - df['Low']) / (df['High'] - df['Low'] + 1e-6)
    df['SellPressure'] = (df['High'] - df['Close']) / (df['High'] - df['Low'] + 1e-6)

    df['RVOL'] = df['Volume'] / (df['Volume'].rolling(20).mean() + 1e-6)

    df['ShockIndex'] = np.abs(df['Close'].diff()) / (df['ATR'] + 1e-6)

    df['SmartDiv'] = (df['MACD'] - df['MACD_Signal']) * (df['EMA_20'] - df['EMA_100'])

    df['Compression'] = (df['High'] - df['Low']).rolling(10).mean()

    df['Hour'] = df.index.hour
    df['Minute'] = df.index.minute

    df['Target_3m'] = (df['Close'].shift(-3) > df['Close']).astype(int)

    df = add_orderflow_features(df)
    df = add_multi_timeframe_features(df)
    df = compute_regime(df)
    df = add_market_quality(df)

    df.dropna(inplace=True)

    return df


def get_feature_columns(df: pd.DataFrame):
    return [c for c in df.columns if c not in ['Target_3m']]
