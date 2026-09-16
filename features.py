# features.py

import pandas as pd
import numpy as np

def kalman_filter(series, process_variance=1e-5, measurement_variance=1e-2):
    n = len(series)
    xhat = np.zeros(n)
    P = np.zeros(n)
    xhat[0] = series.iloc[0]
    P[0] = 1.0
    Q = process_variance
    R = measurement_variance
    for k in range(1, n):
        xhat_minus = xhat[k-1]
        P_minus = P[k-1] + Q
        K = P_minus / (P_minus + R)
        xhat[k] = xhat_minus + K * (series.iloc[k] - xhat_minus)
        P[k] = (1 - K) * P_minus
    return pd.Series(xhat, index=series.index)

def create_pro_features(df):
    df = df.copy()

    df['EMA_20']  = df['Close'].ewm(span=20).mean()
    df['EMA_100'] = df['Close'].ewm(span=100).mean()
    df['EMA_300'] = df['Close'].ewm(span=300).mean()

    exp1 = df['Close'].ewm(span=12).mean()
    exp2 = df['Close'].ewm(span=26).mean()
    df['MACD'] = exp1 - exp2
    df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()

    df['ATR'] = (df['High'] - df['Low']).ewm(span=14).mean()

    df['TrendStrength'] = df['EMA_20'] - df['EMA_100']

    df['Kalman_Fast'] = kalman_filter(df['Close'], 1e-4, 1e-2)
    df['Kalman_Fast_Slope'] = df['Kalman_Fast'].diff()

    df['BuyPressure']  = (df['Close'] - df['Low']) / (df['High'] - df['Low'] + 1e-6)
    df['SellPressure'] = (df['High'] - df['Close']) / (df['High'] - df['Low'] + 1e-6)

    df['RVOL'] = df['Volume'] / (df['Volume'].rolling(20).mean() + 1e-6)

    df['ShockIndex'] = np.abs(df['Close'].diff()) / (df['ATR'] + 1e-6)

    df['SmartDiv'] = (df['MACD'] - df['MACD_Signal']) * df['TrendStrength']

    df['Compression'] = (df['High'] - df['Low']).rolling(10).mean()

    df['Hour'] = df.index.hour
    df['Minute'] = df.index.minute

    df['Target_3m'] = (df['Close'].shift(-3) > df['Close']).astype(int)

    df.dropna(inplace=True)
    return df

def get_feature_columns(df):
    return [c for c in df.columns if c not in ['Target_3m']]
