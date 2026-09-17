import numpy as np
import pandas as pd

from config import SPREAD
from data_quality import sanitize_candles
from orderflow import add_orderflow_features
from regime_engine import compute_regime


def kalman_filter(series, process_variance=1e-5, measurement_variance=1e-2):
    if series.empty:
        return pd.Series(dtype=float, index=series.index)

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
    df = sanitize_candles(df)
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
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("Feature data must use a DatetimeIndex")

    df_5m = df['Close'].resample('5min').ohlc()
    df_15m = df['Close'].resample('15min').ohlc()

    df_5m['EMA_5m_20'] = df_5m['close'].ewm(span=20).mean()
    df_15m['EMA_15m_20'] = df_15m['close'].ewm(span=20).mean()

    df_5m = df_5m[['EMA_5m_20']].shift(1)
    df_15m = df_15m[['EMA_15m_20']].shift(1)

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
    required = {"Open", "High", "Low", "Close", "Volume"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing candle columns: {sorted(missing)}")

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
    df["Range"] = df["High"] - df["Low"]
    df["Range_Mean"] = df["Range"].rolling(50).mean()
    df["NoiseIndex"] = df["Range"] / (df["Range_Mean"] + 1e-6)

    returns = df["Close"].pct_change()
    direction_changes = returns.fillna(0).ne(returns.shift(1).fillna(0))
    df["AlternationRate"] = direction_changes.rolling(20).mean()
    df["GapRate"] = (
        (df["Open"] - df["Close"].shift(1)).abs() / (df["ATR"] + 1e-6)
    ).rolling(20).mean()
    for lag in (1, 5, 10):
        df[f"ReturnAutocorr_{lag}"] = returns.rolling(50).corr(
            returns.shift(lag)
        )
    df["MarketTemperature"] = (
        0.35 * df["RVOL"].clip(0, 3)
        + 0.30 * df["ShockIndex"].clip(0, 3)
        + 0.20 * (df["Kalman_Fast_Slope"].abs() / (df["ATR"] + 1e-6)).clip(0, 3)
        + 0.15 * df["NoiseIndex"].clip(0, 3)
    )

    df['Hour'] = df.index.hour
    df['Minute'] = df.index.minute

    for horizon in (3, 15, 60):
        future_close = df["Close"].shift(-horizon)
        df[f"Return_{horizon}m"] = (future_close - df["Close"]) / (
            df["ATR"] + 1e-6
        )
        df[f"Target_{horizon}m"] = (future_close > df["Close"]).astype("float64")
        df.loc[future_close.isna(), f"Target_{horizon}m"] = np.nan

    move_3m = df["Close"].shift(-3) - df["Close"]
    cost_band = 2 * SPREAD
    df["Target_3m_cost"] = np.select(
        [move_3m > cost_band, move_3m < -cost_band],
        [1, -1],
        default=0,
    ).astype("float64")
    df.loc[move_3m.isna(), "Target_3m_cost"] = np.nan

    df = add_orderflow_features(df)
    df = add_multi_timeframe_features(df)
    df = compute_regime(df)
    df = add_market_quality(df)

    # Keep the latest rows for live inference. They have no future target yet,
    # so model training filters Target_3m separately.
    feature_columns = [
        c for c in df.columns if not c.startswith(("Target_", "Return_"))
    ]
    df.dropna(subset=feature_columns, inplace=True)

    return df


def get_feature_columns(df: pd.DataFrame):
    return [
        c for c in df.columns if not c.startswith(("Target_", "Return_"))
    ]
