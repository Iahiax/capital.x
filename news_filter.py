import pandas as pd
import requests

from config import FINNHUB_API_KEY, NEWS_COOLDOWN_MINUTES, NEWS_LOOKBACK_MINUTES

FINNHUB_URL = "https://finnhub.io/api/v1/news"

def fetch_forex_news():
    if not FINNHUB_API_KEY:
        return pd.DataFrame(index=pd.DatetimeIndex([], name="Time"))

    params = {"category": "forex", "token": FINNHUB_API_KEY}
    try:
        r = requests.get(FINNHUB_URL, params=params, timeout=20)
    except requests.RequestException:
        return pd.DataFrame(index=pd.DatetimeIndex([], name="Time"))

    if r.status_code != 200:
        return pd.DataFrame(index=pd.DatetimeIndex([], name="Time"))

    try:
        data = r.json()
    except ValueError:
        return pd.DataFrame(index=pd.DatetimeIndex([], name="Time"))
    if not isinstance(data, list):
        return pd.DataFrame(index=pd.DatetimeIndex([], name="Time"))

    rows = []
    for item in data:
        ts = item.get("datetime") or item.get("time")
        if ts is None:
            continue
        if isinstance(ts, (int, float)):
            t = pd.to_datetime(ts, unit="s", errors="coerce", utc=True)
        else:
            t = pd.to_datetime(ts, errors="coerce", utc=True)
        rows.append({"Time": t})

    df_news = pd.DataFrame(rows).dropna().set_index("Time").sort_index()
    return df_news

def build_news_blackout(df_prices, df_news):
    blackout = pd.Series(False, index=df_prices.index)
    if df_news is None or df_news.empty:
        return blackout

    price_index = pd.DatetimeIndex(df_prices.index)
    for news_time in df_news.index:
        news_time = pd.Timestamp(news_time)
        if price_index.tz is not None and news_time.tzinfo is None:
            news_time = news_time.tz_localize(price_index.tz)
        elif price_index.tz is None and news_time.tzinfo is not None:
            news_time = news_time.tz_localize(None)
        start = news_time - pd.Timedelta(minutes=NEWS_LOOKBACK_MINUTES)
        end = news_time + pd.Timedelta(minutes=NEWS_COOLDOWN_MINUTES)
        mask = (price_index >= start) & (price_index <= end)
        blackout.loc[mask] = True
    return blackout
