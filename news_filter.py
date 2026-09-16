# news_filter.py

import requests
import pandas as pd
from config import FINNHUB_API_KEY, NEWS_LOOKBACK_MINUTES, NEWS_COOLDOWN_MINUTES

FINNHUB_URL = "https://finnhub.io/api/v1/news"

def fetch_forex_news():
    params = {
        "category": "forex",
        "token": FINNHUB_API_KEY
    }
    r = requests.get(FINNHUB_URL, params=params)
    if r.status_code != 200:
        print("News API error:", r.text)
        return pd.DataFrame()

    data = r.json()
    rows = []
    for item in data:
        # Finnhub يعيد datetime كـ timestamp أو string حسب الإعداد
        ts = item.get("datetime") or item.get("time") or None
        if ts is None:
            continue
        t = pd.to_datetime(ts, unit='s', errors='coerce')
        rows.append({
            "Time": t,
            "Headline": item.get("headline", ""),
            "Source": item.get("source", ""),
            "Category": item.get("category", "forex")
        })

    df_news = pd.DataFrame(rows).dropna(subset=["Time"]).set_index("Time").sort_index()
    return df_news

def build_news_blackout(df_prices, df_news):
    """
    تبني فلتر زمني يمنع التداول قبل وبعد الأخبار
    """
    blackout = pd.Series(False, index=df_prices.index)

    for news_time in df_news.index:
        start = news_time - pd.Timedelta(minutes=NEWS_LOOKBACK_MINUTES)
        end = news_time + pd.Timedelta(minutes=NEWS_COOLDOWN_MINUTES)
        mask = (df_prices.index >= start) & (df_prices.index <= end)
        blackout.loc[mask] = True

    return blackout
