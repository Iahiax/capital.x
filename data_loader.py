# data_loader.py

import requests
import pandas as pd
import time
from config import (
    API_KEY,
    EMAIL,
    PASSWORD,
    EPIC,
    RESOLUTION,
    USE_DEMO
)

BASE_URL = (
    "https://demo-api-capital.backend-capital.com/api/v1"
    if USE_DEMO else
    "https://api-capital.backend-capital.com/api/v1"
)

def create_session():
    url = f"{BASE_URL}/session"

    headers = {
        "X-CAP-API-KEY": API_KEY,
        "Content-Type": "application/json"
    }

    data = {
        "identifier": EMAIL,
        "password": PASSWORD,
        "encryptedPassword": False
    }

    r = requests.post(url, headers=headers, json=data)

    if r.status_code != 200:
        print("❌ error.invalid.details:", r.text)
        raise Exception("فشل تسجيل الدخول")

    CST = r.headers.get("CST")
    XST = r.headers.get("X-SECURITY-TOKEN")

    if not CST or not XST:
        raise Exception("❌ لم يتم استلام الرموز")

    print("✅ تم تسجيل الدخول بنجاح")
    return CST, XST


def fetch_batch(CST, XST, start, end):
    url = f"{BASE_URL}/prices/{EPIC}/{RESOLUTION}"

    headers = {
        "X-CAP-API-KEY": API_KEY,
        "CST": CST,
        "X-SECURITY-TOKEN": XST
    }

    params = {
        "from": start,
        "to": end,
        "pageSize": 1000
    }

    r = requests.get(url, headers=headers, params=params)

    if r.status_code != 200:
        print("❌ API Error:", r.text)
        return pd.DataFrame()

    data = r.json()

    if "prices" not in data:
        print("⚠️ لا يوجد حقل prices")
        return pd.DataFrame()

    rows = []
    for item in data["prices"]:
        if "snapshotTime" not in item:
            continue

        rows.append({
            "Time": pd.to_datetime(item["snapshotTime"]),
            "Open": item["openPrice"]["bid"],
            "High": item["highPrice"]["bid"],
            "Low": item["lowPrice"]["bid"],
            "Close": item["closePrice"]["bid"],
            "Volume": item.get("lastTradedVolume", 0)
        })

    return pd.DataFrame(rows)


def load_full_year_data():
    print("🚀 بدء جلب بيانات سنة كاملة")

    CST, XST = create_session()

    end = pd.Timestamp.utcnow()
    start = end - pd.Timedelta(days=365)

    all_data = []
    current = start

    while current < end:
        batch_end = current + pd.Timedelta(days=7)

        print(f"📦 جلب البيانات من {current} إلى {batch_end}")

        df_batch = fetch_batch(
            CST, XST,
            current.strftime("%Y-%m-%dT%H:%M:%SZ"),
            batch_end.strftime("%Y-%m-%dT%H:%M:%SZ")
        )

        if not df_batch.empty:
            all_data.append(df_batch)

        current = batch_end
        time.sleep(0.5)

    if not all_data:
        raise Exception("❌ لم يتم جلب أي بيانات")

    df = pd.concat(all_data, ignore_index=True)

    if "Time" not in df.columns:
        raise Exception("❌ عمود Time غير موجود")

    df = df.drop_duplicates().set_index("Time").sort_index()

    print("✅ تم تحميل البيانات – عدد الشموع:", len(df))
    return df
