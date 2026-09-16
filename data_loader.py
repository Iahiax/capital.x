# data_loader.py
# ملف جلب البيانات كامل – يدعم جلب سنة كاملة بدون أخطاء

import requests
import pandas as pd
import time
from config import API_KEY, API_KEY_PASSWORD, EPIC, RESOLUTION

BASE_URL = "https://api.ig.com/gateway/deal"


def fetch_batch(start, end):
    """
    يجلب دفعة بيانات من API بين start و end
    ويعيد DataFrame يحتوي على Time و OHLC و Volume
    """

    url = f"{BASE_URL}/prices/{EPIC}/{RESOLUTION}"
    headers = {
        "X-IG-API-KEY": API_KEY,
        "Content-Type": "application/json",
        "Accept": "application/json",
        "VERSION": "3"
    }

    params = {
        "from": start,
        "to": end,
        "pageSize": 1000
    }

    r = requests.get(url, headers=headers, params=params)

    # إذا API رجعت خطأ
    if r.status_code != 200:
        print("❌ API Error:", r.text)
        return pd.DataFrame()

    data = r.json()

    if "prices" not in data:
        print("❌ No 'prices' field in API response")
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
    """
    يجلب بيانات سنة كاملة على دفعات صغيرة
    ويتأكد من وجود عمود Time
    """

    print("🚀 جلب بيانات سنة كاملة لزوج", EPIC)

    # نحدد التاريخ الحالي
    end = pd.Timestamp.utcnow()
    start = end - pd.Timedelta(days=365)

    all_data = []

    # نقسم السنة إلى دفعات شهرية
    current = start
    while current < end:
        batch_end = current + pd.Timedelta(days=30)

        print(f"📦 جلب البيانات من {current} إلى {batch_end}")

        df_batch = fetch_batch(
            current.strftime("%Y-%m-%dT%H:%M:%SZ"),
            batch_end.strftime("%Y-%m-%dT%H:%M:%SZ")
        )

        if df_batch.empty:
            print("⚠️ دفعة فارغة – قد يكون API رفض الطلب")
        else:
            all_data.append(df_batch)

        current = batch_end
        time.sleep(0.5)  # منع الضغط على API

    if len(all_data) == 0:
        raise Exception("❌ لم يتم جلب أي بيانات – تحقق من API أو المفاتيح")

    df = pd.concat(all_data, ignore_index=True)

    # التأكد من وجود عمود Time
    if "Time" not in df.columns:
        raise Exception("❌ خطأ: عمود Time غير موجود في البيانات")

    df = df.drop_duplicates().set_index("Time").sort_index()

    print("✅ تم تحميل البيانات بنجاح – عدد الشموع:", len(df))

    return df
